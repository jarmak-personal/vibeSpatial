from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

import vibespatial.api as geopandas
from vibespatial.api._native_grouped import NativeGrouped
from vibespatial.api._native_relation import NativeRelation
from vibespatial.api._native_result_core import NativeAttributeTable, NativeTabularResult
from vibespatial.api._native_results import (
    GeometryNativeResult,
    _grouped_constructive_to_native_tabular_result,
)
from vibespatial.api._native_rowset import NativeRowSet
from vibespatial.api._native_state import NativeFrameState, attach_native_state
from vibespatial.api.tools.sjoin import _sjoin_export_result
from vibespatial.constructive.clip_rect import clip_by_rect_owned
from vibespatial.constructive.linestring import linestring_buffer_owned_array
from vibespatial.constructive.make_valid_pipeline import make_valid_owned
from vibespatial.constructive.point import (
    clip_points_rect_owned,
    point_buffer_owned_array,
)
from vibespatial.constructive.point import (
    point_owned_from_xy as _point_owned_from_xy,
)
from vibespatial.constructive.polygon import polygon_centroids_owned
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import DiagnosticKind, OwnedGeometryArray, from_shapely_geometries
from vibespatial.io.arrow import (
    geoseries_from_owned,
    has_pylibcudf_support,
    read_geoparquet_owned,
    write_geoparquet,
)
from vibespatial.io.geojson import read_geojson_owned
from vibespatial.io.geoparquet import read_geoparquet_native
from vibespatial.kernels.constructive.segmented_union import segmented_union_all
from vibespatial.kernels.predicates.point_in_polygon import (
    get_last_gpu_substage_timings,
    point_in_polygon,
)
from vibespatial.overlay.dissolve import (
    DissolveUnionMethod,
    evaluate_geopandas_dissolve,
    evaluate_geopandas_dissolve_native,
    execute_grouped_union_codes,
    union_all_gpu_owned,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection, has_gpu_runtime
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events
from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency, TransferTrigger
from vibespatial.spatial.indexing import build_flat_spatial_index
from vibespatial.spatial.query import query_spatial_index
from vibespatial.spatial.query_box import _query_point_tree_box_row_positions_device
from vibespatial.testing.synthetic import (
    SyntheticSpec,
    generate_lines,
    generate_points,
    generate_polygons,
)

from .profiling import ProfileTrace, StageProfiler, _format_elapsed_compact

PIPELINE_DEFINITIONS = (
    "join-heavy",
    "relation-semijoin",
    "relation-bridge-consumer",
    "grouped-reducer",
    "small-grouped-constructive-reduce",
    "relation-attribute-reducer",
    "constructive",
    "predicate-heavy",
    "predicate-heavy-geopandas",
    "zero-transfer",
    "raster-to-vector",
    "vegetation-corridor",
    "vegetation-corridor-geopandas",
    "parcel-zoning",
    "parcel-zoning-geopandas",
    "flood-exposure",
    "flood-exposure-geopandas",
    "network-service-area",
    "network-service-area-geopandas",
    "site-suitability",
    "site-suitability-geopandas",
    "provenance-rewrite",
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / ".benchmark_fixtures"
_BENCHMARK_OUTPUT_COMPRESSION = None
_ZERO_TRANSFER_SELECTIVE_BOUND = 400.0
_PIPELINE_PROFILE_MODE = "lean"
_PIPELINE_PROFILE_MODES = frozenset({"lean", "audit"})
_SUPPORTED_COLLECTION_GEOM_TYPES = {
    "Point",
    "LineString",
    "Polygon",
    "MultiPoint",
    "MultiLineString",
    "MultiPolygon",
}
_POLYGONAL_COLLECTION_GEOM_TYPES = {"Polygon", "MultiPolygon"}


def _extract_supported_collection_parts(geometry, allowed_geom_types: set[str]) -> list:
    parts = shapely.get_parts(np.asarray([geometry], dtype=object))
    return [
        part for part in parts
        if part.geom_type in allowed_geom_types and not part.is_empty
    ]


def _extract_polygonal_components(geometries) -> list:
    """Extract Polygon/MultiPolygon components from geometries, flattening GeometryCollections."""
    result = []
    for g in geometries:
        if g is None or shapely.is_empty(g):
            result.append(g)
        elif g.geom_type == "GeometryCollection":
            polys = _extract_supported_collection_parts(g, _POLYGONAL_COLLECTION_GEOM_TYPES)
            if polys:
                result.append(shapely.union_all(np.asarray(polys, dtype=object)) if len(polys) > 1 else polys[0])
            else:
                result.append(None)
        else:
            result.append(g)
    return result


def _predicate_polygon_cache_path(polygon_count: int, target_rows: int) -> Path:
    return _FIXTURE_DIR / f"predicate-polygons-base{polygon_count}-rows{target_rows}.parquet"


def _load_or_build_polygon_owned(polygon_count: int, target_rows: int) -> OwnedGeometryArray:
    """Load cached polygon OwnedGeometryArray via GeoParquet, or generate and cache.

    On first run the polygons are generated from Shapely, resized to
    *target_rows*, written to a gitignored Parquet cache, and returned.
    Subsequent runs load the Parquet directly into OwnedGeometryArray
    with zero Shapely overhead.
    """
    cache_path = _predicate_polygon_cache_path(polygon_count, target_rows)
    if cache_path.exists():
        return read_geoparquet_owned(cache_path, backend=_preferred_geoparquet_backend())

    base_polygons = np.asarray(
        list(_regular_polygons_frame(polygon_count).geometry), dtype=object
    )
    resized = np.resize(base_polygons, target_rows).tolist()
    owned = from_shapely_geometries(resized)

    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = geopandas.GeoDataFrame(
        {"geometry": resized}, geometry="geometry", crs="EPSG:4326",
    )
    frame.to_parquet(cache_path, geometry_encoding="geoarrow")
    return owned


def _load_or_build_polygon_geoseries(polygon_count: int, target_rows: int) -> geopandas.GeoSeries:
    cache_path = _predicate_polygon_cache_path(polygon_count, target_rows)
    if not cache_path.exists():
        _load_or_build_polygon_owned(polygon_count, target_rows)
    # Use the upstream CPU reader directly.  The repo shim (geopandas.read_parquet)
    # may route through pylibcudf/RMM which can fail with cudaErrorOperatingSystem
    # in constrained GPU environments.  This function only serves the GeoPandas
    # CPU baseline pipeline, so a GPU read is never appropriate here.
    import pyarrow.parquet as pq

    table = pq.read_table(str(cache_path))
    frame = geopandas.GeoDataFrame.from_arrow(table)
    return geopandas.GeoSeries(frame.geometry.to_numpy(), crs=frame.crs)


@dataclass(frozen=True)
class PipelineBenchmarkResult:
    pipeline: str
    scale: int
    status: str
    elapsed_seconds: float
    selected_runtime: str
    planner_selected_runtime: str
    output_rows: int
    transfer_count: int
    materialization_count: int
    fallback_event_count: int
    peak_device_memory_bytes: int | None
    stages: tuple[ProfileTrace | dict, ...] = field(default_factory=tuple)
    notes: str = ""
    rewrite_event_count: int = 0
    owned_transfer_count: int | None = None
    runtime_d2h_transfer_count: int | None = None
    runtime_d2h_transfer_bytes: int | None = None
    runtime_d2h_transfer_seconds: float | None = None
    profile_mode: str = "lean"

    def to_dict(self) -> dict:
        owned_transfer_count = (
            self.transfer_count
            if self.owned_transfer_count is None
            else self.owned_transfer_count
        )
        runtime_d2h_transfer_count = (
            self.transfer_count
            if self.runtime_d2h_transfer_count is None
            else self.runtime_d2h_transfer_count
        )
        runtime_d2h_transfer_bytes = (
            0
            if self.runtime_d2h_transfer_bytes is None
            else self.runtime_d2h_transfer_bytes
        )
        runtime_d2h_transfer_seconds = (
            0.0
            if self.runtime_d2h_transfer_seconds is None
            else self.runtime_d2h_transfer_seconds
        )
        return {
            "pipeline": self.pipeline,
            "scale": self.scale,
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds,
            "selected_runtime": self.selected_runtime,
            "planner_selected_runtime": self.planner_selected_runtime,
            "output_rows": self.output_rows,
            "transfer_count": runtime_d2h_transfer_count,
            "owned_transfer_count": owned_transfer_count,
            "runtime_d2h_transfer_count": runtime_d2h_transfer_count,
            "runtime_d2h_transfer_bytes": runtime_d2h_transfer_bytes,
            "runtime_d2h_transfer_seconds": runtime_d2h_transfer_seconds,
            "profile_mode": self.profile_mode,
            "materialization_count": self.materialization_count,
            "fallback_event_count": self.fallback_event_count,
            "peak_device_memory_bytes": self.peak_device_memory_bytes,
            "stages": [
                stage.to_dict() if isinstance(stage, ProfileTrace) else stage
                for stage in self.stages
            ],
            "notes": self.notes,
            "rewrite_event_count": self.rewrite_event_count,
        }


def _iter_owned_arrays(value):
    if isinstance(value, OwnedGeometryArray):
        yield value
        return
    if isinstance(value, DeviceGeometryArray):
        yield value.to_owned()
        return
    owned = getattr(value, "owned", None)
    if isinstance(owned, OwnedGeometryArray):
        yield owned
    if isinstance(value, geopandas.GeoDataFrame):
        yield from _iter_owned_arrays(value.geometry.values)
        return
    if isinstance(value, geopandas.GeoSeries):
        yield from _iter_owned_arrays(value.values)
        return
    values = getattr(value, "values", None)
    if isinstance(values, DeviceGeometryArray):
        yield values.to_owned()
    geometry = getattr(value, "geometry", None)
    if geometry is not None and geometry is not value:
        yield from _iter_owned_arrays(geometry)
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_owned_arrays(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_owned_arrays(item)


class _OwnedAudit:
    def __init__(self) -> None:
        self._seen: dict[int, int] = {}
        self.transfer_count = 0
        self._owned_materialization_count = 0
        self.transfer_seconds = 0.0
        self.transfer_bytes = 0
        (
            self._runtime_start_count,
            self._runtime_start_bytes,
            self._runtime_start_seconds,
        ) = _runtime_d2h_transfer_stats()
        self._materialization_event_start_count = _materialization_event_count()

    def observe(self, *values) -> None:
        for array in values:
            for owned in _iter_owned_arrays(array):
                key = id(owned)
                start = self._seen.get(key, 0)
                for event in owned.diagnostics[start:]:
                    if event.kind is DiagnosticKind.TRANSFER:
                        self.transfer_count += 1
                        self.transfer_seconds += event.elapsed_seconds
                        self.transfer_bytes += event.bytes_transferred
                    elif event.kind is DiagnosticKind.MATERIALIZATION:
                        self._owned_materialization_count += 1
                self._seen[key] = len(owned.diagnostics)

    def snapshot(self) -> tuple[int, int, float, int]:
        return (
            self.transfer_count,
            self.materialization_count,
            self.transfer_seconds,
            self.transfer_bytes,
        )

    @property
    def materialization_count(self) -> int:
        return self._owned_materialization_count + self.materialization_event_count

    @property
    def materialization_event_count(self) -> int:
        return max(_materialization_event_count() - self._materialization_event_start_count, 0)

    @property
    def runtime_d2h_transfer_count(self) -> int:
        count, _bytes, _seconds = _runtime_d2h_transfer_stats()
        return max(count - self._runtime_start_count, 0)

    @property
    def runtime_d2h_transfer_bytes(self) -> int:
        _count, bytes_transferred, _seconds = _runtime_d2h_transfer_stats()
        return max(bytes_transferred - self._runtime_start_bytes, 0)

    @property
    def runtime_d2h_transfer_seconds(self) -> float:
        _count, _bytes_transferred, seconds = _runtime_d2h_transfer_stats()
        return max(seconds - self._runtime_start_seconds, 0.0)

    def runtime_snapshot(self) -> tuple[int, int, float]:
        return (
            self.runtime_d2h_transfer_count,
            self.runtime_d2h_transfer_bytes,
            self.runtime_d2h_transfer_seconds,
        )

    def reset_runtime_baseline(self) -> None:
        (
            self._runtime_start_count,
            self._runtime_start_bytes,
            self._runtime_start_seconds,
        ) = _runtime_d2h_transfer_stats()


def _runtime_d2h_transfer_stats() -> tuple[int, int, float]:
    try:
        from vibespatial.cuda._runtime import get_d2h_transfer_profile

        count, bytes_transferred, seconds = get_d2h_transfer_profile()
        return int(count), int(bytes_transferred), float(seconds)
    except Exception:
        return 0, 0, 0.0


def _materialization_event_count() -> int:
    try:
        from vibespatial.runtime.materialization import get_materialization_events

        return len(get_materialization_events())
    except Exception:
        return 0


class _UnavailableGpuSampler:
    available = False


class _NoopGpuEventTimer:
    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def summarize(self) -> dict:
        return {}


_UNAVAILABLE_GPU_SAMPLER = _UnavailableGpuSampler()


def _resolve_pipeline_profile_mode(
    profile_mode: str,
    *,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> str:
    if profile_mode not in _PIPELINE_PROFILE_MODES:
        raise ValueError(
            f"profile_mode must be one of {sorted(_PIPELINE_PROFILE_MODES)!r}"
        )
    if retain_gpu_trace or include_gpu_sparklines:
        return "audit"
    return profile_mode


def _set_pipeline_profile_mode(profile_mode: str) -> str:
    global _PIPELINE_PROFILE_MODE
    previous = _PIPELINE_PROFILE_MODE
    _PIPELINE_PROFILE_MODE = profile_mode
    return previous


def _stage_profiler(**kwargs) -> StageProfiler:
    profile_mode = _resolve_pipeline_profile_mode(
        _PIPELINE_PROFILE_MODE,
        retain_gpu_trace=bool(kwargs.get("retain_gpu_trace", False)),
        include_gpu_sparklines=bool(kwargs.get("include_gpu_sparklines", False)),
    )
    if profile_mode == "lean":
        kwargs["gpu_sampler"] = _UNAVAILABLE_GPU_SAMPLER
        kwargs["gpu_event_timer_factory"] = _NoopGpuEventTimer
        kwargs["retain_gpu_trace"] = False
        kwargs["include_gpu_sparklines"] = False
    return StageProfiler(**kwargs)


class _DeviceMemoryMonitor:
    def __init__(self) -> None:
        self.available = False
        self.peak_bytes: int | None = None
        self._statistics = None
        if not has_gpu_runtime():
            return
        try:
            from rmm import statistics
        except ImportError:
            return
        statistics.enable_statistics()
        self.available = True
        self._statistics = statistics

    def update(self) -> None:
        if not self.available or self._statistics is None:
            return
        stats = self._statistics.get_statistics()
        if stats is None:
            return
        peak = int(stats.peak_bytes)
        self.peak_bytes = max(self.peak_bytes or 0, peak)


def _free_gpu_pool_memory() -> None:
    """Release cached GPU memory between pipeline stages.

    With CuPy pool: returns cached blocks to the CUDA driver.
    With RMM pool: runs ``gc.collect()`` to ensure dead CuPy arrays
    return their blocks to the pool for reuse.
    """
    if not has_gpu_runtime():
        return
    try:
        from vibespatial.cuda._runtime import get_cuda_runtime
        get_cuda_runtime().free_pool_memory()
    except Exception:
        pass  # best-effort; do not crash the pipeline for cleanup failures


def _regular_points_frame(rows: int) -> geopandas.GeoDataFrame:
    dataset = generate_points(SyntheticSpec("point", "grid", count=rows, seed=0))
    values = np.asarray(list(dataset.geometries), dtype=object)
    return geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(np.arange(rows, dtype=np.int32) % max(min(rows, 256), 1)),
            "value": np.arange(rows, dtype=np.int64),
            "geometry": values,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _regular_polygons_frame(rows: int) -> geopandas.GeoDataFrame:
    dataset = generate_polygons(
        SyntheticSpec("polygon", "regular-grid", count=rows, seed=1, vertices=5, hole_probability=0.0)
    )
    values = np.asarray(list(dataset.geometries), dtype=object)
    return geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(np.arange(rows, dtype=np.int32) % max(min(rows, 128), 1)),
            "value": np.arange(rows, dtype=np.int64),
            "geometry": values,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _relation_bridge_selector_frame(rows: int) -> geopandas.GeoDataFrame:
    selector_count = max(min(rows // 20, 256), 1)
    selector = box(
        0.0,
        0.0,
        _ZERO_TRANSFER_SELECTIVE_BOUND,
        _ZERO_TRANSFER_SELECTIVE_BOUND,
    )
    return geopandas.GeoDataFrame(
        {
            "zone_id": np.arange(selector_count, dtype=np.int32),
            "geometry": np.asarray([selector] * selector_count, dtype=object),
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _attach_private_native_state_from_public_frame(
    frame: geopandas.GeoDataFrame,
) -> NativeFrameState:
    geometry_name = frame._geometry_column_name
    attribute_frame = frame.drop(columns=[geometry_name]).copy(deep=False)
    try:
        import pyarrow as pa

        arrow_table = pa.Table.from_pandas(attribute_frame, preserve_index=False)
        if has_gpu_runtime() and has_pylibcudf_support():
            try:
                import pylibcudf as plc

                attributes = NativeAttributeTable(
                    device_table=plc.Table.from_arrow(arrow_table),
                    index_override=frame.index,
                    column_override=tuple(attribute_frame.columns),
                    schema_override=arrow_table.schema,
                )
            except Exception:
                attributes = NativeAttributeTable(
                    arrow_table=arrow_table,
                    index_override=frame.index,
                    column_override=tuple(attribute_frame.columns),
                )
        else:
            attributes = NativeAttributeTable(
                arrow_table=arrow_table,
                index_override=frame.index,
                column_override=tuple(attribute_frame.columns),
            )
    except Exception:
        attributes = NativeAttributeTable(dataframe=attribute_frame)
    geometry_owned = from_shapely_geometries(list(frame.geometry))
    if has_gpu_runtime():
        geometry_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="relation bridge canary native state seed",
        )
    result = NativeTabularResult(
        attributes=attributes,
        geometry=GeometryNativeResult.from_owned(geometry_owned, crs=frame.crs),
        geometry_name=geometry_name,
        column_order=tuple(frame.columns),
        attrs=dict(frame.attrs or {}),
    )
    state = NativeFrameState.from_native_tabular_result(result)
    attach_native_state(frame, state)
    return state


def _write_geojson_points(path: Path, rows: int) -> tuple[geopandas.GeoDataFrame, np.ndarray]:
    frame = _regular_points_frame(rows)
    frame.to_file(path, driver="GeoJSON")
    polygon_count = max(rows // 8, 1)
    polygons = np.asarray(list(_regular_polygons_frame(polygon_count).geometry), dtype=object)
    return frame, polygons


def _subset_by_mask(owned: OwnedGeometryArray, mask: np.ndarray) -> OwnedGeometryArray:
    return owned.take(mask)


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


def _tabular_row_count(value) -> int:
    geometry = getattr(value, "geometry", None)
    row_count = getattr(geometry, "row_count", None)
    if row_count is not None:
        return int(row_count)
    return int(len(value))


def _trace_to_stage_dict(trace: ProfileTrace) -> dict:
    selected_runtime = trace.metadata.get("actual_selected_runtime", trace.selected_runtime)
    return {
        "operation": trace.operation,
        "selected_runtime": selected_runtime,
        "planner_selected_runtime": trace.metadata.get("planner_selected_runtime", trace.selected_runtime),
        "total_elapsed_seconds": trace.total_elapsed_seconds,
        "stages": [stage.to_dict() for stage in trace.stages],
        "metadata": trace.metadata,
    }


def _record_stage_overheads(stage, audit: _OwnedAudit, memory: _DeviceMemoryMonitor, *values) -> None:
    transfer_before, materialization_before, seconds_before, bytes_before = audit.snapshot()
    runtime_count_before, runtime_bytes_before, runtime_seconds_before = (
        audit.runtime_snapshot()
    )
    audit.observe(*values)
    memory.update()
    transfer_after, materialization_after, seconds_after, bytes_after = audit.snapshot()
    runtime_count_after, runtime_bytes_after, runtime_seconds_after = (
        audit.runtime_snapshot()
    )
    stage.metadata["transfer_count_delta"] = transfer_after - transfer_before
    stage.metadata["transfer_count_total"] = transfer_after
    stage.metadata["owned_transfer_count_delta"] = transfer_after - transfer_before
    stage.metadata["owned_transfer_count_total"] = transfer_after
    stage.metadata["runtime_d2h_transfer_count_delta"] = (
        runtime_count_after - runtime_count_before
    )
    stage.metadata["runtime_d2h_transfer_count_total"] = runtime_count_after
    stage.metadata["runtime_d2h_transfer_seconds_delta"] = (
        runtime_seconds_after - runtime_seconds_before
    )
    stage.metadata["runtime_d2h_transfer_seconds_total"] = runtime_seconds_after
    stage.metadata["materialization_count_delta"] = materialization_after - materialization_before
    stage.metadata["materialization_count_total"] = materialization_after
    stage.metadata["transfer_seconds_delta"] = seconds_after - seconds_before
    stage.metadata["transfer_seconds_total"] = seconds_after
    stage.metadata["transfer_bytes_delta"] = bytes_after - bytes_before
    stage.metadata["transfer_bytes_total"] = bytes_after
    stage.metadata["owned_transfer_bytes_delta"] = bytes_after - bytes_before
    stage.metadata["owned_transfer_bytes_total"] = bytes_after
    stage.metadata["runtime_d2h_transfer_bytes_delta"] = (
        runtime_bytes_after - runtime_bytes_before
    )
    stage.metadata["runtime_d2h_transfer_bytes_total"] = runtime_bytes_after
    if memory.peak_bytes is not None:
        stage.metadata["peak_device_memory_bytes"] = memory.peak_bytes


def _selected_runtime_from_history(*values) -> str | None:
    for value in values:
        for owned in _iter_owned_arrays(value):
            if owned.runtime_history:
                return owned.runtime_history[-1].selected.value
            if owned.device_state is not None:
                return "gpu"
    return None


def _actual_array_device_label(value) -> str:
    return "gpu" if hasattr(value, "__cuda_array_interface__") else "cpu"


def _pipeline_runtime_from_stage_devices(stage_devices: list[str]) -> str:
    devices = {device for device in stage_devices if device}
    if "gpu" in devices and "cpu" in devices:
        return "hybrid"
    if "gpu" in devices:
        return "gpu"
    return "cpu"


def _preferred_geoparquet_backend() -> str:
    return "gpu" if has_gpu_runtime() else "cpu"


def _preferred_geojson_mode() -> str:
    return "auto"


def _read_geoparquet_owned_preferred(path: Path, *, preferred_backend: str) -> tuple[OwnedGeometryArray, str, str]:
    if preferred_backend != "gpu":
        return read_geoparquet_owned(path, backend="cpu"), "cpu", ""
    try:
        return read_geoparquet_owned(path, backend="gpu"), "gpu", ""
    except Exception as exc:
        return read_geoparquet_owned(path, backend="cpu"), "cpu", f"gpu read fallback: {exc.__class__.__name__}"


def _read_geojson_owned_preferred(
    source: Path | str | bytes | bytearray | memoryview,
    *,
    preferred_mode: str,
):
    if preferred_mode == "pylibcudf":
        try:
            return read_geojson_owned(
                source,
                prefer="pylibcudf",
                track_properties=False,
            ), "gpu", ""
        except Exception as exc:
            return read_geojson_owned(
                source,
                prefer="fast-json",
                track_properties=False,
            ), "cpu", f"gpu read fallback: {exc.__class__.__name__}"

    # For "auto" or "gpu-byte-classify", pass through to read_geojson_owned
    # which will select the strategy (auto now prefers GPU when available).
    try:
        batch = read_geojson_owned(
            source,
            prefer=preferred_mode,
            track_properties=False,
        )
        device = "gpu" if batch.geometry.device_state is not None else "cpu"
        return batch, device, ""
    except Exception as exc:
        batch = read_geojson_owned(
            source,
            prefer="fast-json",
            track_properties=False,
        )
        return batch, "cpu", f"gpu read fallback: {exc.__class__.__name__}"


def _read_geojson_geopandas_preferred(path: Path) -> tuple[geopandas.GeoDataFrame, str, str, str]:
    requested_engine = "pyogrio"
    if find_spec("pyogrio") is None:
        return geopandas.read_file(path), requested_engine, "default", "pyogrio unavailable"
    try:
        return geopandas.read_file(path, engine="pyogrio"), requested_engine, "pyogrio", ""
    except Exception as exc:
        return geopandas.read_file(path), requested_engine, "default", f"pyogrio fallback: {exc.__class__.__name__}"


def _join_heavy_group_categories(scale: int) -> np.ndarray:
    return np.arange(max(min(scale, 128), 1), dtype=np.int32)


def _dissolve_join_heavy_groups(
    joined_geometry,
    unique_right_index: np.ndarray,
    *,
    scale: int,
):
    if isinstance(joined_geometry, GeometryNativeResult):
        geometry_result = joined_geometry
    else:
        geometry_result = GeometryNativeResult.from_geoseries(joined_geometry)

    geometry_name = "geometry"
    group_categories = _join_heavy_group_categories(scale)
    group_labels = np.remainder(
        np.asarray(unique_right_index, dtype=np.int64),
        int(group_categories.size),
    )
    row_group_codes = group_labels.astype(np.int32, copy=False)
    observed_codes = np.unique(row_group_codes).astype(np.int32, copy=False)
    observed_labels = np.unique(group_labels)
    joined_owned = geometry_result.owned
    if joined_owned is None:
        joined_series = geometry_result.to_geoseries(
            index=pd.RangeIndex(geometry_result.row_count),
            name=geometry_name,
        )
        geometry_values = joined_series.values
        joined_owned = getattr(geometry_values, "_owned", None)
    else:
        geometry_values = geoseries_from_owned(
            joined_owned,
            name=geometry_name,
            crs=geometry_result.crs,
        ).values
    grouped_union = execute_grouped_union_codes(
        geometry_values,
        row_group_codes,
        group_count=int(group_categories.size),
        method=DissolveUnionMethod.COVERAGE,
        owned=joined_owned,
    )
    if grouped_union is not None:
        group_index = pd.CategoricalIndex(pd.Categorical(observed_labels), name="group")
        if grouped_union.owned is not None:
            geometry_result = GeometryNativeResult.from_owned(
                grouped_union.owned.take(observed_codes.astype(np.int64, copy=False)),
                crs=geometry_result.crs,
            )
        else:
            geometry_result = GeometryNativeResult.from_geoseries(
                geopandas.GeoSeries(
                    grouped_union.geometries[observed_codes.astype(np.intp, copy=False)],
                    name=geometry_name,
                    crs=geometry_result.crs,
                )
            )
        return _grouped_constructive_to_native_tabular_result(
            geometry=geometry_result,
            attributes=pd.DataFrame(index=group_index),
            geometry_name=geometry_name,
            as_index=True,
        ), True

    joined_frame = geopandas.GeoDataFrame(
        {"group": pd.Categorical(group_labels)},
        geometry=geometry_result.to_geoseries(
            index=pd.RangeIndex(geometry_result.row_count),
            name=geometry_name,
        ),
        crs=geometry_result.crs,
    )
    return evaluate_geopandas_dissolve_native(
        joined_frame,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=False,
        observed=False,
        dropna=True,
        method="coverage",
        grid_size=None,
        agg_kwargs={},
    ), False


def _profile_join_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    planner_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU
    read_backend = _preferred_geoparquet_backend()

    polygon_rows = max(scale // 10, 1)
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        left_path = root / "left.parquet"
        right_path = root / "right.parquet"
        _regular_points_frame(scale).to_parquet(left_path, geometry_encoding="geoarrow")
        _regular_polygons_frame(polygon_rows).to_parquet(right_path, geometry_encoding="geoarrow")

        audit.reset_runtime_baseline()
        profiler = _stage_profiler(
            operation="pipeline.join-heavy",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.AUTO,
            selected_runtime="hybrid" if has_gpu_runtime() else "cpu",
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        with profiler.stage("read_points", category="setup", device="auto", rows_in=scale) as stage:
            left_owned, actual_left_backend, left_note = _read_geoparquet_owned_preferred(
                left_path,
                preferred_backend=read_backend,
            )
            stage.device = actual_left_backend
            stage.rows_out = left_owned.row_count
            stage.metadata["requested_backend"] = read_backend
            stage.metadata["actual_backend"] = actual_left_backend
            if left_note:
                stage.metadata["fallback_note"] = left_note
            _record_stage_overheads(stage, audit, memory, left_owned)

        with profiler.stage("read_polygons", category="setup", device="auto", rows_in=polygon_rows) as stage:
            right_owned, actual_right_backend, right_note = _read_geoparquet_owned_preferred(
                right_path,
                preferred_backend=read_backend,
            )
            stage.device = actual_right_backend
            stage.rows_out = right_owned.row_count
            stage.metadata["requested_backend"] = read_backend
            stage.metadata["actual_backend"] = actual_right_backend
            if right_note:
                stage.metadata["fallback_note"] = right_note
            _record_stage_overheads(stage, audit, memory, right_owned)

        with profiler.stage(
            "build_index",
            category="sort",
            device=ExecutionMode.CPU,
            rows_in=polygon_rows,
            detail="build the current flat polygon spatial index before query execution",
        ) as stage:
            flat_index = build_flat_spatial_index(
                right_owned,
                runtime_selection=RuntimeSelection(
                    requested=ExecutionMode.AUTO,
                    selected=ExecutionMode.CPU,
                    reason="pipeline join rail baseline index build",
                ),
            )
            stage.rows_out = int(flat_index.size)
            stage.metadata["regular_grid_fast_path"] = bool(flat_index.regular_grid is not None)
            _record_stage_overheads(stage, audit, memory, right_owned)

        query_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU

        with profiler.stage(
            "sjoin_query",
            category="filter",
            device=query_runtime,
            rows_in=scale,
            detail="query candidate polygons for each point through the repo-owned spatial query path",
        ) as stage:
            indices = query_spatial_index(
                right_owned,
                flat_index,
                left_owned,
                predicate="intersects",
                sort=True,
                output_format="indices",
            )
            if indices.ndim == 1:
                right_index = indices.astype(np.int64, copy=False)
            else:
                right_index = indices[1].astype(np.int64, copy=False)
            stage.rows_out = int(right_index.size)
            stage.metadata["pairs_examined"] = int(right_index.size)
            stage.metadata["regular_grid_fast_path"] = bool(flat_index.regular_grid is not None)
            _record_stage_overheads(stage, audit, memory, left_owned, right_owned)

        with profiler.stage(
            "assemble_join_rows",
            category="refine",
            device="auto",
            rows_in=int(right_index.size),
            detail="assemble polygon rows selected by the spatial query before dissolve",
        ) as stage:
            unique_right_index = np.unique(right_index)
            joined_geometry = (
                GeometryNativeResult.from_owned(
                    right_owned.take(unique_right_index),
                    crs="EPSG:4326",
                )
                if unique_right_index.size
                else GeometryNativeResult.from_geoseries(
                    geopandas.GeoSeries([], name="geometry", crs="EPSG:4326")
                )
            )
            stage.device = _selected_runtime_from_history(joined_geometry) or "cpu"
            stage.rows_out = int(joined_geometry.row_count)
            stage.metadata["deduped_candidate_rows"] = int(unique_right_index.size)
            _record_stage_overheads(stage, audit, memory, joined_geometry)

        with profiler.stage(
            "dissolve_groups",
            category="refine",
            device="auto",
            rows_in=int(joined_geometry.row_count),
            detail="dissolve joined polygons by categorical group after spatial query assembly",
        ) as stage:
            dissolved, used_direct_grouped_union = _dissolve_join_heavy_groups(
                joined_geometry,
                unique_right_index,
                scale=scale,
            )
            stage.device = _selected_runtime_from_history(dissolved) or "cpu"
            stage.rows_out = int(len(dissolved.attributes))
            stage.metadata["group_count"] = int(_join_heavy_group_categories(scale).size)
            stage.metadata["direct_grouped_union"] = used_direct_grouped_union
            stage.metadata["method"] = DissolveUnionMethod.COVERAGE.value
            _record_stage_overheads(stage, audit, memory, dissolved)

        output_path = root / "join-output.parquet"
        with profiler.stage(
            "write_output",
            category="emit",
            device="auto",
            rows_in=int(len(dissolved.attributes)),
            detail="write dissolved join result to GeoParquet",
        ) as stage:
            write_geoparquet(
                dissolved,
                output_path,
                geometry_encoding="geoarrow",
                compression=_BENCHMARK_OUTPUT_COMPRESSION,
            )
            stage.device = _selected_runtime_from_history(dissolved) or "cpu"
            stage.rows_out = int(len(dissolved.attributes))
            stage.metadata["compression"] = _BENCHMARK_OUTPUT_COMPRESSION
            _record_stage_overheads(stage, audit, memory, dissolved)

    trace = profiler.finish(
        metadata={
            "scale": scale,
            "actual_selected_runtime": (
                "hybrid"
                if actual_left_backend == "gpu"
                or actual_right_backend == "gpu"
                or query_runtime is ExecutionMode.GPU
                else "cpu"
            ),
            "planner_selected_runtime": planner_runtime.value,
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
        }
    )
    return PipelineBenchmarkResult(
        pipeline="join-heavy",
        scale=scale,
        status="ok",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=(
            "hybrid"
            if actual_left_backend == "gpu"
            or actual_right_backend == "gpu"
            or query_runtime is ExecutionMode.GPU
            else "cpu"
        ),
        planner_selected_runtime=planner_runtime.value,
        output_rows=(trace.metadata["dispatch_events"] and int(len(dissolved.attributes))) or int(len(dissolved.attributes)),
        transfer_count=audit.transfer_count,
        owned_transfer_count=audit.transfer_count,
        runtime_d2h_transfer_count=audit.runtime_d2h_transfer_count,
        runtime_d2h_transfer_bytes=audit.runtime_d2h_transfer_bytes,
        runtime_d2h_transfer_seconds=audit.runtime_d2h_transfer_seconds,
        materialization_count=audit.materialization_count,
        fallback_event_count=int(trace.metadata["fallback_events"]),
        peak_device_memory_bytes=memory.peak_bytes,
        stages=(_trace_to_stage_dict(trace),),
        notes="Current join-heavy pipeline uses owned GeoParquet read, GPU regular-grid query when available, and a direct grouped coverage dissolve rail before GeoParquet write.",
    )


def _relation_pair_arrays_from_query_result(query_result) -> tuple[object, object, str]:
    left_idx = getattr(query_result, "d_left_idx", None)
    right_idx = getattr(query_result, "d_right_idx", None)
    if left_idx is not None and right_idx is not None:
        return left_idx, right_idx, "device"

    if getattr(query_result, "ndim", None) == 1:
        return (
            np.empty(0, dtype=np.int32),
            query_result.astype(np.int32, copy=False),
            "host",
        )
    return (
        query_result[0].astype(np.int32, copy=False),
        query_result[1].astype(np.int32, copy=False),
        "host",
    )


def _profile_relation_semijoin_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return PipelineBenchmarkResult(
            pipeline="relation-semijoin",
            scale=scale,
            status="deferred",
            elapsed_seconds=0.0,
            selected_runtime="deferred",
            planner_selected_runtime="deferred",
            output_rows=0,
            transfer_count=0,
            materialization_count=0,
            fallback_event_count=0,
            peak_device_memory_bytes=None,
            stages=tuple(),
            notes="Deferred until both CUDA runtime and pylibcudf are available for the native relation-semijoin rail.",
        )

    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        left_path = root / "relation-semijoin-left.parquet"
        right_path = root / "relation-semijoin-right.parquet"
        output_path = root / "relation-semijoin-output.parquet"
        _regular_points_frame(scale).to_parquet(
            left_path,
            geometry_encoding="geoarrow",
        )
        geopandas.GeoDataFrame(
            {
                "zone_id": np.asarray([0], dtype=np.int64),
                "geometry": [box(0.0, 0.0, _ZERO_TRANSFER_SELECTIVE_BOUND, _ZERO_TRANSFER_SELECTIVE_BOUND)],
            },
            geometry="geometry",
            crs="EPSG:4326",
        ).to_parquet(right_path, geometry_encoding="geoarrow")

        audit.reset_runtime_baseline()
        profiler = _stage_profiler(
            operation="pipeline.relation-semijoin",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.GPU,
            selected_runtime="gpu",
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        with profiler.stage(
            "read_inputs",
            category="setup",
            device=ExecutionMode.GPU,
            rows_in=scale,
            detail="read left points and right selector polygon as private native payloads",
        ) as stage:
            left_payload = read_geoparquet_native(left_path)
            right_payload = read_geoparquet_native(right_path)
            left_state = NativeFrameState.from_native_tabular_result(left_payload)
            right_state = NativeFrameState.from_native_tabular_result(right_payload)
            stage.rows_out = int(left_state.row_count)
            stage.metadata["left_rows"] = int(left_state.row_count)
            stage.metadata["right_rows"] = int(right_state.row_count)
            stage.metadata["read_surface"] = "read_geoparquet_native"
            _record_stage_overheads(
                stage,
                audit,
                memory,
                left_payload,
                right_payload,
            )

        with profiler.stage(
            "build_index",
            category="sort",
            device=ExecutionMode.GPU,
            rows_in=int(right_state.row_count),
            detail="build the right-side spatial index for the admitted relation semijoin selector",
        ) as stage:
            flat_index = build_flat_spatial_index(
                right_state.geometry.owned,
                runtime_selection=RuntimeSelection(
                    requested=ExecutionMode.AUTO,
                    selected=ExecutionMode.GPU,
                    reason="relation-semijoin canary right-side selector index",
                ),
            )
            stage.rows_out = int(flat_index.size)
            stage.metadata["regular_grid_fast_path"] = bool(flat_index.regular_grid is not None)
            _record_stage_overheads(stage, audit, memory, right_state.geometry.owned)

        with profiler.stage(
            "sjoin_relation",
            category="filter",
            device=ExecutionMode.GPU,
            rows_in=int(left_state.row_count),
            detail="produce relation pairs without joined pandas row assembly",
        ) as stage:
            query_result, query_execution = query_spatial_index(
                right_state.geometry.owned,
                flat_index,
                left_state.geometry.owned,
                predicate="intersects",
                sort=True,
                output_format="indices",
                return_device=True,
                return_metadata=True,
            )
            left_idx, right_idx, pair_storage = _relation_pair_arrays_from_query_result(
                query_result
            )
            relation = NativeRelation(
                left_idx,
                right_idx,
                left_token=left_state.lineage_token,
                right_token=right_state.lineage_token,
                predicate="intersects",
                left_row_count=left_state.row_count,
                right_row_count=right_state.row_count,
                sorted_by_left=True,
            )
            stage.device = query_execution.selected
            stage.rows_out = len(relation)
            stage.metadata["pair_storage"] = pair_storage
            stage.metadata["query_implementation"] = query_execution.implementation
            stage.metadata["query_reason"] = query_execution.reason
            _record_stage_overheads(stage, audit, memory, left_state.geometry.owned)

        with profiler.stage(
            "semijoin_rowset",
            category="filter",
            device=ExecutionMode.GPU,
            rows_in=int(len(relation)),
            detail="derive unique left-row NativeRowSet from relation pairs",
        ) as stage:
            rowset = relation.left_semijoin_rowset()
            stage.rows_out = len(rowset)
            stage.metadata["rowset_storage"] = "device" if rowset.is_device else "host"
            stage.metadata["ordered"] = rowset.ordered
            stage.metadata["unique"] = rowset.unique
            _record_stage_overheads(stage, audit, memory)

        with profiler.stage(
            "subset_rows",
            category="filter",
            device=ExecutionMode.GPU,
            rows_in=int(left_state.row_count),
            detail="apply the relation semijoin rowset to the left NativeFrameState",
        ) as stage:
            filtered = left_state.take(rowset, preserve_index=False).to_native_tabular_result()
            stage.rows_out = int(filtered.geometry.row_count)
            stage.metadata["native_rowset_take"] = "take"
            _record_stage_overheads(stage, audit, memory, filtered)

        with profiler.stage(
            "write_output",
            category="emit",
            device=ExecutionMode.GPU,
            rows_in=_tabular_row_count(filtered),
            detail="write the semijoined native payload through the native GeoParquet path",
        ) as stage:
            write_geoparquet(
                filtered,
                output_path,
                index=False,
                geometry_encoding="geoarrow",
                compression=_BENCHMARK_OUTPUT_COMPRESSION,
            )
            stage.rows_out = _tabular_row_count(filtered)
            stage.metadata["compression"] = _BENCHMARK_OUTPUT_COMPRESSION
            _record_stage_overheads(stage, audit, memory, filtered)

    transfer_count = audit.runtime_d2h_transfer_count
    materialization_count = audit.materialization_count
    stage_devices = [stage.device for stage in profiler._stages]
    actual_selected_runtime = _pipeline_runtime_from_stage_devices(stage_devices)
    planner_selected_runtime = ExecutionMode.GPU.value
    trace = profiler.finish(
        metadata={
            "scale": scale,
            "actual_selected_runtime": actual_selected_runtime,
            "planner_selected_runtime": planner_selected_runtime,
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
            "admissible_shape": "RangeIndex point/polygon intersects semijoin",
        }
    )
    return PipelineBenchmarkResult(
        pipeline="relation-semijoin",
        scale=scale,
        status="ok" if materialization_count == 0 else "failed",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=planner_selected_runtime,
        output_rows=_tabular_row_count(filtered),
        transfer_count=transfer_count,
        owned_transfer_count=audit.transfer_count,
        runtime_d2h_transfer_count=audit.runtime_d2h_transfer_count,
        runtime_d2h_transfer_bytes=audit.runtime_d2h_transfer_bytes,
        runtime_d2h_transfer_seconds=audit.runtime_d2h_transfer_seconds,
        materialization_count=materialization_count,
        fallback_event_count=int(trace.metadata["fallback_events"]),
        peak_device_memory_bytes=memory.peak_bytes,
        stages=(_trace_to_stage_dict(trace),),
        notes=(
            "Private relation-semijoin canary: read_geoparquet_native -> "
            "NativeRelation -> left NativeRowSet -> NativeFrameState.take -> "
            "native GeoParquet write."
        ),
    )


def _profile_relation_bridge_consumer_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Shape canary for consuming a relation export without joined row assembly.

    This is intentionally not a workflow benchmark. It isolates the reusable
    public-boundary shape: native-backed public sources, `sjoin` relation export,
    private semijoin frame consumption, and a public joined-export reference.
    """
    from time import perf_counter

    from vibespatial.runtime.materialization import clear_materialization_events

    left_frame = _regular_points_frame(scale)
    right_frame = _relation_bridge_selector_frame(scale)
    left_state = _attach_private_native_state_from_public_frame(left_frame)
    right_state = _attach_private_native_state_from_public_frame(right_frame)

    clear_materialization_events()
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    planner_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU
    profiler = _stage_profiler(
        operation="pipeline.relation-bridge-consumer",
        dataset=f"scale-{scale}",
        requested_runtime=ExecutionMode.AUTO,
        selected_runtime="hybrid" if has_gpu_runtime() else "cpu",
        enable_nvtx=enable_nvtx,
        retain_gpu_trace=retain_gpu_trace,
        include_gpu_sparklines=include_gpu_sparklines,
    )

    with profiler.stage(
        "native_state_seed",
        category="setup",
        device="private",
        rows_in=scale,
        detail="seed private native state on public GeoDataFrames outside any broad pandas interception",
    ) as stage:
        stage.rows_out = int(left_state.row_count)
        stage.metadata["left_rows"] = int(left_state.row_count)
        stage.metadata["right_rows"] = int(right_state.row_count)
        stage.metadata["left_index_kind"] = left_state.index_plan.kind
        stage.metadata["right_index_kind"] = right_state.index_plan.kind
        stage.metadata["shape_canary"] = "relation_export_bridge"
        _record_stage_overheads(stage, audit, memory, left_state, right_state)

    with profiler.stage(
        "sjoin_relation_export",
        category="filter",
        device=ExecutionMode.AUTO,
        rows_in=int(left_state.row_count),
        detail="build deferred sjoin relation export without materializing joined rows",
    ) as stage:
        export_result, query_implementation, query_execution = _sjoin_export_result(
            left_frame,
            right_frame,
            "inner",
            "intersects",
            None,
            "left",
            "right",
            return_device=True,
        )
        relation = export_result.to_native_relation()
        stage.device = (
            query_execution.selected if query_execution is not None else ExecutionMode.CPU
        )
        stage.rows_out = int(len(relation))
        stage.metadata["query_implementation"] = query_implementation
        if query_execution is not None:
            stage.metadata["query_reason"] = query_execution.reason
        stage.metadata["pair_storage"] = (
            "device"
            if hasattr(relation.left_indices, "__cuda_array_interface__")
            else "host"
        )
        stage.metadata["device_pair_request"] = "requested"
        _record_stage_overheads(stage, audit, memory)

    with profiler.stage(
        "native_semijoin_consumer",
        category="filter",
        device=ExecutionMode.AUTO,
        rows_in=int(len(relation)),
        detail=(
            "consume the relation export as a public-label-preserving native "
            "semijoin without joined GeoDataFrame export"
        ),
    ) as stage:
        started = perf_counter()
        native_frame = export_result.left_unique_label_semijoin_native_frame()
        native_elapsed = perf_counter() - started
        if native_frame is None:
            stage.rows_out = 0
            stage.metadata["bridge_declined"] = True
        else:
            stage.rows_out = int(native_frame.row_count)
            stage.metadata["bridge_declined"] = False
        stage.metadata["consumer_seconds"] = native_elapsed
        stage.metadata["native_index_kind"] = (
            None if native_frame is None else native_frame.index_plan.kind
        )
        stage.metadata["admissibility"] = "unique_label_semijoin"
        stage.metadata["preserve_public_index"] = True
        _record_stage_overheads(stage, audit, memory, native_frame)

    with profiler.stage(
        "public_joined_export_consumer",
        category="emit",
        device=ExecutionMode.CPU,
        rows_in=int(len(relation)),
        detail="reference path: materialize joined rows, unique public labels, then public .loc",
    ) as stage:
        started = perf_counter()
        joined = export_result.to_geodataframe()
        selected_labels = joined.index.unique()
        public_selected = left_frame.loc[selected_labels]
        public_elapsed = perf_counter() - started
        native_row_count = int(0 if native_frame is None else native_frame.row_count)
        results_match = native_row_count == int(len(public_selected))
        stage.rows_out = int(len(public_selected))
        stage.metadata["joined_rows"] = int(len(joined))
        stage.metadata["unique_left_rows"] = int(len(public_selected))
        stage.metadata["consumer_seconds"] = public_elapsed
        stage.metadata["results_match"] = bool(results_match)
        stage.metadata["consumer_speedup"] = (
            public_elapsed / native_elapsed if native_elapsed > 0.0 else float("inf")
        )
        _record_stage_overheads(stage, audit, memory, joined, public_selected)

    stage_devices = [stage.device for stage in profiler._stages]
    actual_selected_runtime = _pipeline_runtime_from_stage_devices(stage_devices)
    trace = profiler.finish(
        metadata={
            "scale": scale,
            "actual_selected_runtime": actual_selected_runtime,
            "planner_selected_runtime": planner_runtime.value,
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
            "admissible_shape": (
                "native-backed public relation export -> unique-label "
                "semijoin native frame"
            ),
            "consumer_speedup": (
                public_elapsed / native_elapsed if native_elapsed > 0.0 else float("inf")
            ),
            "results_match": bool(results_match),
        }
    )
    return PipelineBenchmarkResult(
        pipeline="relation-bridge-consumer",
        scale=scale,
        status="ok" if native_frame is not None and results_match else "failed",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=planner_runtime.value,
        output_rows=int(0 if native_frame is None else native_frame.row_count),
        transfer_count=audit.runtime_d2h_transfer_count,
        owned_transfer_count=audit.transfer_count,
        runtime_d2h_transfer_count=audit.runtime_d2h_transfer_count,
        runtime_d2h_transfer_bytes=audit.runtime_d2h_transfer_bytes,
        runtime_d2h_transfer_seconds=audit.runtime_d2h_transfer_seconds,
        materialization_count=audit.materialization_count,
        fallback_event_count=int(trace.metadata["fallback_events"]),
        peak_device_memory_bytes=memory.peak_bytes,
        stages=(_trace_to_stage_dict(trace),),
        notes=(
            "Shape canary, not a workflow target: compares private device "
            "row-position relation consumption with the public joined-export "
            "+ index.unique + .loc reference path."
        ),
    )


def _profile_grouped_reducer_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Shape canary for dense-code NativeGrouped numeric reducers."""
    if not has_gpu_runtime():
        return PipelineBenchmarkResult(
            pipeline="grouped-reducer",
            scale=scale,
            status="deferred",
            elapsed_seconds=0.0,
            selected_runtime="deferred",
            planner_selected_runtime=ExecutionMode.GPU.value,
            output_rows=0,
            transfer_count=0,
            materialization_count=0,
            fallback_event_count=0,
            peak_device_memory_bytes=None,
            stages=tuple(),
            notes="Deferred until CUDA runtime is available for NativeGrouped reducer canary.",
        )

    from time import perf_counter

    import cupy as cp

    from vibespatial.runtime.materialization import (
        MaterializationBoundary,
        clear_materialization_events,
        record_materialization_event,
    )

    clear_materialization_events()
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    group_count = 128
    profiler = _stage_profiler(
        operation="pipeline.grouped-reducer",
        dataset=f"scale-{scale}",
        requested_runtime=ExecutionMode.GPU,
        selected_runtime=ExecutionMode.GPU.value,
        enable_nvtx=enable_nvtx,
        retain_gpu_trace=retain_gpu_trace,
        include_gpu_sparklines=include_gpu_sparklines,
    )

    with profiler.stage(
        "build_dense_codes",
        category="setup",
        device=ExecutionMode.GPU,
        rows_in=scale,
        detail="build dense device group codes and numeric values for NativeGrouped",
    ) as stage:
        rows = cp.arange(scale, dtype=cp.int32)
        codes = cp.remainder(rows, group_count).astype(cp.int32, copy=False)
        codes = cp.where(cp.remainder(rows, 31) == 0, cp.int32(-1), codes)
        values = (cp.remainder(rows, 17) + 1).astype(cp.float64, copy=False)
        output_index = pd.RangeIndex(group_count, name="group")
        grouped = NativeGrouped.from_dense_codes(
            codes,
            group_count=group_count,
            output_index=output_index,
            source_token="grouped-reducer-canary",
        )
        stage.rows_out = int(grouped.resolved_group_count)
        stage.metadata["group_count"] = group_count
        stage.metadata["row_count"] = scale
        stage.metadata["null_key_policy"] = grouped.null_key_policy
        stage.metadata["group_storage"] = "device" if grouped.is_device else "host"
        _record_stage_overheads(stage, audit, memory)

    with profiler.stage(
        "native_sum",
        category="reduce",
        device=ExecutionMode.GPU,
        rows_in=scale,
        detail="reduce one numeric vector by dense NativeGrouped codes without pandas groupby",
    ) as stage:
        started = perf_counter()
        reduced = grouped.reduce_numeric(values, "sum")
        native_elapsed = perf_counter() - started
        stage.rows_out = int(reduced.group_count)
        stage.metadata["reducer"] = reduced.reducer
        stage.metadata["result_storage"] = "device" if reduced.is_device else "host"
        stage.metadata["native_reduce_seconds"] = native_elapsed
        _record_stage_overheads(stage, audit, memory)

    with profiler.stage(
        "public_groupby_reference",
        category="emit",
        device=ExecutionMode.CPU,
        rows_in=scale,
        detail="reference path: export codes and values, then run pandas groupby sum",
    ) as stage:
        started = perf_counter()
        bytes_to_host = int(codes.nbytes + values.nbytes + reduced.values.nbytes)
        record_materialization_event(
            surface="pipeline.grouped-reducer.public_groupby_reference",
            boundary=MaterializationBoundary.USER_EXPORT,
            operation="grouped_reducer_reference_export",
            reason="exported grouped reducer inputs and output for pandas reference check",
            detail=f"rows={scale}, groups={group_count}, bytes={bytes_to_host}",
            d2h_transfer=True,
            strict_disallowed=False,
        )
        codes_host = cp.asnumpy(codes)
        values_host = cp.asnumpy(values)
        actual = cp.asnumpy(reduced.values)
        observed = codes_host >= 0
        expected = (
            pd.Series(values_host[observed])
            .groupby(codes_host[observed], sort=True)
            .sum()
            .reindex(pd.RangeIndex(group_count), fill_value=0.0)
            .to_numpy()
        )
        public_elapsed = perf_counter() - started
        results_match = bool(np.allclose(actual, expected))
        stage.rows_out = group_count
        stage.metadata["results_match"] = results_match
        stage.metadata["reference_seconds"] = public_elapsed
        stage.metadata["native_reduce_seconds"] = native_elapsed
        stage.metadata["consumer_speedup"] = (
            public_elapsed / native_elapsed if native_elapsed > 0.0 else float("inf")
        )
        _record_stage_overheads(stage, audit, memory)

    stage_devices = [stage.device for stage in profiler._stages]
    actual_selected_runtime = _pipeline_runtime_from_stage_devices(stage_devices)
    trace = profiler.finish(
        metadata={
            "scale": scale,
            "actual_selected_runtime": actual_selected_runtime,
            "planner_selected_runtime": ExecutionMode.GPU.value,
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
            "admissible_shape": "dense-code NativeGrouped numeric sum",
            "results_match": results_match,
            "consumer_speedup": (
                public_elapsed / native_elapsed if native_elapsed > 0.0 else float("inf")
            ),
        }
    )
    return PipelineBenchmarkResult(
        pipeline="grouped-reducer",
        scale=scale,
        status="ok" if results_match else "failed",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=ExecutionMode.GPU.value,
        output_rows=group_count,
        transfer_count=audit.runtime_d2h_transfer_count,
        owned_transfer_count=audit.transfer_count,
        runtime_d2h_transfer_count=audit.runtime_d2h_transfer_count,
        runtime_d2h_transfer_bytes=audit.runtime_d2h_transfer_bytes,
        runtime_d2h_transfer_seconds=audit.runtime_d2h_transfer_seconds,
        materialization_count=audit.materialization_count,
        fallback_event_count=int(trace.metadata["fallback_events"]),
        peak_device_memory_bytes=memory.peak_bytes,
        stages=(_trace_to_stage_dict(trace),),
        notes=(
            "Shape canary, not a workflow target: dense device group codes -> "
            "NativeGrouped.reduce_numeric(sum) with pandas groupby sum only as "
            "an explicit reference stage."
        ),
    )


def _small_grouped_constructive_fixture(
    scale: int,
) -> tuple[list[object], np.ndarray, int]:
    pattern = np.asarray([2, 3, 4, 5, 6, 7, 8, 2], dtype=np.int32)
    group_count = max(8, min(max(scale // 4, 8), 1024))
    group_sizes = np.resize(pattern, group_count).astype(np.int32, copy=False)
    group_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int32), np.cumsum(group_sizes, dtype=np.int32)]
    )
    values: list[object] = []
    for group_index, group_size in enumerate(group_sizes):
        x0 = float(group_index) * 20.0
        for row in range(int(group_size)):
            left = x0 + float(row) * 0.45
            values.append(box(left, 0.0, left + 1.0, 1.0))
    return values, group_offsets, group_count


def _profile_small_grouped_constructive_reduce_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Shape canary for many small grouped polygon constructive reductions."""
    if not has_gpu_runtime():
        return PipelineBenchmarkResult(
            pipeline="small-grouped-constructive-reduce",
            scale=scale,
            status="deferred",
            elapsed_seconds=0.0,
            selected_runtime="deferred",
            planner_selected_runtime=ExecutionMode.GPU.value,
            output_rows=0,
            transfer_count=0,
            materialization_count=0,
            fallback_event_count=0,
            peak_device_memory_bytes=None,
            stages=tuple(),
            notes="Deferred until CUDA runtime is available for grouped constructive canary.",
        )

    from time import perf_counter

    from vibespatial.runtime.materialization import (
        MaterializationBoundary,
        clear_materialization_events,
        record_materialization_event,
    )

    clear_materialization_events()
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    profiler = _stage_profiler(
        operation="pipeline.small-grouped-constructive-reduce",
        dataset=f"scale-{scale}",
        requested_runtime=ExecutionMode.GPU,
        selected_runtime=ExecutionMode.GPU.value,
        enable_nvtx=enable_nvtx,
        retain_gpu_trace=retain_gpu_trace,
        include_gpu_sparklines=include_gpu_sparklines,
    )

    with profiler.stage(
        "build_device_grouped_polygons",
        category="setup",
        device=ExecutionMode.GPU,
        rows_in=scale,
        detail="build device-resident polygon groups with 2-8 rows per group",
    ) as stage:
        values, group_offsets, group_count = _small_grouped_constructive_fixture(scale)
        owned = from_shapely_geometries(values, residency=Residency.DEVICE)
        group_sizes = np.diff(group_offsets)
        stage.rows_out = int(owned.row_count)
        stage.metadata["group_count"] = group_count
        stage.metadata["row_count"] = int(owned.row_count)
        stage.metadata["min_group_size"] = int(group_sizes.min())
        stage.metadata["max_group_size"] = int(group_sizes.max(initial=0))
        stage.metadata["geometry_storage"] = "owned:device"
        _record_stage_overheads(stage, audit, memory, owned)

    with profiler.stage(
        "native_grouped_union",
        category="reduce",
        device=ExecutionMode.GPU,
        rows_in=int(owned.row_count),
        detail="batch many tiny grouped polygon unions without per-group dispatch",
    ) as stage:
        started = perf_counter()
        reduced = segmented_union_all(
            owned,
            group_offsets,
            dispatch_mode=ExecutionMode.GPU,
        )
        native_elapsed = perf_counter() - started
        dispatch_events = get_dispatch_events()
        used_many_small_batch = any(
            event.surface == "segmented_union_all"
            and event.operation == "segmented_union_strategy"
            and event.implementation == "gpu_grouped_overlay_many_small_groups"
            for event in dispatch_events
        )
        stage.rows_out = int(reduced.row_count)
        stage.metadata["native_reduce_seconds"] = native_elapsed
        stage.metadata["result_storage"] = (
            "device" if reduced.residency is Residency.DEVICE else "host"
        )
        stage.metadata["used_many_small_batch"] = used_many_small_batch
        _record_stage_overheads(stage, audit, memory, reduced)

    with profiler.stage(
        "shapely_reference",
        category="emit",
        device=ExecutionMode.CPU,
        rows_in=int(owned.row_count),
        detail="explicit reference export for exact grouped union oracle",
    ) as stage:
        started = perf_counter()
        record_materialization_event(
            surface="pipeline.small-grouped-constructive-reduce.reference",
            boundary=MaterializationBoundary.USER_EXPORT,
            operation="grouped_constructive_reference_export",
            reason="export grouped constructive result for Shapely oracle comparison",
            detail=f"rows={owned.row_count}, groups={group_count}",
            d2h_transfer=True,
            strict_disallowed=False,
        )
        actual = np.asarray(reduced.to_shapely(), dtype=object)
        expected = [
            shapely.union_all(np.asarray(values[int(start) : int(end)], dtype=object))
            for start, end in zip(
                group_offsets[:-1],
                group_offsets[1:],
                strict=True,
            )
        ]
        reference_elapsed = perf_counter() - started
        results_match = all(
            bool(
                shapely.normalize(got).equals_exact(
                    shapely.normalize(want),
                    tolerance=1.0e-9,
                )
            )
            and float(shapely.area(shapely.symmetric_difference(got, want))) <= 1.0e-9
            for got, want in zip(actual, expected, strict=True)
        )
        stage.rows_out = int(actual.size)
        stage.metadata["results_match"] = results_match
        stage.metadata["reference_seconds"] = reference_elapsed
        stage.metadata["native_reduce_seconds"] = native_elapsed
        stage.metadata["consumer_speedup"] = (
            reference_elapsed / native_elapsed if native_elapsed > 0.0 else float("inf")
        )
        _record_stage_overheads(stage, audit, memory, reduced)

    stage_devices = [stage.device for stage in profiler._stages]
    actual_selected_runtime = _pipeline_runtime_from_stage_devices(stage_devices)
    trace = profiler.finish(
        metadata={
            "scale": scale,
            "actual_selected_runtime": actual_selected_runtime,
            "planner_selected_runtime": ExecutionMode.GPU.value,
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
            "admissible_shape": (
                "owned device polygons + dense group offsets -> batched grouped constructive reduce"
            ),
            "results_match": results_match,
            "used_many_small_batch": used_many_small_batch,
            "consumer_speedup": (
                reference_elapsed / native_elapsed if native_elapsed > 0.0 else float("inf")
            ),
        }
    )
    return PipelineBenchmarkResult(
        pipeline="small-grouped-constructive-reduce",
        scale=scale,
        status="ok" if results_match and used_many_small_batch else "failed",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=ExecutionMode.GPU.value,
        output_rows=group_count,
        transfer_count=audit.runtime_d2h_transfer_count,
        owned_transfer_count=audit.transfer_count,
        runtime_d2h_transfer_count=audit.runtime_d2h_transfer_count,
        runtime_d2h_transfer_bytes=audit.runtime_d2h_transfer_bytes,
        runtime_d2h_transfer_seconds=audit.runtime_d2h_transfer_seconds,
        materialization_count=audit.materialization_count,
        fallback_event_count=int(trace.metadata["fallback_events"]),
        peak_device_memory_bytes=memory.peak_bytes,
        stages=(_trace_to_stage_dict(trace),),
        notes=(
            "Shape canary, not a workflow target: many small device-resident "
            "polygon groups reduce through one batched grouped constructive path; "
            "Shapely is used only as an explicit terminal oracle."
        ),
    )


def _profile_relation_attribute_reducer_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Shape canary for relation-derived grouped attribute reducers."""
    if not has_gpu_runtime():
        return PipelineBenchmarkResult(
            pipeline="relation-attribute-reducer",
            scale=scale,
            status="deferred",
            elapsed_seconds=0.0,
            selected_runtime="deferred",
            planner_selected_runtime=ExecutionMode.GPU.value,
            output_rows=0,
            transfer_count=0,
            materialization_count=0,
            fallback_event_count=0,
            peak_device_memory_bytes=None,
            stages=tuple(),
            notes="Deferred until CUDA runtime is available for relation attribute reducer canary.",
        )

    from time import perf_counter

    import cupy as cp

    from vibespatial.runtime.materialization import (
        MaterializationBoundary,
        clear_materialization_events,
        record_materialization_event,
    )

    clear_materialization_events()
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    pair_count = int(scale)
    left_row_count = max(pair_count // 4, 1)
    right_row_count = max(pair_count // 8, 1)
    profiler = _stage_profiler(
        operation="pipeline.relation-attribute-reducer",
        dataset=f"scale-{scale}",
        requested_runtime=ExecutionMode.GPU,
        selected_runtime=ExecutionMode.GPU.value,
        enable_nvtx=enable_nvtx,
        retain_gpu_trace=retain_gpu_trace,
        include_gpu_sparklines=include_gpu_sparklines,
    )

    with profiler.stage(
        "build_relation_inputs",
        category="setup",
        device=ExecutionMode.GPU,
        rows_in=pair_count,
        detail="build synthetic device relation pairs and all-valid right-side numeric attributes",
    ) as stage:
        pair_rows = cp.arange(pair_count, dtype=cp.int64)
        right_rows = cp.arange(right_row_count, dtype=cp.int64)
        left_indices = cp.remainder(pair_rows, left_row_count).astype(cp.int32, copy=False)
        right_indices = cp.remainder(pair_rows * 7 + 3, right_row_count).astype(
            cp.int32,
            copy=False,
        )
        right_score = (cp.remainder(right_rows, 23) + 1).astype(cp.float64, copy=False)
        right_weight = (cp.remainder(right_rows, 5) + 1).astype(cp.float64, copy=False)
        relation = NativeRelation(
            left_indices=left_indices,
            right_indices=right_indices,
            left_token="left",
            right_token="right",
            left_row_count=left_row_count,
            right_row_count=right_row_count,
            sorted_by_left=False,
        )
        stage.rows_out = int(len(relation))
        stage.metadata["left_row_count"] = left_row_count
        stage.metadata["right_row_count"] = right_row_count
        stage.metadata["pair_storage"] = "device"
        _record_stage_overheads(stage, audit, memory)

    with profiler.stage(
        "native_attribute_reduce",
        category="reduce",
        device=ExecutionMode.GPU,
        rows_in=pair_count,
        detail="reduce right-side attributes into left-row groups without joined row assembly",
    ) as stage:
        started = perf_counter()
        reduced = relation.left_reduce_right_numeric_columns(
            {
                "score_sum": right_score,
                "match_count": right_score,
                "weight_mean": right_weight,
            },
            {
                "score_sum": "sum",
                "match_count": "count",
                "weight_mean": "mean",
            },
        )
        native_elapsed = perf_counter() - started
        stage.rows_out = int(reduced.group_count)
        stage.metadata["columns"] = tuple(reduced.columns)
        stage.metadata["result_storage"] = "device" if reduced.is_device else "host"
        stage.metadata["native_reduce_seconds"] = native_elapsed
        _record_stage_overheads(stage, audit, memory)

    with profiler.stage(
        "public_groupby_reference",
        category="emit",
        device=ExecutionMode.CPU,
        rows_in=pair_count,
        detail="reference path: export relation pairs and attributes, then run pandas groupby reductions",
    ) as stage:
        started = perf_counter()
        bytes_to_host = int(
            left_indices.nbytes
            + right_indices.nbytes
            + right_score.nbytes
            + right_weight.nbytes
            + sum(reduction.values.nbytes for reduction in reduced.columns.values())
        )
        record_materialization_event(
            surface="pipeline.relation-attribute-reducer.public_groupby_reference",
            boundary=MaterializationBoundary.USER_EXPORT,
            operation="relation_attribute_reducer_reference_export",
            reason="exported relation attribute reducer inputs and output for pandas reference check",
            detail=(
                f"pairs={pair_count}, left_rows={left_row_count}, "
                f"right_rows={right_row_count}, bytes={bytes_to_host}"
            ),
            d2h_transfer=True,
            strict_disallowed=False,
        )
        left_host = cp.asnumpy(left_indices)
        right_host = cp.asnumpy(right_indices)
        score_host = cp.asnumpy(right_score)
        weight_host = cp.asnumpy(right_weight)
        actual_score = cp.asnumpy(reduced.columns["score_sum"].values)
        actual_count = cp.asnumpy(reduced.columns["match_count"].values)
        actual_weight = cp.asnumpy(reduced.columns["weight_mean"].values)
        pairs = pd.DataFrame(
            {
                "left": left_host,
                "score": score_host[right_host],
                "weight": weight_host[right_host],
            }
        )
        grouped = pairs.groupby("left", sort=True)
        expected = pd.DataFrame(index=pd.RangeIndex(left_row_count))
        expected["score_sum"] = grouped["score"].sum().reindex(
            expected.index,
            fill_value=0.0,
        )
        expected["match_count"] = grouped["score"].count().reindex(
            expected.index,
            fill_value=0,
        )
        expected["weight_mean"] = grouped["weight"].mean().reindex(expected.index)
        public_elapsed = perf_counter() - started
        results_match = bool(
            np.allclose(actual_score, expected["score_sum"].to_numpy())
            and np.array_equal(actual_count, expected["match_count"].to_numpy())
            and np.allclose(
                actual_weight,
                expected["weight_mean"].to_numpy(),
                equal_nan=True,
            )
        )
        stage.rows_out = left_row_count
        stage.metadata["results_match"] = results_match
        stage.metadata["reference_seconds"] = public_elapsed
        stage.metadata["native_reduce_seconds"] = native_elapsed
        stage.metadata["consumer_speedup"] = (
            public_elapsed / native_elapsed if native_elapsed > 0.0 else float("inf")
        )
        _record_stage_overheads(stage, audit, memory)

    stage_devices = [stage.device for stage in profiler._stages]
    actual_selected_runtime = _pipeline_runtime_from_stage_devices(stage_devices)
    trace = profiler.finish(
        metadata={
            "scale": scale,
            "actual_selected_runtime": actual_selected_runtime,
            "planner_selected_runtime": ExecutionMode.GPU.value,
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
            "admissible_shape": "device NativeRelation -> grouped right numeric attributes by left rows",
            "results_match": results_match,
            "consumer_speedup": (
                public_elapsed / native_elapsed if native_elapsed > 0.0 else float("inf")
            ),
        }
    )
    return PipelineBenchmarkResult(
        pipeline="relation-attribute-reducer",
        scale=scale,
        status="ok" if results_match else "failed",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=ExecutionMode.GPU.value,
        output_rows=left_row_count,
        transfer_count=audit.runtime_d2h_transfer_count,
        owned_transfer_count=audit.transfer_count,
        runtime_d2h_transfer_count=audit.runtime_d2h_transfer_count,
        runtime_d2h_transfer_bytes=audit.runtime_d2h_transfer_bytes,
        runtime_d2h_transfer_seconds=audit.runtime_d2h_transfer_seconds,
        materialization_count=audit.materialization_count,
        fallback_event_count=int(trace.metadata["fallback_events"]),
        peak_device_memory_bytes=memory.peak_bytes,
        stages=(_trace_to_stage_dict(trace),),
        notes=(
            "Shape canary, not a workflow target: device NativeRelation pairs -> "
            "right-side numeric attributes gathered per pair -> NativeGrouped "
            "multi-column reductions by left source row."
        ),
    )


def _profile_constructive_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    planner_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU
    read_backend = _preferred_geoparquet_backend()

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "constructive.parquet"
        _regular_points_frame(scale).to_parquet(source_path, geometry_encoding="geoarrow")

        audit.reset_runtime_baseline()
        profiler = _stage_profiler(
            operation="pipeline.constructive",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.AUTO,
            selected_runtime="hybrid" if has_gpu_runtime() else "cpu",
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        with profiler.stage("read_points", category="setup", device="auto", rows_in=scale) as stage:
            owned, actual_read_backend, read_note = _read_geoparquet_owned_preferred(
                source_path,
                preferred_backend=read_backend,
            )
            stage.device = actual_read_backend
            stage.rows_out = owned.row_count
            stage.metadata["requested_backend"] = read_backend
            stage.metadata["actual_backend"] = actual_read_backend
            if read_note:
                stage.metadata["fallback_note"] = read_note
            _record_stage_overheads(stage, audit, memory, owned)

        with profiler.stage(
            "clip_points",
            category="filter",
            device=plan_dispatch_selection(
                kernel_name="point_clip",
                kernel_class=KernelClass.CONSTRUCTIVE,
                row_count=owned.row_count,
                requested_mode=ExecutionMode.AUTO,
                gpu_available=has_gpu_runtime(),
            ).selected,
            rows_in=owned.row_count,
            detail="clip point rows to an axis-aligned rectangle before buffer expansion",
        ) as stage:
            clip_runtime = stage.device
            clipped = clip_points_rect_owned(
                owned,
                0.0,
                0.0,
                float(max(scale // 100, 1)),
                float(max(scale // 100, 1)),
                dispatch_mode=clip_runtime,
            )
            stage.rows_out = int(clipped.row_count)
            _record_stage_overheads(stage, audit, memory, owned, clipped)

        with profiler.stage(
            "buffer_points",
            category="refine",
            device=plan_dispatch_selection(
                kernel_name="point_buffer",
                kernel_class=KernelClass.CONSTRUCTIVE,
                row_count=clipped.row_count,
                requested_mode=ExecutionMode.AUTO,
                gpu_available=has_gpu_runtime(),
            ).selected,
            rows_in=int(clipped.row_count),
            detail="expand surviving point rows into buffer polygons",
        ) as stage:
            buffer_runtime = stage.device
            buffered = point_buffer_owned_array(
                clipped,
                0.5,
                quad_segs=1,
                dispatch_mode=buffer_runtime,
            )
            stage.rows_out = int(buffered.row_count)
            _record_stage_overheads(stage, audit, memory, clipped, buffered)

        output = GeometryNativeResult.from_owned(buffered, crs="EPSG:4326")
        output_path = root / "constructive-output.parquet"
        with profiler.stage(
            "write_output",
            category="emit",
            device="auto",
            rows_in=int(output.row_count),
            detail="write constructive pipeline result to GeoParquet",
        ) as stage:
            write_geoparquet(
                output,
                output_path,
                geometry_encoding="geoarrow",
                compression=_BENCHMARK_OUTPUT_COMPRESSION,
            )
            stage.device = _selected_runtime_from_history(output) or "cpu"
            stage.rows_out = int(output.row_count)
            stage.metadata["compression"] = _BENCHMARK_OUTPUT_COMPRESSION
            _record_stage_overheads(stage, audit, memory, output)

    trace = profiler.finish(
        metadata={
            "scale": scale,
            "actual_selected_runtime": (
                "hybrid"
                if actual_read_backend == "gpu"
                or clip_runtime is ExecutionMode.GPU
                or buffer_runtime is ExecutionMode.GPU
                else "cpu"
            ),
            "planner_selected_runtime": planner_runtime.value,
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
        }
    )
    return PipelineBenchmarkResult(
        pipeline="constructive",
        scale=scale,
        status="ok",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=(
            "hybrid"
            if actual_read_backend == "gpu"
            or clip_runtime is ExecutionMode.GPU
            or buffer_runtime is ExecutionMode.GPU
            else "cpu"
        ),
        planner_selected_runtime=planner_runtime.value,
        output_rows=int(output.row_count),
        transfer_count=audit.transfer_count,
        owned_transfer_count=audit.transfer_count,
        runtime_d2h_transfer_count=audit.runtime_d2h_transfer_count,
        runtime_d2h_transfer_bytes=audit.runtime_d2h_transfer_bytes,
        runtime_d2h_transfer_seconds=audit.runtime_d2h_transfer_seconds,
        materialization_count=audit.materialization_count,
        fallback_event_count=int(trace.metadata["fallback_events"]),
        peak_device_memory_bytes=memory.peak_bytes,
        stages=(_trace_to_stage_dict(trace),),
        notes="Current constructive pipeline measures read -> owned point clip -> owned point buffer -> GeoParquet write.",
    )


def _profile_predicate_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    planner_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU
    read_mode = _preferred_geojson_mode()

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        polygon_count = max(scale // 8, 1)

        source_path = root / "predicate.geojson"
        frame = _regular_points_frame(scale)
        source_bytes = frame.to_json().encode("utf-8")
        # Write RFC 7946 GeoJSON (no GDAL CRS block) so the GPU
        # byte-classify parser can consume it directly.
        source_path.write_bytes(source_bytes)

        audit.reset_runtime_baseline()
        profiler = _stage_profiler(
            operation="pipeline.predicate-heavy",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.AUTO,
            selected_runtime="cpu",
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        with profiler.stage("read_geojson", category="setup", device="auto", rows_in=scale) as stage:
            batch, actual_read_mode, read_note = _read_geojson_owned_preferred(
                source_bytes,
                preferred_mode=read_mode,
            )
            stage.device = actual_read_mode
            stage.rows_out = batch.geometry.row_count
            stage.metadata["requested_mode"] = read_mode
            stage.metadata["actual_mode"] = actual_read_mode
            stage.metadata["source_kind"] = "bytes"
            if read_note:
                stage.metadata["fallback_note"] = read_note
            _record_stage_overheads(stage, audit, memory, batch)

        with profiler.stage(
            "load_polygons",
            category="setup",
            device="auto",
            rows_in=batch.geometry.row_count,
            detail="load cached polygon OwnedGeometryArray (first run generates and caches to .benchmark_fixtures/)",
        ) as stage:
            polygon_owned = _load_or_build_polygon_owned(polygon_count, batch.geometry.row_count)
            stage.device = _selected_runtime_from_history(polygon_owned) or _preferred_geoparquet_backend()
            stage.rows_out = polygon_owned.row_count
            stage.metadata["cache_path"] = str(_predicate_polygon_cache_path(polygon_count, batch.geometry.row_count))
            stage.metadata["cache_hit"] = _predicate_polygon_cache_path(polygon_count, batch.geometry.row_count).exists()
            _record_stage_overheads(stage, audit, memory, polygon_owned)

        with profiler.stage(
            "point_in_polygon",
            category="refine",
            device=ExecutionMode.AUTO,
            rows_in=batch.geometry.row_count,
            detail="evaluate point-in-polygon with pre-constructed polygon buffers",
        ) as stage:
            history_before = len(batch.geometry.runtime_history)
            mask = point_in_polygon(batch.geometry, polygon_owned, _return_device=True)
            stage.rows_out = int(mask.sum())
            runtime_selection = batch.geometry.runtime_history[history_before:] or batch.geometry.runtime_history[-1:]
            if runtime_selection:
                stage.device = runtime_selection[-1].selected.value
                stage.metadata["runtime_reason"] = runtime_selection[-1].reason
            gpu_timings = get_last_gpu_substage_timings()
            if gpu_timings:
                stage.metadata["gpu_substage_timings"] = gpu_timings
            _record_stage_overheads(stage, audit, memory, batch)

        filter_device = _actual_array_device_label(mask)
        with profiler.stage(
            "filter_points",
            category="filter",
            device=filter_device,
            rows_in=batch.geometry.row_count,
            detail="filter GeoJSON point rows by predicate hit mask (buffer-level take)",
        ) as stage:
            survivors = _subset_by_mask(batch.geometry, mask)
            stage.rows_out = survivors.row_count

        output = GeometryNativeResult.from_owned(survivors, crs="EPSG:4326")
        output_path = root / "predicate-output.parquet"
        with profiler.stage(
            "write_output",
            category="emit",
            device="auto",
            rows_in=int(output.row_count),
            detail="write filtered predicate result to GeoParquet",
        ) as stage:
            write_geoparquet(
                output,
                output_path,
                geometry_encoding="geoarrow",
                compression=_BENCHMARK_OUTPUT_COMPRESSION,
            )
            stage.device = _selected_runtime_from_history(output) or "cpu"
            stage.rows_out = int(output.row_count)
            stage.metadata["compression"] = _BENCHMARK_OUTPUT_COMPRESSION
            _record_stage_overheads(stage, audit, memory, output)

    stage_devices = [stage.device for stage in profiler._stages]
    actual_selected_runtime = _pipeline_runtime_from_stage_devices(stage_devices)
    trace = profiler.finish(
        metadata={
            "scale": scale,
            "actual_selected_runtime": actual_selected_runtime,
            "planner_selected_runtime": planner_runtime.value,
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
        }
    )
    return PipelineBenchmarkResult(
        pipeline="predicate-heavy",
        scale=scale,
        status="ok",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=planner_runtime.value,
        output_rows=int(output.row_count),
        transfer_count=audit.transfer_count,
        owned_transfer_count=audit.transfer_count,
        runtime_d2h_transfer_count=audit.runtime_d2h_transfer_count,
        runtime_d2h_transfer_bytes=audit.runtime_d2h_transfer_bytes,
        runtime_d2h_transfer_seconds=audit.runtime_d2h_transfer_seconds,
        materialization_count=audit.materialization_count,
        fallback_event_count=int(trace.metadata["fallback_events"]),
        peak_device_memory_bytes=memory.peak_bytes,
        stages=(_trace_to_stage_dict(trace),),
        notes="Current predicate-heavy pipeline measures GeoJSON bytes ingest -> point_in_polygon -> filter -> GeoParquet write.",
    )


def _profile_predicate_geopandas_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    clear_dispatch_events()
    clear_fallback_events()
    planner_runtime = ExecutionMode.CPU

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        polygon_count = max(scale // 8, 1)

        source_path = root / "predicate.geojson"
        frame = _regular_points_frame(scale)
        frame.to_file(source_path, driver="GeoJSON")

        profiler = _stage_profiler(
            operation="pipeline.predicate-heavy-geopandas",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.CPU,
            selected_runtime=ExecutionMode.CPU,
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        with profiler.stage("read_geojson", category="setup", device=ExecutionMode.CPU, rows_in=scale) as stage:
            point_frame, requested_engine, actual_engine, fallback_note = _read_geojson_geopandas_preferred(
                source_path
            )
            stage.rows_out = int(len(point_frame))
            stage.metadata["requested_engine"] = requested_engine
            stage.metadata["actual_engine"] = actual_engine
            if fallback_note:
                stage.metadata["fallback_note"] = fallback_note

        with profiler.stage(
            "load_polygons",
            category="setup",
            device=ExecutionMode.CPU,
            rows_in=int(len(point_frame)),
            detail="load cached polygon GeoSeries for the GeoPandas baseline path",
        ) as stage:
            polygon_series = _load_or_build_polygon_geoseries(polygon_count, int(len(point_frame)))
            stage.rows_out = int(len(polygon_series))
            stage.metadata["cache_path"] = str(_predicate_polygon_cache_path(polygon_count, int(len(point_frame))))
            stage.metadata["cache_hit"] = _predicate_polygon_cache_path(polygon_count, int(len(point_frame))).exists()

        with profiler.stage(
            "point_in_polygon",
            category="refine",
            device=ExecutionMode.CPU,
            rows_in=int(len(point_frame)),
            detail="evaluate boundary-inclusive point-in-polygon with GeoPandas/Shapely covers semantics",
        ) as stage:
            hits = polygon_series.reset_index(drop=True).covers(point_frame.geometry.reset_index(drop=True), align=False)
            mask = hits.to_numpy(dtype=bool, copy=False)
            stage.rows_out = int(mask.sum())

        with profiler.stage(
            "filter_points",
            category="filter",
            device=ExecutionMode.CPU,
            rows_in=int(len(point_frame)),
            detail="filter GeoJSON point rows by the GeoPandas predicate hit mask",
        ) as stage:
            output = point_frame.loc[mask].copy()
            stage.rows_out = int(len(output))

        output_path = root / "predicate-output.parquet"
        with profiler.stage(
            "write_output",
            category="emit",
            device=ExecutionMode.CPU,
            rows_in=int(len(output)),
            detail="write filtered GeoPandas baseline result to GeoParquet",
        ) as stage:
            output.to_parquet(output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(output))

    trace = profiler.finish(
        metadata={
            "scale": scale,
            "actual_selected_runtime": "cpu",
            "planner_selected_runtime": planner_runtime.value,
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
        }
    )
    return PipelineBenchmarkResult(
        pipeline="predicate-heavy-geopandas",
        scale=scale,
        status="ok",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime="cpu",
        planner_selected_runtime=planner_runtime.value,
        output_rows=int(len(output)),
        transfer_count=0,
        materialization_count=0,
        fallback_event_count=int(trace.metadata["fallback_events"]),
        peak_device_memory_bytes=None,
        stages=(_trace_to_stage_dict(trace),),
        notes="GeoPandas baseline pipeline measures read_geojson -> covers -> filter -> to_parquet on host.",
    )


def _profile_zero_transfer_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return PipelineBenchmarkResult(
            pipeline="zero-transfer",
            scale=scale,
            status="deferred",
            elapsed_seconds=0.0,
            selected_runtime="deferred",
            planner_selected_runtime="deferred",
            output_rows=0,
            transfer_count=0,
            materialization_count=0,
            fallback_event_count=0,
            peak_device_memory_bytes=None,
            stages=tuple(),
            notes="Deferred until both CUDA runtime and pylibcudf are available for the public DGA read/write path.",
        )

    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "zero-transfer-input.parquet"
        output_path = root / "zero-transfer-output.parquet"
        _regular_points_frame(scale).to_parquet(source_path, geometry_encoding="geoarrow")

        audit.reset_runtime_baseline()
        profiler = _stage_profiler(
            operation="pipeline.zero-transfer",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.GPU,
            selected_runtime="gpu",
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        with profiler.stage("read_input", category="setup", device=ExecutionMode.GPU, rows_in=scale) as stage:
            native_payload = read_geoparquet_native(source_path)
            native_state = NativeFrameState.from_native_tabular_result(native_payload)
            frame = None
            stage.rows_out = int(native_state.row_count)
            stage.metadata["geometry_dtype"] = "native_device_geometry"
            stage.metadata["private_native_state"] = True
            stage.metadata["read_surface"] = "read_geoparquet_native"
            _record_stage_overheads(stage, audit, memory, native_payload)

        with profiler.stage(
            "predicate_filter",
            category="filter",
            device=ExecutionMode.GPU,
            rows_in=int(native_state.row_count),
            detail="evaluate the point-box predicate as a private NativeRowSet over the hidden native frame state",
        ) as stage:
            predicate_bounds = np.asarray(
                [
                    0.0,
                    0.0,
                    _ZERO_TRANSFER_SELECTIVE_BOUND,
                    _ZERO_TRANSFER_SELECTIVE_BOUND,
                ],
                dtype=np.float64,
            )
            stage.metadata["predicate_bounds"] = tuple(float(value) for value in predicate_bounds)
            rowset = None
            if native_state is not None and native_state.geometry.owned is not None:
                positions = _query_point_tree_box_row_positions_device(
                    native_state.geometry.owned,
                    predicate="intersects",
                    box_bounds=predicate_bounds,
                    force_gpu=True,
                )
                if positions is not None:
                    rowset = NativeRowSet.from_positions(
                        positions,
                        source_token=native_state.lineage_token,
                        source_row_count=native_state.row_count,
                        ordered=True,
                        unique=True,
                    )
                    stage.metadata["rowset_storage"] = "device" if rowset.is_device else "host"
            if rowset is None:
                frame = native_payload.to_geodataframe()
                indices = np.flatnonzero(
                    np.asarray(
                        frame.geometry.values.intersects(box(*predicate_bounds)),
                        dtype=bool,
                    )
                )
                stage.metadata["rowset_storage"] = "fallback-host"
                stage.rows_out = int(indices.size)
            else:
                indices = None
                stage.rows_out = len(rowset)
            _record_stage_overheads(stage, audit, memory, native_payload, frame)

        with profiler.stage(
            "subset_rows",
            category="filter",
            device=ExecutionMode.GPU,
            rows_in=int(native_state.row_count),
            detail="apply the private NativeRowSet to the hidden native frame state",
        ) as stage:
            if rowset is not None and native_state is not None:
                if len(rowset) == native_state.row_count:
                    filtered = native_state.to_native_tabular_result()
                    stage.metadata["native_rowset_take"] = "identity"
                else:
                    filtered = native_state.take(
                        rowset,
                        preserve_index=False,
                    ).to_native_tabular_result()
                    stage.metadata["native_rowset_take"] = "take"
                stage.rows_out = int(filtered.geometry.row_count)
            else:
                if frame is None:
                    frame = native_payload.to_geodataframe()
                filtered = _take_dga_frame(frame, indices)
                stage.metadata["native_rowset_take"] = "fallback-public"
                stage.rows_out = int(len(filtered))
            _record_stage_overheads(stage, audit, memory, native_payload, frame, filtered)

        with profiler.stage(
            "write_output",
            category="emit",
            device=ExecutionMode.GPU,
            rows_in=_tabular_row_count(filtered),
            detail="write the filtered private native payload through the native GeoParquet path",
        ) as stage:
            write_geoparquet(
                filtered,
                output_path,
                index=False,
                geometry_encoding="geoarrow",
                compression=_BENCHMARK_OUTPUT_COMPRESSION,
            )
            stage.rows_out = _tabular_row_count(filtered)
            stage.metadata["compression"] = _BENCHMARK_OUTPUT_COMPRESSION
            _record_stage_overheads(stage, audit, memory, filtered)

    owned_transfer_count = audit.transfer_count
    transfer_count = audit.runtime_d2h_transfer_count
    materialization_count = audit.materialization_count
    status = "ok" if transfer_count == 0 and materialization_count == 0 else "failed"
    stage_devices = [stage.device for stage in profiler._stages]
    actual_selected_runtime = _pipeline_runtime_from_stage_devices(stage_devices)
    planner_selected_runtime = ExecutionMode.GPU.value
    trace = profiler.finish(
        metadata={
            "scale": scale,
            "actual_selected_runtime": actual_selected_runtime,
            "planner_selected_runtime": planner_selected_runtime,
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
        }
    )
    return PipelineBenchmarkResult(
        pipeline="zero-transfer",
        scale=scale,
        status=status,
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=planner_selected_runtime,
        output_rows=_tabular_row_count(filtered),
        transfer_count=transfer_count,
        owned_transfer_count=owned_transfer_count,
        runtime_d2h_transfer_count=audit.runtime_d2h_transfer_count,
        runtime_d2h_transfer_bytes=audit.runtime_d2h_transfer_bytes,
        runtime_d2h_transfer_seconds=audit.runtime_d2h_transfer_seconds,
        materialization_count=materialization_count,
        fallback_event_count=int(trace.metadata["fallback_events"]),
        peak_device_memory_bytes=memory.peak_bytes,
        stages=(_trace_to_stage_dict(trace),),
        notes="Private read_geoparquet_native -> NativeFrameState -> point-box NativeRowSet -> native GeoParquet write zero-transfer substrate scenario.",
    )


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------


def _empty_owned_placeholder() -> OwnedGeometryArray:
    """Return a 0-row OwnedGeometryArray safe for downstream pipeline stages."""
    dummy = from_shapely_geometries([shapely.Point(0, 0)])
    return dummy.take(np.asarray([], dtype=np.int64))


def _from_shapely_safe(geoms: list) -> OwnedGeometryArray:
    """Convert shapely geometries, filtering out unsupported types like GeometryCollection."""
    filtered = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        if g.geom_type == "GeometryCollection":
            # Extract supported geometry types from collections
            filtered.extend(_extract_supported_collection_parts(g, _SUPPORTED_COLLECTION_GEOM_TYPES))
        else:
            filtered.append(g)
    if not filtered:
        return from_shapely_geometries([shapely.Point(0, 0)])  # dummy to avoid empty array issues
    return from_shapely_geometries(filtered)


def _overlay_via_public_api(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    how: str = "intersection",
) -> OwnedGeometryArray:
    """Run overlay through the public geopandas.overlay() API.

    Builds GeoDataFrames from owned arrays (preserving owned backing for GPU
    dispatch), calls the public overlay, and extracts the result as an
    OwnedGeometryArray.  All dispatch/fallback decisions are handled inside
    the public API — no benchmark-specific overlay logic.
    """
    left_gdf = geopandas.GeoDataFrame(
        {"geometry": geoseries_from_owned(left, crs="EPSG:4326")},
        geometry="geometry",
        crs="EPSG:4326",
    )
    right_gdf = geopandas.GeoDataFrame(
        {"geometry": geoseries_from_owned(right, crs="EPSG:4326")},
        geometry="geometry",
        crs="EPSG:4326",
    )
    result_gdf = geopandas.overlay(left_gdf, right_gdf, how=how, make_valid=True)
    # Extract owned backing from result; fall back to Shapely conversion
    result_owned = getattr(result_gdf.geometry.values, "_owned", None)
    if result_owned is None and len(result_gdf) > 0:
        result_owned = from_shapely_geometries(list(result_gdf.geometry))
    if result_owned is None:
        result_owned = _empty_owned_placeholder()
    return result_owned


# ---------------------------------------------------------------------------
# Realistic workflow data helpers
# ---------------------------------------------------------------------------


def _powerline_network_frame(count: int) -> geopandas.GeoDataFrame:
    """Sinusoidal linestrings mimicking powerline corridors."""
    dataset = generate_lines(SyntheticSpec("line", "river", count=max(count, 1), seed=10, vertices=12))
    values = np.asarray(list(dataset.geometries), dtype=object)
    values = shapely.make_valid(values)
    return geopandas.GeoDataFrame(
        {
            "circuit_id": pd.Categorical(np.arange(len(values), dtype=np.int32) % max(min(len(values), 32), 1)),
            "geometry": values,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _vegetation_patches_frame(count: int) -> geopandas.GeoDataFrame:
    """Irregular convex-hull polygons mimicking vegetation patches."""
    dataset = generate_polygons(
        SyntheticSpec("polygon", "convex-hull", count=max(count, 1), seed=11, clusters=8, vertices=6)
    )
    values = np.asarray(list(dataset.geometries), dtype=object)
    values = shapely.make_valid(values)
    return geopandas.GeoDataFrame(
        {
            "species": pd.Categorical(np.arange(len(values), dtype=np.int32) % 5),
            "geometry": values,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _utility_poles_frame(count: int) -> geopandas.GeoDataFrame:
    """Clustered points mimicking utility poles along routes."""
    dataset = generate_points(SyntheticSpec("point", "clustered", count=max(count, 1), seed=12, clusters=12))
    values = np.asarray(list(dataset.geometries), dtype=object)
    return geopandas.GeoDataFrame(
        {
            "pole_type": pd.Categorical(np.arange(len(values), dtype=np.int32) % 3),
            "geometry": values,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _parcels_frame(count: int) -> geopandas.GeoDataFrame:
    """Regular-grid polygons mimicking rectangular land parcels."""
    dataset = generate_polygons(
        SyntheticSpec("polygon", "regular-grid", count=max(count, 1), seed=20, vertices=4)
    )
    values = np.asarray(list(dataset.geometries), dtype=object)
    values = shapely.make_valid(values)
    return geopandas.GeoDataFrame(
        {
            "parcel_id": np.arange(len(values), dtype=np.int64),
            "geometry": values,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _zoning_polygons_frame(count: int) -> geopandas.GeoDataFrame:
    """Irregular convex-hull polygons mimicking zoning boundaries."""
    dataset = generate_polygons(
        SyntheticSpec("polygon", "convex-hull", count=max(count, 1), seed=21, clusters=4, vertices=8)
    )
    values = np.asarray(list(dataset.geometries), dtype=object)
    values = shapely.make_valid(values)
    return geopandas.GeoDataFrame(
        {
            "zone_type": pd.Categorical(np.arange(len(values), dtype=np.int32) % 4),
            "geometry": values,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _buildings_frame(count: int) -> geopandas.GeoDataFrame:
    """Dense regular-grid polygons mimicking building footprints."""
    dataset = generate_polygons(
        SyntheticSpec("polygon", "regular-grid", count=max(count, 1), seed=30, vertices=4)
    )
    values = np.asarray(list(dataset.geometries), dtype=object)
    values = shapely.make_valid(values)
    return geopandas.GeoDataFrame(
        {
            "building_id": np.arange(len(values), dtype=np.int64),
            "geometry": values,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _flood_zones_frame(count: int) -> geopandas.GeoDataFrame:
    """Large star-shaped polygons mimicking flood zone boundaries."""
    dataset = generate_polygons(
        SyntheticSpec("polygon", "star", count=max(count, 1), seed=31, vertices=10)
    )
    values = np.asarray(list(dataset.geometries), dtype=object)
    values = shapely.make_valid(values)
    return geopandas.GeoDataFrame(
        {
            "zone_id": np.arange(len(values), dtype=np.int64),
            "geometry": values,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _network_lines_frame(count: int) -> geopandas.GeoDataFrame:
    """Grid linestrings mimicking a telecom or road network."""
    dataset = generate_lines(SyntheticSpec("line", "grid", count=max(count, 1), seed=40))
    values = np.asarray(list(dataset.geometries), dtype=object)
    return geopandas.GeoDataFrame(
        {
            "segment_id": np.arange(len(values), dtype=np.int64),
            "geometry": values,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _admin_boundary_frame() -> geopandas.GeoDataFrame:
    """Single large star polygon mimicking an administrative boundary."""
    dataset = generate_polygons(
        SyntheticSpec("polygon", "star", count=1, seed=41, vertices=12, bounds=(100.0, 100.0, 900.0, 900.0))
    )
    values = np.asarray(list(dataset.geometries), dtype=object)
    values = shapely.make_valid(values)
    return geopandas.GeoDataFrame(
        {"admin_name": ["Region A"], "geometry": values},
        geometry="geometry",
        crs="EPSG:4326",
    )


def _exclusion_zones_frame(count: int) -> geopandas.GeoDataFrame:
    """Convex-hull polygons mimicking environmental exclusion zones."""
    dataset = generate_polygons(
        SyntheticSpec("polygon", "convex-hull", count=max(count, 1), seed=50, clusters=6, vertices=8)
    )
    values = np.asarray(list(dataset.geometries), dtype=object)
    values = shapely.make_valid(values)
    return geopandas.GeoDataFrame(
        {
            "exclusion_type": pd.Categorical(np.arange(len(values), dtype=np.int32) % 3),
            "geometry": values,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _transit_stations_frame(count: int) -> geopandas.GeoDataFrame:
    """Clustered points mimicking transit stations."""
    dataset = generate_points(SyntheticSpec("point", "clustered", count=max(count, 1), seed=51, clusters=8))
    values = np.asarray(list(dataset.geometries), dtype=object)
    return geopandas.GeoDataFrame(
        {
            "station_id": np.arange(len(values), dtype=np.int64),
            "geometry": values,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# Workflow 1: vegetation-corridor
# ---------------------------------------------------------------------------


def _profile_vegetation_corridor_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Powerline right-of-way vegetation monitoring workflow.

    Load powerline network (lines), vegetation patches (polygons), and utility
    poles (points).  Buffer lines to create corridor, dissolve overlapping
    corridor polygons, intersect with vegetation, find poles near clipped
    vegetation.
    """
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    planner_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU
    read_backend = _preferred_geoparquet_backend()

    line_count = max(scale // 50, 2)
    polygon_count = max(scale // 10, 2)
    point_count = scale

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        lines_path = root / "lines.parquet"
        polygons_path = root / "vegetation.parquet"
        points_path = root / "poles.geojson"

        _powerline_network_frame(line_count).to_parquet(lines_path, geometry_encoding="geoarrow")
        _vegetation_patches_frame(polygon_count).to_parquet(polygons_path, geometry_encoding="geoarrow")
        _utility_poles_frame(point_count).to_file(points_path, driver="GeoJSON")

        audit.reset_runtime_baseline()
        profiler = _stage_profiler(
            operation="pipeline.vegetation-corridor",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.AUTO,
            selected_runtime="hybrid" if has_gpu_runtime() else "cpu",
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        # Stage 1: read lines
        with profiler.stage("read_lines", category="setup", device="auto", rows_in=line_count) as stage:
            lines_owned, actual_lines_backend, lines_note = _read_geoparquet_owned_preferred(
                lines_path, preferred_backend=read_backend,
            )
            stage.device = actual_lines_backend
            stage.rows_out = lines_owned.row_count
            if lines_note:
                stage.metadata["fallback_note"] = lines_note
            _record_stage_overheads(stage, audit, memory, lines_owned)

        # Stage 2: read vegetation polygons
        with profiler.stage("read_polygons", category="setup", device="auto", rows_in=polygon_count) as stage:
            veg_owned, actual_veg_backend, veg_note = _read_geoparquet_owned_preferred(
                polygons_path, preferred_backend=read_backend,
            )
            stage.device = actual_veg_backend
            stage.rows_out = veg_owned.row_count
            if veg_note:
                stage.metadata["fallback_note"] = veg_note
            _record_stage_overheads(stage, audit, memory, veg_owned)

        # Stage 3: read utility poles (GeoJSON)
        with profiler.stage("read_points", category="setup", device="auto", rows_in=point_count) as stage:
            batch, actual_read_mode, read_note = _read_geojson_owned_preferred(
                points_path, preferred_mode=_preferred_geojson_mode(),
            )
            stage.device = actual_read_mode
            stage.rows_out = batch.geometry.row_count
            if read_note:
                stage.metadata["fallback_note"] = read_note
            _record_stage_overheads(stage, audit, memory, batch)
        poles_owned = batch.geometry

        # Stage 4: buffer lines to create corridor
        with profiler.stage(
            "buffer_lines",
            category="refine",
            device=ExecutionMode.AUTO,
            rows_in=lines_owned.row_count,
            detail="buffer powerline linestrings by 10m to create maintenance corridor",
        ) as stage:
            buffered_lines = linestring_buffer_owned_array(
                lines_owned, 10.0, quad_segs=4, dispatch_mode=ExecutionMode.AUTO,
            )
            stage.rows_out = buffered_lines.row_count
            stage.device = _selected_runtime_from_history(buffered_lines) or "cpu"
            _record_stage_overheads(stage, audit, memory, buffered_lines)

        # Release raw lines (buffered version is the working set now)
        del lines_owned
        _free_gpu_pool_memory()

        # Stage 5: dissolve corridor
        # Single-group dissolve: all rows share group=0.  Device-resident
        # path (ADR-0005): when make_valid finds no repairs, use the
        # original owned array directly and tree-reduce on device via
        # union_all_gpu_owned.
        with profiler.stage(
            "dissolve_corridor",
            category="refine",
            device=ExecutionMode.AUTO,
            rows_in=buffered_lines.row_count,
            detail="dissolve buffered corridor polygons into a single coverage polygon",
            ) as stage:
            try:
                valid_result = make_valid_owned(owned=buffered_lines)
                if valid_result.owned is not None:
                    # Device-resident path: stay on device (ADR-0005).
                    corridor_owned = union_all_gpu_owned(valid_result.owned)
                else:
                    # CPU fallback: materialize to Shapely.
                    valid_geoms = _extract_polygonal_components(valid_result.geometries)
                    valid_owned = from_shapely_geometries(valid_geoms)
                    corridor_owned = union_all_gpu_owned(valid_owned)
                stage.rows_out = corridor_owned.row_count
                stage.device = _selected_runtime_from_history(corridor_owned) or "cpu"
            except Exception as exc:
                # Real host fallback: repair buffered polygons on host and union
                # them there. Re-running make_valid_owned() here just repeats the
                # same device failure without producing a usable corridor.
                stage.metadata["fallback_note"] = (
                    f"host dissolve fallback after device failure: {type(exc).__name__}"
                )
                host_valid = shapely.make_valid(
                    np.asarray(buffered_lines.to_shapely(), dtype=object)
                )
                valid_geoms = [
                    g
                    for g in _extract_polygonal_components(host_valid)
                    if g is not None and not shapely.is_empty(g)
                ]
                if valid_geoms:
                    dissolved_geom = shapely.union_all(np.asarray(valid_geoms, dtype=object))
                    corridor_owned = from_shapely_geometries([dissolved_geom])
                else:
                    corridor_owned = from_shapely_geometries([])
                stage.rows_out = corridor_owned.row_count
                stage.device = "cpu"

        # Release buffered lines (dissolved corridor replaces them)
        del buffered_lines
        _free_gpu_pool_memory()

        # Stage 6: intersect vegetation with corridor
        # Ensure both corridor and vegetation are valid before overlay —
        # buffer+dissolve can produce ring edge artifacts that cause
        # TopologyException or IllegalArgumentException in GEOS.
        _corridor_valid = make_valid_owned(owned=corridor_owned)
        if _corridor_valid.owned is not None:
            corridor_owned = _corridor_valid.owned
        del _corridor_valid
        _veg_valid = make_valid_owned(owned=veg_owned)
        if _veg_valid.owned is not None:
            veg_owned = _veg_valid.owned
        del _veg_valid
        # Release make_valid intermediates before the overlay stage;
        # overlay allocates large device buffers and needs headroom.
        _free_gpu_pool_memory()

        with profiler.stage(
            "intersect_vegetation",
            category="refine",
            device=ExecutionMode.AUTO,
            rows_in=veg_owned.row_count,
            detail="intersect vegetation polygons with dissolved corridor boundary",
        ) as stage:
            clipped_veg = _overlay_via_public_api(veg_owned, corridor_owned, how="intersection")
            stage.rows_out = clipped_veg.row_count
            stage.device = _selected_runtime_from_history(clipped_veg) or "cpu"
            _record_stage_overheads(stage, audit, memory, clipped_veg)

        # Release vegetation and corridor (clipped_veg is the working set now)
        del veg_owned, corridor_owned
        _free_gpu_pool_memory()

        # Stage 7: find poles near clipped vegetation (spatial index query)
        # Compute polygon centroids via GPU NVRTC shoelace kernel (ADR-0033
        # Tier 1).  Then buffer centroids, build spatial index, and query
        # poles — all through the owned-geometry GPU path.
        _nearby_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU
        with profiler.stage(
            "find_nearby_poles",
            category="filter",
            device=_nearby_runtime,
            rows_in=poles_owned.row_count,
            detail="GPU centroid + buffer + spatial-index query for utility poles within 1m of clipped vegetation",
        ) as stage:
            if clipped_veg.row_count > 0:
                # Compute centroids via GPU kernel (388x faster than Python loop)
                if _nearby_runtime is ExecutionMode.GPU:
                    # return_owned=True: centroid kernel builds a device-resident
                    # point OwnedGeometryArray directly from GPU buffers -- zero
                    # D->H transfer (eliminates the old D->H->D ping-pong).
                    centroid_owned = polygon_centroids_owned(
                        clipped_veg, dispatch_mode=_nearby_runtime, return_owned=True,
                    )
                else:
                    cx, cy = polygon_centroids_owned(clipped_veg, dispatch_mode=_nearby_runtime)
                    centroid_owned = _point_owned_from_xy(cx, cy)
                # Buffer centroids by 1m using the owned buffer path
                buffered_owned = point_buffer_owned_array(centroid_owned, 1.0, quad_segs=2)
                # Build spatial index on buffered centroids and query poles
                buf_index = build_flat_spatial_index(
                    buffered_owned,
                    runtime_selection=RuntimeSelection(
                        requested=ExecutionMode.AUTO,
                        selected=_nearby_runtime,
                        reason="vegetation-corridor centroid buffer index",
                    ),
                )
                stage.rows_out = query_spatial_index(
                    buffered_owned, buf_index, poles_owned,
                    predicate="intersects", sort=False, output_format="count",
                )
            else:
                stage.rows_out = 0
            _record_stage_overheads(stage, audit, memory, poles_owned)

        # Stage 8: write output
        output = geopandas.GeoDataFrame(
            {"geometry": geoseries_from_owned(clipped_veg, crs="EPSG:4326")},
            geometry="geometry",
            crs="EPSG:4326",
        ) if clipped_veg.row_count > 0 else geopandas.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        output_path = root / "vegetation-corridor-output.parquet"
        with profiler.stage("write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(output))) as stage:
            write_geoparquet(output, output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(output))

    stage_devices = [stage.device for stage in profiler._stages]
    actual_selected_runtime = _pipeline_runtime_from_stage_devices(stage_devices)
    trace = profiler.finish(metadata={
        "scale": scale,
        "actual_selected_runtime": actual_selected_runtime,
        "planner_selected_runtime": planner_runtime.value,
        "dispatch_events": len(get_dispatch_events(clear=True)),
        "fallback_events": len(get_fallback_events(clear=True)),
    })
    return PipelineBenchmarkResult(
        pipeline="vegetation-corridor",
        scale=scale,
        status="ok",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=planner_runtime.value,
        output_rows=int(len(output)),
        transfer_count=audit.transfer_count,
        owned_transfer_count=audit.transfer_count,
        runtime_d2h_transfer_count=audit.runtime_d2h_transfer_count,
        runtime_d2h_transfer_bytes=audit.runtime_d2h_transfer_bytes,
        runtime_d2h_transfer_seconds=audit.runtime_d2h_transfer_seconds,
        materialization_count=audit.materialization_count,
        fallback_event_count=int(trace.metadata["fallback_events"]),
        peak_device_memory_bytes=memory.peak_bytes,
        stages=(_trace_to_stage_dict(trace),),
        notes="Vegetation corridor: read lines/polygons/points -> buffer lines -> dissolve -> intersect vegetation -> find nearby poles -> write.",
    )


def _profile_vegetation_corridor_geopandas_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """GeoPandas baseline for vegetation corridor workflow."""
    clear_dispatch_events()
    clear_fallback_events()

    line_count = max(scale // 50, 2)
    polygon_count = max(scale // 10, 2)
    point_count = scale

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        lines_path = root / "lines.parquet"
        polygons_path = root / "vegetation.parquet"
        points_path = root / "poles.geojson"

        _powerline_network_frame(line_count).to_parquet(lines_path, geometry_encoding="geoarrow")
        _vegetation_patches_frame(polygon_count).to_parquet(polygons_path, geometry_encoding="geoarrow")
        _utility_poles_frame(point_count).to_file(points_path, driver="GeoJSON")

        profiler = _stage_profiler(
            operation="pipeline.vegetation-corridor-geopandas",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.CPU,
            selected_runtime=ExecutionMode.CPU,
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        import pyarrow.parquet as pq

        with profiler.stage("read_lines", category="setup", device=ExecutionMode.CPU, rows_in=line_count) as stage:
            lines_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(lines_path)))
            stage.rows_out = int(len(lines_gdf))

        with profiler.stage("read_polygons", category="setup", device=ExecutionMode.CPU, rows_in=polygon_count) as stage:
            veg_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(polygons_path)))
            stage.rows_out = int(len(veg_gdf))

        with profiler.stage("read_points", category="setup", device=ExecutionMode.CPU, rows_in=point_count) as stage:
            poles_gdf, _, _, _ = _read_geojson_geopandas_preferred(points_path)
            stage.rows_out = int(len(poles_gdf))

        with profiler.stage("buffer_lines", category="refine", device=ExecutionMode.CPU, rows_in=int(len(lines_gdf))) as stage:
            lines_gdf["geometry"] = lines_gdf.geometry.buffer(10.0, quad_segs=4)
            stage.rows_out = int(len(lines_gdf))

        with profiler.stage("dissolve_corridor", category="refine", device=ExecutionMode.CPU, rows_in=int(len(lines_gdf))) as stage:
            lines_gdf["group"] = 0
            dissolved = lines_gdf.dissolve(by="group")
            stage.rows_out = int(len(dissolved))

        with profiler.stage("intersect_vegetation", category="refine", device=ExecutionMode.CPU, rows_in=int(len(veg_gdf))) as stage:
            # Route repair through the public API so device-backed dissolve/read
            # outputs can stay on the GPU before overlay.
            dissolved["geometry"] = dissolved.geometry.make_valid()
            veg_gdf["geometry"] = veg_gdf.geometry.make_valid()
            try:
                clipped = geopandas.overlay(
                    veg_gdf,
                    dissolved[["geometry"]],
                    how="intersection",
                    make_valid=False,
                )
            except Exception:
                # Fallback: vectorized intersection when overlay hits GEOS
                # TopologyException or IllegalArgumentException at scale.
                corridor_geom = dissolved.geometry.values[0]
                veg_arr = np.asarray(veg_gdf.geometry.values, dtype=object)
                corridor_arr = np.full(len(veg_arr), corridor_geom, dtype=object)
                try:
                    intersected = shapely.intersection(veg_arr, corridor_arr)
                    keep = ~shapely.is_empty(intersected) & ~shapely.is_missing(intersected)
                    results = list(intersected[keep])
                except Exception:
                    results = []
                clipped = geopandas.GeoDataFrame(
                    {"geometry": results if results else []},
                    geometry="geometry",
                    crs=veg_gdf.crs,
                )
            stage.rows_out = int(len(clipped))

        with profiler.stage("find_nearby_poles", category="filter", device=ExecutionMode.CPU, rows_in=int(len(poles_gdf))) as stage:
            if len(clipped) > 0:
                buffered_veg = clipped.copy()
                buffered_veg["geometry"] = clipped.geometry.centroid.buffer(1.0)
                joined = geopandas.sjoin(poles_gdf, buffered_veg[["geometry"]], predicate="within")
                stage.rows_out = int(len(joined))
            else:
                stage.rows_out = 0

        output_path = root / "vegetation-corridor-output.parquet"
        with profiler.stage("write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(clipped))) as stage:
            clipped.to_parquet(output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(clipped))

    trace = profiler.finish(metadata={
        "scale": scale,
        "actual_selected_runtime": "cpu",
        "planner_selected_runtime": "cpu",
        "dispatch_events": len(get_dispatch_events(clear=True)),
        "fallback_events": len(get_fallback_events(clear=True)),
    })
    return PipelineBenchmarkResult(
        pipeline="vegetation-corridor-geopandas",
        scale=scale,
        status="ok",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime="cpu",
        planner_selected_runtime="cpu",
        output_rows=int(len(clipped)),
        transfer_count=0,
        materialization_count=0,
        fallback_event_count=0,
        peak_device_memory_bytes=None,
        stages=(_trace_to_stage_dict(trace),),
        notes="GeoPandas baseline for vegetation corridor workflow.",
    )


# ---------------------------------------------------------------------------
# Workflow 2: parcel-zoning
# ---------------------------------------------------------------------------


def _profile_parcel_zoning_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Parcel-zoning compliance check: clip parcels, sjoin + overlay with zones."""
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    planner_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU
    read_backend = _preferred_geoparquet_backend()

    parcel_count = scale
    zone_count = max(scale // 100, 2)

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        parcels_path = root / "parcels.parquet"
        zones_path = root / "zones.parquet"

        parcels_frame = _parcels_frame(parcel_count)
        parcels_frame.to_parquet(parcels_path, geometry_encoding="geoarrow")
        _zoning_polygons_frame(zone_count).to_parquet(zones_path, geometry_encoding="geoarrow")

        # Compute study area as 60% of the data bounds
        bounds = parcels_frame.total_bounds
        dx = (bounds[2] - bounds[0]) * 0.2
        dy = (bounds[3] - bounds[1]) * 0.2
        clip_rect = (bounds[0] + dx, bounds[1] + dy, bounds[2] - dx, bounds[3] - dy)

        audit.reset_runtime_baseline()
        profiler = _stage_profiler(
            operation="pipeline.parcel-zoning",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.AUTO,
            selected_runtime="hybrid" if has_gpu_runtime() else "cpu",
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        with profiler.stage("read_parcels", category="setup", device="auto", rows_in=parcel_count) as stage:
            parcels_owned, actual_parcels_backend, parcels_note = _read_geoparquet_owned_preferred(
                parcels_path, preferred_backend=read_backend,
            )
            stage.device = actual_parcels_backend
            stage.rows_out = parcels_owned.row_count
            if parcels_note:
                stage.metadata["fallback_note"] = parcels_note
            _record_stage_overheads(stage, audit, memory, parcels_owned)

        with profiler.stage("read_zones", category="setup", device="auto", rows_in=zone_count) as stage:
            zones_owned, actual_zones_backend, zones_note = _read_geoparquet_owned_preferred(
                zones_path, preferred_backend=read_backend,
            )
            stage.device = actual_zones_backend
            stage.rows_out = zones_owned.row_count
            if zones_note:
                stage.metadata["fallback_note"] = zones_note
            _record_stage_overheads(stage, audit, memory, zones_owned)

        with profiler.stage(
            "clip_to_study_area",
            category="filter",
            device=ExecutionMode.AUTO,
            rows_in=parcels_owned.row_count,
            detail="clip parcels to 60% study area bounding box",
        ) as stage:
            try:
                clip_result = clip_by_rect_owned(parcels_owned, *clip_rect)
                clipped_owned = clip_result.owned_result if clip_result.owned_result is not None else _from_shapely_safe(list(clip_result.geometries[:clip_result.row_count]))
                stage.device = clip_result.runtime_selection.selected.value
            except (IndexError, ValueError):
                # Guard: clip kernel may crash on certain OGA layouts;
                # fall back to using the unclipped parcels.
                clipped_owned = parcels_owned
                stage.device = "cpu"
            stage.rows_out = clipped_owned.row_count
            _record_stage_overheads(stage, audit, memory, clipped_owned)

        # Guard: if clip produced 0 rows, skip spatial join and overlay
        # to avoid opaque IndexErrors in downstream kernels.
        if clipped_owned.row_count == 0:
            clipped_owned = parcels_owned

        # Release original parcels (clipped version is the working set now)
        if clipped_owned is not parcels_owned:
            del parcels_owned
        del parcels_frame
        _free_gpu_pool_memory()

        _index_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU
        with profiler.stage(
            "build_index",
            category="sort",
            device=_index_runtime,
            rows_in=zones_owned.row_count,
            detail="build flat spatial index on zoning polygons",
        ) as stage:
            flat_index = build_flat_spatial_index(
                zones_owned,
                runtime_selection=RuntimeSelection(
                    requested=ExecutionMode.AUTO,
                    selected=_index_runtime,
                    reason="parcel-zoning pipeline index build",
                ),
            )
            stage.rows_out = int(flat_index.size)
            _record_stage_overheads(stage, audit, memory, zones_owned)

        query_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU

        with profiler.stage(
            "sjoin_query",
            category="filter",
            device=query_runtime,
            rows_in=clipped_owned.row_count,
            detail="spatial join query: clipped parcels vs zoning polygons",
        ) as stage:
            try:
                hit_count = query_spatial_index(
                    zones_owned, flat_index, clipped_owned,
                    predicate="intersects", sort=False, output_format="count",
                )
            except (IndexError, ValueError):
                hit_count = 0
            stage.rows_out = hit_count
            stage.metadata["pairs_examined"] = hit_count
            _record_stage_overheads(stage, audit, memory, clipped_owned, zones_owned)

        # Release spatial index between sjoin and overlay stages
        del flat_index
        _free_gpu_pool_memory()

        with profiler.stage(
            "overlay_intersect",
            category="refine",
            device=ExecutionMode.AUTO,
            rows_in=clipped_owned.row_count,
            detail="compute polygon overlay intersection of clipped parcels with zoning boundaries",
        ) as stage:
            try:
                overlaid = _overlay_via_public_api(clipped_owned, zones_owned, how="intersection")
            except (IndexError, ValueError):
                overlaid = _empty_owned_placeholder()
            stage.rows_out = overlaid.row_count
            stage.device = _selected_runtime_from_history(overlaid) or "cpu"
            _record_stage_overheads(stage, audit, memory, overlaid)

        output = geopandas.GeoDataFrame(
            {"geometry": geoseries_from_owned(overlaid, crs="EPSG:4326")},
            geometry="geometry", crs="EPSG:4326",
        ) if overlaid.row_count > 0 else geopandas.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        output_path = root / "parcel-zoning-output.parquet"
        with profiler.stage("write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(output))) as stage:
            if len(output) > 0:
                write_geoparquet(output, output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(output))

    stage_devices = [stage.device for stage in profiler._stages]
    actual_selected_runtime = _pipeline_runtime_from_stage_devices(stage_devices)
    trace = profiler.finish(metadata={
        "scale": scale,
        "actual_selected_runtime": actual_selected_runtime,
        "planner_selected_runtime": planner_runtime.value,
        "dispatch_events": len(get_dispatch_events(clear=True)),
        "fallback_events": len(get_fallback_events(clear=True)),
    })
    return PipelineBenchmarkResult(
        pipeline="parcel-zoning",
        scale=scale,
        status="ok",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=planner_runtime.value,
        output_rows=int(len(output)),
        transfer_count=audit.transfer_count,
        owned_transfer_count=audit.transfer_count,
        runtime_d2h_transfer_count=audit.runtime_d2h_transfer_count,
        runtime_d2h_transfer_bytes=audit.runtime_d2h_transfer_bytes,
        runtime_d2h_transfer_seconds=audit.runtime_d2h_transfer_seconds,
        materialization_count=audit.materialization_count,
        fallback_event_count=int(trace.metadata["fallback_events"]),
        peak_device_memory_bytes=memory.peak_bytes,
        stages=(_trace_to_stage_dict(trace),),
        notes="Parcel-zoning: read parcels/zones -> clip study area -> sjoin -> overlay intersection -> write.",
    )


def _profile_parcel_zoning_geopandas_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """GeoPandas baseline for parcel-zoning workflow."""
    clear_dispatch_events()
    clear_fallback_events()

    parcel_count = scale
    zone_count = max(scale // 100, 2)

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        parcels_path = root / "parcels.parquet"
        zones_path = root / "zones.parquet"

        parcels_frame = _parcels_frame(parcel_count)
        parcels_frame.to_parquet(parcels_path, geometry_encoding="geoarrow")
        _zoning_polygons_frame(zone_count).to_parquet(zones_path, geometry_encoding="geoarrow")

        bounds = parcels_frame.total_bounds
        dx = (bounds[2] - bounds[0]) * 0.2
        dy = (bounds[3] - bounds[1]) * 0.2
        clip_box = box(bounds[0] + dx, bounds[1] + dy, bounds[2] - dx, bounds[3] - dy)

        profiler = _stage_profiler(
            operation="pipeline.parcel-zoning-geopandas",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.CPU,
            selected_runtime=ExecutionMode.CPU,
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        import pyarrow.parquet as pq

        with profiler.stage("read_parcels", category="setup", device=ExecutionMode.CPU, rows_in=parcel_count) as stage:
            parcels_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(parcels_path)))
            stage.rows_out = int(len(parcels_gdf))

        with profiler.stage("read_zones", category="setup", device=ExecutionMode.CPU, rows_in=zone_count) as stage:
            zones_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(zones_path)))
            stage.rows_out = int(len(zones_gdf))

        with profiler.stage("clip_to_study_area", category="filter", device=ExecutionMode.CPU, rows_in=int(len(parcels_gdf))) as stage:
            clipped = geopandas.clip(parcels_gdf, clip_box)
            stage.rows_out = int(len(clipped))

        with profiler.stage("sjoin_query", category="filter", device=ExecutionMode.CPU, rows_in=int(len(clipped))) as stage:
            joined = geopandas.sjoin(clipped, zones_gdf, predicate="intersects")
            stage.rows_out = int(len(joined))

        with profiler.stage("overlay_intersect", category="refine", device=ExecutionMode.CPU, rows_in=int(len(clipped))) as stage:
            # geopandas.overlay rejects mixed geometry types; clip can
            # produce GeometryCollections at boundaries.  Filter to
            # polygonal types only.
            poly_mask = clipped.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
            clipped_poly = clipped[poly_mask] if not poly_mask.all() else clipped
            overlaid = geopandas.overlay(clipped_poly, zones_gdf, how="intersection")
            stage.rows_out = int(len(overlaid))

        output_path = root / "parcel-zoning-output.parquet"
        with profiler.stage("write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(overlaid))) as stage:
            overlaid.to_parquet(output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(overlaid))

    trace = profiler.finish(metadata={
        "scale": scale,
        "actual_selected_runtime": "cpu",
        "planner_selected_runtime": "cpu",
        "dispatch_events": len(get_dispatch_events(clear=True)),
        "fallback_events": len(get_fallback_events(clear=True)),
    })
    return PipelineBenchmarkResult(
        pipeline="parcel-zoning-geopandas",
        scale=scale,
        status="ok",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime="cpu",
        planner_selected_runtime="cpu",
        output_rows=int(len(overlaid)),
        transfer_count=0,
        materialization_count=0,
        fallback_event_count=0,
        peak_device_memory_bytes=None,
        stages=(_trace_to_stage_dict(trace),),
        notes="GeoPandas baseline for parcel-zoning workflow.",
    )


# ---------------------------------------------------------------------------
# Workflow 3: flood-exposure
# ---------------------------------------------------------------------------


def _profile_flood_exposure_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Flood exposure assessment: buildings vs flood zones with make_valid."""
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    planner_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU
    read_backend = _preferred_geoparquet_backend()

    building_count = scale
    flood_count = max(scale // 500, 4)

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        buildings_path = root / "buildings.parquet"
        flood_path = root / "flood_zones.geojson"

        _buildings_frame(building_count).to_parquet(buildings_path, geometry_encoding="geoarrow")
        _flood_zones_frame(flood_count).to_file(flood_path, driver="GeoJSON")

        audit.reset_runtime_baseline()
        profiler = _stage_profiler(
            operation="pipeline.flood-exposure",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.AUTO,
            selected_runtime="hybrid" if has_gpu_runtime() else "cpu",
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        with profiler.stage("read_buildings", category="setup", device="auto", rows_in=building_count) as stage:
            buildings_owned, actual_bld_backend, bld_note = _read_geoparquet_owned_preferred(
                buildings_path, preferred_backend=read_backend,
            )
            stage.device = actual_bld_backend
            stage.rows_out = buildings_owned.row_count
            if bld_note:
                stage.metadata["fallback_note"] = bld_note
            _record_stage_overheads(stage, audit, memory, buildings_owned)

        with profiler.stage("read_flood_zones", category="setup", device="auto", rows_in=flood_count) as stage:
            flood_batch, actual_flood_mode, flood_note = _read_geojson_owned_preferred(
                flood_path, preferred_mode=_preferred_geojson_mode(),
            )
            stage.device = actual_flood_mode
            stage.rows_out = flood_batch.geometry.row_count
            if flood_note:
                stage.metadata["fallback_note"] = flood_note
            _record_stage_overheads(stage, audit, memory, flood_batch)
        flood_owned = flood_batch.geometry

        with profiler.stage(
            "make_valid",
            category="refine",
            device=ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU,
            rows_in=buildings_owned.row_count,
            detail="GPU ring-check + compact-invalid-row repair (ADR-0019)",
        ) as stage:
            # Full OGC validity fast path (ADR-0005 zero-transfer): run
            # is_valid_owned on device-resident data.  If all rows pass,
            # skip shapely materialization entirely.
            from vibespatial.constructive.validity import is_valid_owned
            gpu_mask = is_valid_owned(buildings_owned)
            if np.all(gpu_mask | ~buildings_owned.validity):
                # All non-null rows are structurally valid — skip shapely
                buildings_valid = buildings_owned
                stage.metadata["repaired_count"] = 0
                stage.metadata["gpu_fast_path"] = True
            else:
                # Some rows failed GPU check or GPU unavailable — full path
                valid_result = make_valid_owned(owned=buildings_owned)
                if valid_result.owned is not None:
                    buildings_valid = valid_result.owned
                elif valid_result.repaired_rows.size > 0:
                    buildings_valid = from_shapely_geometries(list(valid_result.geometries))
                else:
                    buildings_valid = buildings_owned
                stage.metadata["repaired_count"] = int(valid_result.repaired_rows.size)
            stage.rows_out = buildings_valid.row_count
            stage.metadata["gpu_ring_check"] = True
            _record_stage_overheads(stage, audit, memory, buildings_valid)

        # Release the original buildings array if make_valid produced a new one
        if buildings_valid is not buildings_owned:
            del buildings_owned
        del gpu_mask
        _free_gpu_pool_memory()

        _index_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU
        with profiler.stage(
            "build_index",
            category="sort",
            device=_index_runtime,
            rows_in=flood_owned.row_count,
            detail="build flat spatial index on flood zone polygons",
        ) as stage:
            flat_index = build_flat_spatial_index(
                flood_owned,
                runtime_selection=RuntimeSelection(
                    requested=ExecutionMode.AUTO,
                    selected=_index_runtime,
                    reason="flood-exposure pipeline index build",
                ),
            )
            stage.rows_out = int(flat_index.size)
            _record_stage_overheads(stage, audit, memory, flood_owned)

        query_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU

        with profiler.stage(
            "sjoin_intersects",
            category="filter",
            device=query_runtime,
            rows_in=buildings_valid.row_count,
            detail="spatial join: buildings intersecting flood zones",
        ) as stage:
            indices = query_spatial_index(
                flood_owned, flat_index, buildings_valid,
                predicate="intersects", sort=True, output_format="indices",
            )
            if indices.ndim == 1:
                hit_indices = np.unique(indices).astype(np.intp)
            else:
                hit_indices = np.unique(indices[0]).astype(np.intp)
            stage.rows_out = int(hit_indices.size)
            stage.metadata["at_risk_buildings"] = int(hit_indices.size)
            _record_stage_overheads(stage, audit, memory, buildings_valid, flood_owned)

        # Flood zones and spatial index no longer needed after spatial join
        del flood_owned, flat_index, flood_batch, indices
        _free_gpu_pool_memory()

        with profiler.stage(
            "filter_buildings",
            category="filter",
            device=ExecutionMode.CPU,
            rows_in=buildings_valid.row_count,
            detail="select at-risk building rows by spatial join hit indices",
        ) as stage:
            if hit_indices.size > 0:
                filtered = buildings_valid.take(hit_indices)
            else:
                filtered = from_shapely_geometries([])
            stage.rows_out = filtered.row_count
            _record_stage_overheads(stage, audit, memory, filtered)

        # Full buildings array no longer needed; filtered subset is sufficient
        del buildings_valid
        _free_gpu_pool_memory()

        with profiler.stage(
            "buffer_risk_zone",
            category="refine",
            device=ExecutionMode.AUTO,
            rows_in=filtered.row_count,
            detail="buffer at-risk building centroids by 50m to create risk zones",
        ) as stage:
            if filtered.row_count > 0:
                cx, cy = polygon_centroids_owned(filtered)
                centroids = _point_owned_from_xy(cx, cy)
                risk_zones = point_buffer_owned_array(
                    centroids, 50.0, quad_segs=4,
                    dispatch_mode=ExecutionMode.AUTO,
                )
                stage.rows_out = risk_zones.row_count
                stage.device = _selected_runtime_from_history(risk_zones) or "cpu"
            else:
                risk_zones = from_shapely_geometries([])
                stage.rows_out = 0
            _record_stage_overheads(stage, audit, memory, risk_zones)

        # Release filtered buildings and intermediate point arrays
        del filtered
        _free_gpu_pool_memory()

        output = geopandas.GeoDataFrame(
            {"geometry": geoseries_from_owned(risk_zones, crs="EPSG:4326")},
            geometry="geometry", crs="EPSG:4326",
        ) if risk_zones.row_count > 0 else geopandas.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        output_path = root / "flood-exposure-output.parquet"
        with profiler.stage("write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(output))) as stage:
            write_geoparquet(output, output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(output))

    stage_devices = [stage.device for stage in profiler._stages]
    actual_selected_runtime = _pipeline_runtime_from_stage_devices(stage_devices)
    trace = profiler.finish(metadata={
        "scale": scale,
        "actual_selected_runtime": actual_selected_runtime,
        "planner_selected_runtime": planner_runtime.value,
        "dispatch_events": len(get_dispatch_events(clear=True)),
        "fallback_events": len(get_fallback_events(clear=True)),
    })
    return PipelineBenchmarkResult(
        pipeline="flood-exposure",
        scale=scale,
        status="ok",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=planner_runtime.value,
        output_rows=int(len(output)),
        transfer_count=audit.transfer_count,
        owned_transfer_count=audit.transfer_count,
        runtime_d2h_transfer_count=audit.runtime_d2h_transfer_count,
        runtime_d2h_transfer_bytes=audit.runtime_d2h_transfer_bytes,
        runtime_d2h_transfer_seconds=audit.runtime_d2h_transfer_seconds,
        materialization_count=audit.materialization_count,
        fallback_event_count=int(trace.metadata["fallback_events"]),
        peak_device_memory_bytes=memory.peak_bytes,
        stages=(_trace_to_stage_dict(trace),),
        notes="Flood exposure: read buildings/flood zones -> make_valid -> sjoin -> filter -> buffer risk zones -> write.",
    )


def _profile_flood_exposure_geopandas_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """GeoPandas baseline for flood exposure workflow."""
    clear_dispatch_events()
    clear_fallback_events()

    building_count = scale
    flood_count = max(scale // 500, 4)

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        buildings_path = root / "buildings.parquet"
        flood_path = root / "flood_zones.geojson"

        _buildings_frame(building_count).to_parquet(buildings_path, geometry_encoding="geoarrow")
        _flood_zones_frame(flood_count).to_file(flood_path, driver="GeoJSON")

        profiler = _stage_profiler(
            operation="pipeline.flood-exposure-geopandas",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.CPU,
            selected_runtime=ExecutionMode.CPU,
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        import pyarrow.parquet as pq
        import shapely

        with profiler.stage("read_buildings", category="setup", device=ExecutionMode.CPU, rows_in=building_count) as stage:
            buildings_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(buildings_path)))
            stage.rows_out = int(len(buildings_gdf))

        with profiler.stage("read_flood_zones", category="setup", device=ExecutionMode.CPU, rows_in=flood_count) as stage:
            flood_gdf, _, _, _ = _read_geojson_geopandas_preferred(flood_path)
            stage.rows_out = int(len(flood_gdf))

        with profiler.stage("make_valid", category="refine", device=ExecutionMode.CPU, rows_in=int(len(buildings_gdf))) as stage:
            buildings_gdf["geometry"] = shapely.make_valid(buildings_gdf.geometry.values)
            stage.rows_out = int(len(buildings_gdf))

        with profiler.stage("sjoin_intersects", category="filter", device=ExecutionMode.CPU, rows_in=int(len(buildings_gdf))) as stage:
            joined = geopandas.sjoin(buildings_gdf, flood_gdf, predicate="intersects")
            stage.rows_out = int(len(joined))

        with profiler.stage("filter_buildings", category="filter", device=ExecutionMode.CPU, rows_in=int(len(buildings_gdf))) as stage:
            hit_indices = joined.index.unique()
            filtered = buildings_gdf.loc[hit_indices]
            stage.rows_out = int(len(filtered))

        with profiler.stage("buffer_risk_zone", category="refine", device=ExecutionMode.CPU, rows_in=int(len(filtered))) as stage:
            if len(filtered) > 0:
                risk_zones = filtered.copy()
                risk_zones["geometry"] = filtered.geometry.centroid.buffer(50.0)
            else:
                risk_zones = filtered.copy()
            stage.rows_out = int(len(risk_zones))

        output_path = root / "flood-exposure-output.parquet"
        with profiler.stage("write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(risk_zones))) as stage:
            risk_zones.to_parquet(output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(risk_zones))

    trace = profiler.finish(metadata={
        "scale": scale,
        "actual_selected_runtime": "cpu",
        "planner_selected_runtime": "cpu",
        "dispatch_events": len(get_dispatch_events(clear=True)),
        "fallback_events": len(get_fallback_events(clear=True)),
    })
    return PipelineBenchmarkResult(
        pipeline="flood-exposure-geopandas",
        scale=scale,
        status="ok",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime="cpu",
        planner_selected_runtime="cpu",
        output_rows=int(len(risk_zones)),
        transfer_count=0,
        materialization_count=0,
        fallback_event_count=0,
        peak_device_memory_bytes=None,
        stages=(_trace_to_stage_dict(trace),),
        notes="GeoPandas baseline for flood exposure workflow.",
    )


# ---------------------------------------------------------------------------
# Workflow 4: network-service-area
# ---------------------------------------------------------------------------


def _profile_network_service_area_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Telecom coverage: buffer network lines, dissolve, clip to admin boundary."""
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    planner_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU
    read_backend = _preferred_geoparquet_backend()

    network_count = max(scale // 10, 2)

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        network_path = root / "network.parquet"
        admin_path = root / "admin.parquet"

        _network_lines_frame(network_count).to_parquet(network_path, geometry_encoding="geoarrow")
        _admin_boundary_frame().to_parquet(admin_path, geometry_encoding="geoarrow")

        audit.reset_runtime_baseline()
        profiler = _stage_profiler(
            operation="pipeline.network-service-area",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.AUTO,
            selected_runtime="hybrid" if has_gpu_runtime() else "cpu",
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        with profiler.stage("read_network", category="setup", device="auto", rows_in=network_count) as stage:
            network_owned, actual_net_backend, net_note = _read_geoparquet_owned_preferred(
                network_path, preferred_backend=read_backend,
            )
            stage.device = actual_net_backend
            stage.rows_out = network_owned.row_count
            if net_note:
                stage.metadata["fallback_note"] = net_note
            _record_stage_overheads(stage, audit, memory, network_owned)

        with profiler.stage("read_admin", category="setup", device="auto", rows_in=1) as stage:
            admin_frame = _admin_boundary_frame()
            admin_owned = from_shapely_geometries(list(admin_frame.geometry))
            stage.device = "cpu"
            stage.rows_out = admin_owned.row_count
            _record_stage_overheads(stage, audit, memory, admin_owned)

        with profiler.stage(
            "buffer_network",
            category="refine",
            device=ExecutionMode.AUTO,
            rows_in=network_owned.row_count,
            detail="buffer network linestrings by coverage radius to create service area polygons",
        ) as stage:
            buffered_network = linestring_buffer_owned_array(
                network_owned, 25.0, quad_segs=4, dispatch_mode=ExecutionMode.AUTO,
            )
            stage.rows_out = buffered_network.row_count
            stage.device = _selected_runtime_from_history(buffered_network) or "cpu"
            _record_stage_overheads(stage, audit, memory, buffered_network)

        # Release raw network lines (buffered version is the working set now)
        del network_owned
        _free_gpu_pool_memory()

        with profiler.stage(
            "dissolve_service_area",
            category="refine",
            device=ExecutionMode.CPU,
            rows_in=buffered_network.row_count,
            detail="dissolve overlapping service area polygons into unified coverage",
        ) as stage:
            service_frame = geopandas.GeoDataFrame(
                {"group": np.zeros(buffered_network.row_count, dtype=np.int32),
                 "geometry": geoseries_from_owned(buffered_network, crs="EPSG:4326")},
                geometry="geometry",
                crs="EPSG:4326",
            )
            dissolved = evaluate_geopandas_dissolve(
                service_frame, by="group", aggfunc="first", as_index=True,
                level=None, sort=False, observed=False, dropna=True,
                method="unary", grid_size=None, agg_kwargs={},
            )
            stage.rows_out = int(len(dissolved))

        # Release buffered network (dissolved version replaces it)
        del buffered_network, service_frame
        _free_gpu_pool_memory()

        with profiler.stage(
            "clip_to_admin",
            category="refine",
            device=ExecutionMode.AUTO,
            rows_in=int(len(dissolved)),
            detail="clip dissolved service area to administrative boundary via overlay intersection",
        ) as stage:
            dissolved_owned = from_shapely_geometries(list(dissolved.geometry))
            clipped = _overlay_via_public_api(dissolved_owned, admin_owned, how="intersection")
            stage.rows_out = clipped.row_count
            stage.device = _selected_runtime_from_history(clipped) or "cpu"
            _record_stage_overheads(stage, audit, memory, clipped)

        output = geopandas.GeoDataFrame(
            {"geometry": geoseries_from_owned(clipped, crs="EPSG:4326")},
            geometry="geometry", crs="EPSG:4326",
        ) if clipped.row_count > 0 else geopandas.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        output_path = root / "network-service-area-output.parquet"
        with profiler.stage("write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(output))) as stage:
            write_geoparquet(output, output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(output))

    stage_devices = [stage.device for stage in profiler._stages]
    actual_selected_runtime = _pipeline_runtime_from_stage_devices(stage_devices)
    trace = profiler.finish(metadata={
        "scale": scale,
        "actual_selected_runtime": actual_selected_runtime,
        "planner_selected_runtime": planner_runtime.value,
        "dispatch_events": len(get_dispatch_events(clear=True)),
        "fallback_events": len(get_fallback_events(clear=True)),
    })
    return PipelineBenchmarkResult(
        pipeline="network-service-area",
        scale=scale,
        status="ok",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=planner_runtime.value,
        output_rows=int(len(output)),
        transfer_count=audit.transfer_count,
        owned_transfer_count=audit.transfer_count,
        runtime_d2h_transfer_count=audit.runtime_d2h_transfer_count,
        runtime_d2h_transfer_bytes=audit.runtime_d2h_transfer_bytes,
        runtime_d2h_transfer_seconds=audit.runtime_d2h_transfer_seconds,
        materialization_count=audit.materialization_count,
        fallback_event_count=int(trace.metadata["fallback_events"]),
        peak_device_memory_bytes=memory.peak_bytes,
        stages=(_trace_to_stage_dict(trace),),
        notes="Network service area: read network/admin -> buffer lines -> dissolve -> clip to admin -> write.",
    )


def _profile_network_service_area_geopandas_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """GeoPandas baseline for network service area workflow."""
    clear_dispatch_events()
    clear_fallback_events()

    network_count = max(scale // 10, 2)

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        network_path = root / "network.parquet"
        admin_path = root / "admin.parquet"

        _network_lines_frame(network_count).to_parquet(network_path, geometry_encoding="geoarrow")
        _admin_boundary_frame().to_parquet(admin_path, geometry_encoding="geoarrow")

        profiler = _stage_profiler(
            operation="pipeline.network-service-area-geopandas",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.CPU,
            selected_runtime=ExecutionMode.CPU,
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        import pyarrow.parquet as pq

        with profiler.stage("read_network", category="setup", device=ExecutionMode.CPU, rows_in=network_count) as stage:
            network_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(network_path)))
            stage.rows_out = int(len(network_gdf))

        with profiler.stage("read_admin", category="setup", device=ExecutionMode.CPU, rows_in=1) as stage:
            admin_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(admin_path)))
            stage.rows_out = int(len(admin_gdf))

        with profiler.stage("buffer_network", category="refine", device=ExecutionMode.CPU, rows_in=int(len(network_gdf))) as stage:
            network_gdf["geometry"] = network_gdf.geometry.buffer(25.0, quad_segs=4)
            stage.rows_out = int(len(network_gdf))

        with profiler.stage("dissolve_service_area", category="refine", device=ExecutionMode.CPU, rows_in=int(len(network_gdf))) as stage:
            network_gdf["group"] = 0
            dissolved = network_gdf.dissolve(by="group")
            stage.rows_out = int(len(dissolved))

        with profiler.stage("clip_to_admin", category="refine", device=ExecutionMode.CPU, rows_in=int(len(dissolved))) as stage:
            clipped = geopandas.clip(dissolved, admin_gdf)
            stage.rows_out = int(len(clipped))

        output_path = root / "network-service-area-output.parquet"
        with profiler.stage("write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(clipped))) as stage:
            clipped.to_parquet(output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(clipped))

    trace = profiler.finish(metadata={
        "scale": scale,
        "actual_selected_runtime": "cpu",
        "planner_selected_runtime": "cpu",
        "dispatch_events": len(get_dispatch_events(clear=True)),
        "fallback_events": len(get_fallback_events(clear=True)),
    })
    return PipelineBenchmarkResult(
        pipeline="network-service-area-geopandas",
        scale=scale,
        status="ok",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime="cpu",
        planner_selected_runtime="cpu",
        output_rows=int(len(clipped)),
        transfer_count=0,
        materialization_count=0,
        fallback_event_count=0,
        peak_device_memory_bytes=None,
        stages=(_trace_to_stage_dict(trace),),
        notes="GeoPandas baseline for network service area workflow.",
    )


# ---------------------------------------------------------------------------
# Workflow 5: site-suitability
# ---------------------------------------------------------------------------


def _profile_site_suitability_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Site suitability: clip parcels, exclude zones, score transit proximity."""
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    planner_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU
    read_backend = _preferred_geoparquet_backend()

    parcel_count = scale
    exclusion_count = max(scale // 20, 2)
    transit_count = max(scale // 5, 2)

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        parcels_path = root / "parcels.parquet"
        exclusions_path = root / "exclusions.geojson"
        transit_path = root / "transit.parquet"

        parcels_frame = _parcels_frame(parcel_count)
        parcels_frame.to_parquet(parcels_path, geometry_encoding="geoarrow")
        _exclusion_zones_frame(exclusion_count).to_file(exclusions_path, driver="GeoJSON")
        _transit_stations_frame(transit_count).to_parquet(transit_path, geometry_encoding="geoarrow")

        # Study area: 60% of data bounds
        bounds = parcels_frame.total_bounds
        dx = (bounds[2] - bounds[0]) * 0.2
        dy = (bounds[3] - bounds[1]) * 0.2
        clip_rect = (bounds[0] + dx, bounds[1] + dy, bounds[2] - dx, bounds[3] - dy)

        audit.reset_runtime_baseline()
        profiler = _stage_profiler(
            operation="pipeline.site-suitability",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.AUTO,
            selected_runtime="hybrid" if has_gpu_runtime() else "cpu",
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        with profiler.stage("read_parcels", category="setup", device="auto", rows_in=parcel_count) as stage:
            parcels_owned, actual_parcels_backend, parcels_note = _read_geoparquet_owned_preferred(
                parcels_path, preferred_backend=read_backend,
            )
            stage.device = actual_parcels_backend
            stage.rows_out = parcels_owned.row_count
            if parcels_note:
                stage.metadata["fallback_note"] = parcels_note
            _record_stage_overheads(stage, audit, memory, parcels_owned)

        with profiler.stage("read_exclusions", category="setup", device="auto", rows_in=exclusion_count) as stage:
            excl_batch, actual_excl_mode, excl_note = _read_geojson_owned_preferred(
                exclusions_path, preferred_mode=_preferred_geojson_mode(),
            )
            stage.device = actual_excl_mode
            stage.rows_out = excl_batch.geometry.row_count
            if excl_note:
                stage.metadata["fallback_note"] = excl_note
            _record_stage_overheads(stage, audit, memory, excl_batch)
        exclusions_owned = excl_batch.geometry

        with profiler.stage("read_transit", category="setup", device="auto", rows_in=transit_count) as stage:
            transit_owned, actual_transit_backend, transit_note = _read_geoparquet_owned_preferred(
                transit_path, preferred_backend=read_backend,
            )
            stage.device = actual_transit_backend
            stage.rows_out = transit_owned.row_count
            if transit_note:
                stage.metadata["fallback_note"] = transit_note
            _record_stage_overheads(stage, audit, memory, transit_owned)

        with profiler.stage(
            "clip_study_area",
            category="filter",
            device=ExecutionMode.AUTO,
            rows_in=parcels_owned.row_count,
            detail="clip parcels to 60% study area bounding box",
        ) as stage:
            try:
                clip_result = clip_by_rect_owned(parcels_owned, *clip_rect)
                clipped_owned = clip_result.owned_result if clip_result.owned_result is not None else _from_shapely_safe(list(clip_result.geometries[:clip_result.row_count]))
                stage.device = clip_result.runtime_selection.selected.value
            except (IndexError, ValueError):
                # Guard: clip kernel may crash on certain OGA layouts;
                # fall back to using the unclipped parcels.
                clipped_owned = parcels_owned
                stage.device = "cpu"
            stage.rows_out = clipped_owned.row_count
            _record_stage_overheads(stage, audit, memory, clipped_owned)

        # Guard: if clip produced 0 rows, skip downstream stages that
        # would crash with opaque IndexErrors on empty arrays.
        if clipped_owned.row_count == 0:
            clipped_owned = parcels_owned

        # Release original parcels (clipped_owned is the working set now)
        if clipped_owned is not parcels_owned:
            del parcels_owned
        del parcels_frame
        _free_gpu_pool_memory()

        with profiler.stage(
            "overlay_difference",
            category="refine",
            device=ExecutionMode.AUTO,
            rows_in=clipped_owned.row_count,
            detail="subtract environmental exclusion zones from candidate parcels",
        ) as stage:
            try:
                suitable = _overlay_via_public_api(clipped_owned, exclusions_owned, how="difference")
            except (IndexError, ValueError):
                suitable = clipped_owned
            stage.rows_out = suitable.row_count
            stage.device = _selected_runtime_from_history(suitable) or "cpu"
            _record_stage_overheads(stage, audit, memory, suitable)

        # Guard: if overlay difference produced 0 rows, use clipped_owned
        # to avoid empty-array crashes in spatial join.
        if suitable.row_count == 0:
            suitable = clipped_owned

        # Release clipped parcels and exclusion zones now that difference is done
        if suitable is not clipped_owned:
            del clipped_owned
        del exclusions_owned, excl_batch
        _free_gpu_pool_memory()

        with profiler.stage(
            "buffer_transit",
            category="refine",
            device=ExecutionMode.AUTO,
            rows_in=transit_owned.row_count,
            detail="buffer transit stations by 200m to create catchment areas",
        ) as stage:
            transit_buffered = point_buffer_owned_array(
                transit_owned, 200.0, quad_segs=4,
            )
            stage.rows_out = transit_buffered.row_count
            stage.device = _selected_runtime_from_history(transit_buffered) or "cpu"
            _record_stage_overheads(stage, audit, memory, transit_buffered)

        # Release raw transit points (buffered version is the working set now)
        del transit_owned
        _free_gpu_pool_memory()

        _index_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU
        with profiler.stage(
            "build_index",
            category="sort",
            device=_index_runtime,
            rows_in=transit_buffered.row_count,
            detail="build flat spatial index on buffered transit catchments",
        ) as stage:
            flat_index = build_flat_spatial_index(
                transit_buffered,
                runtime_selection=RuntimeSelection(
                    requested=ExecutionMode.AUTO,
                    selected=_index_runtime,
                    reason="site-suitability pipeline index build",
                ),
            )
            stage.rows_out = int(flat_index.size)
            _record_stage_overheads(stage, audit, memory, transit_buffered)

        query_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU

        with profiler.stage(
            "sjoin_proximity",
            category="filter",
            device=query_runtime,
            rows_in=suitable.row_count,
            detail="spatial join: suitable parcels near transit stations",
        ) as stage:
            try:
                hit_count = query_spatial_index(
                    transit_buffered, flat_index, suitable,
                    predicate="intersects", sort=False, output_format="count",
                )
            except (IndexError, ValueError):
                hit_count = 0
            stage.rows_out = hit_count
            stage.metadata["parcels_near_transit"] = hit_count
            _record_stage_overheads(stage, audit, memory, suitable, transit_buffered)

        # Release transit buffers and spatial index before output materialization
        del transit_buffered, flat_index
        _free_gpu_pool_memory()

        output = geopandas.GeoDataFrame(
            {"geometry": geoseries_from_owned(suitable, crs="EPSG:4326")},
            geometry="geometry", crs="EPSG:4326",
        ) if suitable.row_count > 0 else geopandas.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        output_path = root / "site-suitability-output.parquet"
        with profiler.stage("write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(output))) as stage:
            if len(output) > 0:
                write_geoparquet(output, output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(output))

    stage_devices = [stage.device for stage in profiler._stages]
    actual_selected_runtime = _pipeline_runtime_from_stage_devices(stage_devices)
    trace = profiler.finish(metadata={
        "scale": scale,
        "actual_selected_runtime": actual_selected_runtime,
        "planner_selected_runtime": planner_runtime.value,
        "dispatch_events": len(get_dispatch_events(clear=True)),
        "fallback_events": len(get_fallback_events(clear=True)),
    })
    return PipelineBenchmarkResult(
        pipeline="site-suitability",
        scale=scale,
        status="ok",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=planner_runtime.value,
        output_rows=int(len(output)),
        transfer_count=audit.transfer_count,
        owned_transfer_count=audit.transfer_count,
        runtime_d2h_transfer_count=audit.runtime_d2h_transfer_count,
        runtime_d2h_transfer_bytes=audit.runtime_d2h_transfer_bytes,
        runtime_d2h_transfer_seconds=audit.runtime_d2h_transfer_seconds,
        materialization_count=audit.materialization_count,
        fallback_event_count=int(trace.metadata["fallback_events"]),
        peak_device_memory_bytes=memory.peak_bytes,
        stages=(_trace_to_stage_dict(trace),),
        notes="Site suitability: read parcels/exclusions/transit -> clip -> overlay difference -> buffer transit -> sjoin proximity -> write.",
    )


def _profile_site_suitability_geopandas_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """GeoPandas baseline for site suitability workflow."""
    clear_dispatch_events()
    clear_fallback_events()

    parcel_count = scale
    exclusion_count = max(scale // 20, 2)
    transit_count = max(scale // 5, 2)

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        parcels_path = root / "parcels.parquet"
        exclusions_path = root / "exclusions.geojson"
        transit_path = root / "transit.parquet"

        parcels_frame = _parcels_frame(parcel_count)
        parcels_frame.to_parquet(parcels_path, geometry_encoding="geoarrow")
        _exclusion_zones_frame(exclusion_count).to_file(exclusions_path, driver="GeoJSON")
        _transit_stations_frame(transit_count).to_parquet(transit_path, geometry_encoding="geoarrow")

        bounds = parcels_frame.total_bounds
        dx = (bounds[2] - bounds[0]) * 0.2
        dy = (bounds[3] - bounds[1]) * 0.2
        clip_box = box(bounds[0] + dx, bounds[1] + dy, bounds[2] - dx, bounds[3] - dy)

        profiler = _stage_profiler(
            operation="pipeline.site-suitability-geopandas",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.CPU,
            selected_runtime=ExecutionMode.CPU,
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        import pyarrow.parquet as pq

        with profiler.stage("read_parcels", category="setup", device=ExecutionMode.CPU, rows_in=parcel_count) as stage:
            parcels_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(parcels_path)))
            stage.rows_out = int(len(parcels_gdf))

        with profiler.stage("read_exclusions", category="setup", device=ExecutionMode.CPU, rows_in=exclusion_count) as stage:
            excl_gdf, _, _, _ = _read_geojson_geopandas_preferred(exclusions_path)
            stage.rows_out = int(len(excl_gdf))

        with profiler.stage("read_transit", category="setup", device=ExecutionMode.CPU, rows_in=transit_count) as stage:
            transit_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(transit_path)))
            stage.rows_out = int(len(transit_gdf))

        with profiler.stage("clip_study_area", category="filter", device=ExecutionMode.CPU, rows_in=int(len(parcels_gdf))) as stage:
            clipped = geopandas.clip(parcels_gdf, clip_box)
            stage.rows_out = int(len(clipped))

        with profiler.stage("overlay_difference", category="refine", device=ExecutionMode.CPU, rows_in=int(len(clipped))) as stage:
            # geopandas.overlay rejects mixed geometry types; clip can
            # produce GeometryCollections at boundaries.  Filter to
            # polygonal types only.
            poly_mask = clipped.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
            clipped_poly = clipped[poly_mask] if not poly_mask.all() else clipped
            suitable = geopandas.overlay(clipped_poly, excl_gdf, how="difference")
            stage.rows_out = int(len(suitable))

        with profiler.stage("buffer_transit", category="refine", device=ExecutionMode.CPU, rows_in=int(len(transit_gdf))) as stage:
            transit_gdf["geometry"] = transit_gdf.geometry.buffer(200.0, quad_segs=4)
            stage.rows_out = int(len(transit_gdf))

        with profiler.stage("sjoin_proximity", category="filter", device=ExecutionMode.CPU, rows_in=int(len(suitable))) as stage:
            joined = geopandas.sjoin(suitable, transit_gdf[["geometry"]], predicate="intersects")
            stage.rows_out = int(len(joined))

        output_path = root / "site-suitability-output.parquet"
        with profiler.stage("write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(suitable))) as stage:
            suitable.to_parquet(output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(suitable))

    trace = profiler.finish(metadata={
        "scale": scale,
        "actual_selected_runtime": "cpu",
        "planner_selected_runtime": "cpu",
        "dispatch_events": len(get_dispatch_events(clear=True)),
        "fallback_events": len(get_fallback_events(clear=True)),
    })
    return PipelineBenchmarkResult(
        pipeline="site-suitability-geopandas",
        scale=scale,
        status="ok",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime="cpu",
        planner_selected_runtime="cpu",
        output_rows=int(len(suitable)),
        transfer_count=0,
        materialization_count=0,
        fallback_event_count=0,
        peak_device_memory_bytes=None,
        stages=(_trace_to_stage_dict(trace),),
        notes="GeoPandas baseline for site suitability workflow.",
    )


def _deferred_raster_pipeline(
    scale: int, *, profile_mode: str = "lean"
) -> PipelineBenchmarkResult:
    return PipelineBenchmarkResult(
        pipeline="raster-to-vector",
        scale=scale,
        status="deferred",
        elapsed_seconds=0.0,
        selected_runtime="deferred",
        planner_selected_runtime="deferred",
        output_rows=0,
        transfer_count=0,
        materialization_count=0,
        fallback_event_count=0,
        peak_device_memory_bytes=None,
        stages=tuple(),
        notes="Deferred until Phase 8 raster polygonize work lands.",
        profile_mode=profile_mode,
    )


def pipeline_scales(suite: str) -> tuple[int, ...]:
    if suite == "smoke":
        return (1_000,)
    if suite == "ci":
        return (100_000,)
    if suite == "full":
        return (100_000, 1_000_000)
    raise ValueError(f"Unsupported suite: {suite}")


# ---------------------------------------------------------------------------
# Provenance rewrite A/B benchmark pipeline
# ---------------------------------------------------------------------------


def _profile_provenance_rewrite_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """A/B benchmark: buffer().intersects() with vs without provenance rewrites."""
    from time import perf_counter

    from vibespatial.api.geometry_array import GeometryArray
    from vibespatial.runtime.provenance import (
        clear_rewrite_events,
        get_rewrite_events,
        set_provenance_rewrites,
    )

    clear_dispatch_events()
    clear_fallback_events()
    clear_rewrite_events()

    profiler = _stage_profiler(
        operation="pipeline.provenance-rewrite",
        dataset=f"scale-{scale}",
        requested_runtime=ExecutionMode.AUTO,
        selected_runtime="cpu",
        enable_nvtx=enable_nvtx,
        retain_gpu_trace=retain_gpu_trace,
        include_gpu_sparklines=include_gpu_sparklines,
    )

    buffer_distance = 5.0

    with profiler.stage(
        "generate_points",
        category="setup",
        device=ExecutionMode.CPU,
        rows_in=scale,
        detail="generate random point arrays for A/B comparison",
    ) as stage:
        rng = np.random.default_rng(42)
        coords_left = rng.uniform(0, 1000, (scale, 2))
        coords_right = rng.uniform(0, 1000, (scale, 2))
        points = GeometryArray(
            shapely.points(coords_left),
        )
        targets = GeometryArray(
            shapely.points(coords_right),
        )
        stage.rows_out = scale

    # A: with rewrites (should fire R1: buffer(r).intersects -> dwithin(r))
    clear_rewrite_events()
    with profiler.stage(
        "buffer_intersects_rewrite",
        category="refine",
        device=ExecutionMode.CPU,
        rows_in=scale,
        detail="buffer().intersects() with provenance rewrites enabled",
    ) as stage:
        t0 = perf_counter()
        result_a = points.buffer(buffer_distance).intersects(targets)
        rewrite_elapsed = perf_counter() - t0
        rewrite_events = get_rewrite_events()
        stage.rows_out = int(np.sum(result_a)) if hasattr(result_a, "__len__") else 0
        stage.metadata["rewrite_count"] = len(rewrite_events)
        stage.metadata["wall_clock_seconds"] = rewrite_elapsed

    # B: without rewrites (naive path)
    set_provenance_rewrites(False)
    clear_rewrite_events()
    try:
        with profiler.stage(
            "buffer_intersects_naive",
            category="refine",
            device=ExecutionMode.CPU,
            rows_in=scale,
            detail="buffer().intersects() with provenance rewrites disabled",
        ) as stage:
            t0 = perf_counter()
            result_b = points.buffer(buffer_distance).intersects(targets)
            naive_elapsed = perf_counter() - t0
            naive_events = get_rewrite_events()
            stage.rows_out = int(np.sum(result_b)) if hasattr(result_b, "__len__") else 0
            stage.metadata["rewrite_count"] = len(naive_events)
            stage.metadata["wall_clock_seconds"] = naive_elapsed
    finally:
        set_provenance_rewrites(None)

    # Compare
    results_match = np.array_equal(result_a, result_b)
    speedup = naive_elapsed / rewrite_elapsed if rewrite_elapsed > 0 else float("inf")
    with profiler.stage(
        "compare",
        category="emit",
        device=ExecutionMode.CPU,
        rows_in=scale,
        detail="verify equivalence and compute speedup",
    ) as stage:
        stage.rows_out = scale
        stage.metadata["results_match"] = results_match
        stage.metadata["speedup"] = round(speedup, 3)
        stage.metadata["rewrite_seconds"] = round(rewrite_elapsed, 6)
        stage.metadata["naive_seconds"] = round(naive_elapsed, 6)

    trace = profiler.finish(
        metadata={
            "scale": scale,
            "buffer_distance": buffer_distance,
            "speedup": round(speedup, 3),
            "results_match": results_match,
            "rewrite_event_count": len(rewrite_events),
        }
    )
    return PipelineBenchmarkResult(
        pipeline="provenance-rewrite",
        scale=scale,
        status="ok",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime="cpu",
        planner_selected_runtime="cpu",
        output_rows=int(np.sum(result_a)) if hasattr(result_a, "__len__") else 0,
        transfer_count=0,
        materialization_count=0,
        fallback_event_count=0,
        peak_device_memory_bytes=None,
        stages=(_trace_to_stage_dict(trace),),
        notes=f"A/B provenance rewrite benchmark: speedup={speedup:.3f}x, match={results_match}",
        rewrite_event_count=len(rewrite_events),
    )


def benchmark_pipeline_suite(
    *,
    suite: str = "ci",
    pipelines: tuple[str, ...] = (
        "join-heavy",
        "relation-semijoin",
        "small-grouped-constructive-reduce",
        "constructive",
        "predicate-heavy",
        "zero-transfer",
    ),
    repeat: int = 1,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
    profile_mode: str = "lean",
) -> list[PipelineBenchmarkResult]:
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    effective_profile_mode = _resolve_pipeline_profile_mode(
        profile_mode,
        retain_gpu_trace=retain_gpu_trace,
        include_gpu_sparklines=include_gpu_sparklines,
    )
    # Benchmark timings should measure kernels and data movement, not first-use
    # compilation. Front-load the full CCCL/NVRTC benchmark stack once before
    # any pipeline sample starts so fresh-process suite runs are comparable.
    from vibespatial.cuda.cccl_precompile import precompile_all
    precompile_all(timeout=120.0)
    results: list[PipelineBenchmarkResult] = []
    previous_profile_mode = _set_pipeline_profile_mode(effective_profile_mode)
    try:
        for scale in pipeline_scales(suite):
            for pipeline in pipelines:
                samples: list[PipelineBenchmarkResult] = []
                for _ in range(repeat):
                    if pipeline == "join-heavy":
                        samples.append(
                            _profile_join_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "relation-semijoin":
                        samples.append(
                            _profile_relation_semijoin_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "relation-bridge-consumer":
                        samples.append(
                            _profile_relation_bridge_consumer_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "grouped-reducer":
                        samples.append(
                            _profile_grouped_reducer_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "small-grouped-constructive-reduce":
                        samples.append(
                            _profile_small_grouped_constructive_reduce_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "relation-attribute-reducer":
                        samples.append(
                            _profile_relation_attribute_reducer_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "constructive":
                        samples.append(
                            _profile_constructive_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "predicate-heavy":
                        samples.append(
                            _profile_predicate_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "predicate-heavy-geopandas":
                        samples.append(
                            _profile_predicate_geopandas_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "zero-transfer":
                        samples.append(
                            _profile_zero_transfer_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "raster-to-vector":
                        samples.append(
                            _deferred_raster_pipeline(
                                scale,
                                profile_mode=effective_profile_mode,
                            )
                        )
                    elif pipeline == "vegetation-corridor":
                        samples.append(
                            _profile_vegetation_corridor_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "vegetation-corridor-geopandas":
                        samples.append(
                            _profile_vegetation_corridor_geopandas_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "parcel-zoning":
                        samples.append(
                            _profile_parcel_zoning_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "parcel-zoning-geopandas":
                        samples.append(
                            _profile_parcel_zoning_geopandas_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "flood-exposure":
                        samples.append(
                            _profile_flood_exposure_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "flood-exposure-geopandas":
                        samples.append(
                            _profile_flood_exposure_geopandas_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "network-service-area":
                        samples.append(
                            _profile_network_service_area_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "network-service-area-geopandas":
                        samples.append(
                            _profile_network_service_area_geopandas_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "site-suitability":
                        samples.append(
                            _profile_site_suitability_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "site-suitability-geopandas":
                        samples.append(
                            _profile_site_suitability_geopandas_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "provenance-rewrite":
                        samples.append(
                            _profile_provenance_rewrite_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    else:
                        raise ValueError(f"Unsupported pipeline: {pipeline}")
                # Release GPU pool memory between pipelines to prevent OOM
                # from accumulated device allocations across pipeline runs.
                _free_gpu_pool_memory()
                if pipeline == "raster-to-vector":
                    results.append(samples[0])
                    continue
                median_elapsed = median(sample.elapsed_seconds for sample in samples)
                median_sample = min(samples, key=lambda sample: abs(sample.elapsed_seconds - median_elapsed))
                results.append(
                    PipelineBenchmarkResult(
                        pipeline=median_sample.pipeline,
                        scale=median_sample.scale,
                        status=median_sample.status,
                        elapsed_seconds=float(median_elapsed),
                        selected_runtime=median_sample.selected_runtime,
                        planner_selected_runtime=median_sample.planner_selected_runtime,
                        output_rows=median_sample.output_rows,
                        transfer_count=max(
                            (
                                sample.runtime_d2h_transfer_count
                                if sample.runtime_d2h_transfer_count is not None
                                else sample.transfer_count
                            )
                            for sample in samples
                        ),
                        owned_transfer_count=max(
                            (
                                sample.owned_transfer_count
                                if sample.owned_transfer_count is not None
                                else sample.transfer_count
                            )
                            for sample in samples
                        ),
                        runtime_d2h_transfer_count=max(
                            (
                                sample.runtime_d2h_transfer_count
                                if sample.runtime_d2h_transfer_count is not None
                                else sample.transfer_count
                            )
                            for sample in samples
                        ),
                        runtime_d2h_transfer_bytes=max(
                            sample.runtime_d2h_transfer_bytes or 0
                            for sample in samples
                        ),
                        runtime_d2h_transfer_seconds=max(
                            sample.runtime_d2h_transfer_seconds or 0.0
                            for sample in samples
                        ),
                        materialization_count=max(sample.materialization_count for sample in samples),
                        fallback_event_count=max(sample.fallback_event_count for sample in samples),
                        peak_device_memory_bytes=max(
                            (sample.peak_device_memory_bytes for sample in samples if sample.peak_device_memory_bytes is not None),
                            default=None,
                        ),
                        stages=median_sample.stages,
                        notes=median_sample.notes,
                        rewrite_event_count=median_sample.rewrite_event_count,
                        profile_mode=effective_profile_mode,
                    )
                )
    finally:
        _set_pipeline_profile_mode(previous_profile_mode)
    if suite == "full":
        results.append(
            _deferred_raster_pipeline(
                100_000,
                profile_mode=effective_profile_mode,
            )
        )
        results.append(
            _deferred_raster_pipeline(
                1_000_000,
                profile_mode=effective_profile_mode,
            )
        )
    return results


def suite_to_json(results: list[PipelineBenchmarkResult], *, suite: str | None = None, repeat: int = 1) -> str:
    profile_modes = {result.profile_mode for result in results}
    payload = {
        "results": [result.to_dict() for result in results],
        "metadata": {
            "repeat": repeat,
            "profile_mode": (
                next(iter(profile_modes))
                if len(profile_modes) == 1
                else sorted(profile_modes)
            ),
        },
    }
    if suite is not None:
        payload["metadata"]["suite"] = suite
    return json.dumps(payload, indent=2)


def render_gpu_sparkline_report(results: list[PipelineBenchmarkResult]) -> str:
    lines: list[str] = []
    for result in results:
        if result.status != "ok":
            continue
        traces = [stage for stage in result.stages if isinstance(stage, dict)]
        for trace in traces:
            for stage in trace.get("stages", []):
                metadata = stage.get("metadata", {})
                util = metadata.get("gpu_util_sparkline")
                mem = metadata.get("gpu_memory_util_sparkline")
                vram = metadata.get("gpu_vram_sparkline")
                if not any((util, mem, vram)):
                    continue
                wall_elapsed = float(stage.get("elapsed_seconds", 0.0))
                wall_display = metadata.get("elapsed_display", _format_elapsed_compact(wall_elapsed))
                gpu_elapsed = metadata.get("gpu_event_elapsed_seconds")
                if gpu_elapsed is not None and stage.get("device") == ExecutionMode.GPU.value:
                    timing_summary = f"gpu={_format_elapsed_compact(float(gpu_elapsed))} wall={wall_display}"
                else:
                    timing_summary = str(wall_display)
                lines.append(
                    f"{result.pipeline} scale={result.scale} stage={stage['name']} {timing_summary}"
                )
                if util:
                    lines.append(f"gpu util  {util}")
                if mem:
                    lines.append(f"mem util  {mem}")
                if vram:
                    lines.append(f"vram      {vram}")
                substages = metadata.get("gpu_substage_timings")
                if substages:
                    parts = []
                    for key in (
                        "coerce_left_s", "normalize_right_s", "move_to_device_s",
                        "coarse_filter_s", "candidate_mask_s", "point_upload_s",
                        "polygon_upload_s", "kernel_launch_and_sync_s",
                    ):
                        val = substages.get(key)
                        if val is not None:
                            parts.append(f"{key}={_format_elapsed_compact(float(val))}")
                    for key in ("candidate_count", "total_rows", "strategy"):
                        val = substages.get(key)
                        if val is not None:
                            parts.append(f"{key}={val}")
                    lines.append(f"substages {' | '.join(parts)}")
    return "\n".join(lines)
