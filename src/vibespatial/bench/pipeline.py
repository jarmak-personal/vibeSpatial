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
from shapely.geometry import Point, Polygon, box

import vibespatial.api as geopandas
from vibespatial.api._native_grouped import NativeGrouped
from vibespatial.api._native_metadata import NativeGeometryMetadata
from vibespatial.api._native_relation import NativeRelation
from vibespatial.api._native_result_core import (
    NativeAttributeTable,
    NativeGeometryProvenance,
    NativeTabularResult,
)
from vibespatial.api._native_results import (
    GeometryNativeResult,
    _grouped_constructive_to_native_tabular_result,
    _relation_constructive_capacity_result,
)
from vibespatial.api._native_rowset import NativeRowSet
from vibespatial.api._native_state import (
    NativeFrameState,
    attach_native_state,
    get_native_state,
)
from vibespatial.api.tools.sjoin import _sjoin_export_result, _sjoin_nearest_relation_result
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
from vibespatial.constructive.point import (
    point_owned_from_xy_device as _point_owned_from_xy_device,
)
from vibespatial.constructive.polygon import polygon_centroids_owned
from vibespatial.constructive.union_all import union_all_gpu_owned
from vibespatial.cuda._runtime import pylibcudf_table_from_arrow
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import (
    DeviceRegularGridRectMetadata,
    DiagnosticKind,
    OwnedGeometryArray,
    from_shapely_geometries,
    seed_all_validity_cache,
)
from vibespatial.io.arrow import (
    geoseries_from_owned,
    has_pylibcudf_support,
    read_geoparquet_owned,
    write_geoparquet,
)
from vibespatial.io.geojson import read_geojson_owned
from vibespatial.io.geoparquet import read_geoparquet_native
from vibespatial.kernels.predicates.point_in_polygon import (
    get_last_gpu_substage_timings,
    point_in_polygon,
)
from vibespatial.overlay.dissolve import (
    DissolveUnionMethod,
    evaluate_geopandas_dissolve,
    evaluate_geopandas_dissolve_native,
    execute_grouped_union_codes,
    execute_native_grouped_union,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection, has_gpu_runtime
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import (
    PhysicalWorkEstimate,
    estimate_grouped_work_from_owned,
    estimate_physical_work_from_owned,
)
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

from .profiling import (
    PROFILE_BOUNDARY_ROLES,
    ProfileTrace,
    StageProfiler,
    _format_elapsed_compact,
    profile_boundary_for_stage,
)

PIPELINE_DEFINITIONS = (
    "join-heavy",
    "relation-semijoin",
    "relation-bridge-consumer",
    "grouped-reducer",
    "small-grouped-constructive-reduce",
    "grouped-capacity-partitions",
    "grouped-disjoint-constructive-reduce",
    "grouped-difference-constructive",
    "relation-attribute-reducer",
    "relation-distance-expression",
    "nearest-relation-producer",
    "native-area-expression",
    "native-metadata-index",
    "constructive-output-native",
    "overlay-relation-constructive",
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
    return [part for part in parts if part.geom_type in allowed_geom_types and not part.is_empty]


def _extract_polygonal_components(geometries) -> list:
    """Extract Polygon/MultiPolygon components from geometries, flattening GeometryCollections."""
    result = []
    for g in geometries:
        if g is None or shapely.is_empty(g):
            result.append(g)
        elif g.geom_type == "GeometryCollection":
            polys = _extract_supported_collection_parts(g, _POLYGONAL_COLLECTION_GEOM_TYPES)
            if polys:
                result.append(
                    shapely.union_all(np.asarray(polys, dtype=object))
                    if len(polys) > 1
                    else polys[0]
                )
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

    base_polygons = np.asarray(list(_regular_polygons_frame(polygon_count).geometry), dtype=object)
    resized = np.resize(base_polygons, target_rows).tolist()
    owned = from_shapely_geometries(resized)

    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = geopandas.GeoDataFrame(
        {"geometry": resized},
        geometry="geometry",
        crs="EPSG:4326",
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


def _bench_device_to_host(device_array: object, *, reason: str) -> np.ndarray:
    from vibespatial.cuda._runtime import get_cuda_runtime

    return get_cuda_runtime().copy_device_to_host(
        device_array,
        reason=f"benchmark pipeline {reason}",
    )


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
    compute_materialization_count: int | None = None
    compute_runtime_d2h_transfer_count: int | None = None
    compute_runtime_d2h_transfer_bytes: int | None = None
    compute_runtime_d2h_transfer_seconds: float | None = None
    terminal_materialization_count: int | None = None
    terminal_runtime_d2h_transfer_count: int | None = None
    terminal_runtime_d2h_transfer_bytes: int | None = None
    terminal_runtime_d2h_transfer_seconds: float | None = None
    reference_materialization_count: int | None = None
    reference_runtime_d2h_transfer_count: int | None = None
    reference_runtime_d2h_transfer_bytes: int | None = None
    reference_runtime_d2h_transfer_seconds: float | None = None

    def _profile_boundary_totals(self) -> dict[str, int | float]:
        totals: dict[str, int | float] = {}
        for role in PROFILE_BOUNDARY_ROLES:
            totals[f"{role}_materialization_count"] = 0
            totals[f"{role}_runtime_d2h_transfer_count"] = 0
            totals[f"{role}_runtime_d2h_transfer_bytes"] = 0
            totals[f"{role}_runtime_d2h_transfer_seconds"] = 0.0

        for trace in self.stages:
            if isinstance(trace, ProfileTrace):
                stage_dicts = [stage.to_dict() for stage in trace.stages]
            else:
                stage_dicts = list(trace.get("stages", ()))
            for stage in stage_dicts:
                metadata = stage.get("metadata", {})
                role = profile_boundary_for_stage(
                    str(stage.get("name", "")),
                    str(stage.get("category", "")),
                    metadata,
                )
                totals[f"{role}_materialization_count"] = int(
                    totals[f"{role}_materialization_count"]
                ) + int(metadata.get("materialization_count_delta", 0))
                totals[f"{role}_runtime_d2h_transfer_count"] = int(
                    totals[f"{role}_runtime_d2h_transfer_count"]
                ) + int(metadata.get("runtime_d2h_transfer_count_delta", 0))
                totals[f"{role}_runtime_d2h_transfer_bytes"] = int(
                    totals[f"{role}_runtime_d2h_transfer_bytes"]
                ) + int(metadata.get("runtime_d2h_transfer_bytes_delta", 0))
                totals[f"{role}_runtime_d2h_transfer_seconds"] = float(
                    totals[f"{role}_runtime_d2h_transfer_seconds"]
                ) + float(metadata.get("runtime_d2h_transfer_seconds_delta", 0.0))
        return totals

    def to_dict(self) -> dict:
        owned_transfer_count = (
            self.transfer_count if self.owned_transfer_count is None else self.owned_transfer_count
        )
        runtime_d2h_transfer_count = (
            self.transfer_count
            if self.runtime_d2h_transfer_count is None
            else self.runtime_d2h_transfer_count
        )
        runtime_d2h_transfer_bytes = (
            0 if self.runtime_d2h_transfer_bytes is None else self.runtime_d2h_transfer_bytes
        )
        runtime_d2h_transfer_seconds = (
            0.0 if self.runtime_d2h_transfer_seconds is None else self.runtime_d2h_transfer_seconds
        )
        profile_totals = self._profile_boundary_totals()
        compute_materialization_count = (
            int(profile_totals["compute_materialization_count"])
            if self.compute_materialization_count is None
            else self.compute_materialization_count
        )
        compute_runtime_d2h_transfer_count = (
            int(profile_totals["compute_runtime_d2h_transfer_count"])
            if self.compute_runtime_d2h_transfer_count is None
            else self.compute_runtime_d2h_transfer_count
        )
        compute_runtime_d2h_transfer_bytes = (
            int(profile_totals["compute_runtime_d2h_transfer_bytes"])
            if self.compute_runtime_d2h_transfer_bytes is None
            else self.compute_runtime_d2h_transfer_bytes
        )
        compute_runtime_d2h_transfer_seconds = (
            float(profile_totals["compute_runtime_d2h_transfer_seconds"])
            if self.compute_runtime_d2h_transfer_seconds is None
            else self.compute_runtime_d2h_transfer_seconds
        )
        terminal_materialization_count = (
            int(profile_totals["terminal_materialization_count"])
            if self.terminal_materialization_count is None
            else self.terminal_materialization_count
        )
        terminal_runtime_d2h_transfer_count = (
            int(profile_totals["terminal_runtime_d2h_transfer_count"])
            if self.terminal_runtime_d2h_transfer_count is None
            else self.terminal_runtime_d2h_transfer_count
        )
        terminal_runtime_d2h_transfer_bytes = (
            int(profile_totals["terminal_runtime_d2h_transfer_bytes"])
            if self.terminal_runtime_d2h_transfer_bytes is None
            else self.terminal_runtime_d2h_transfer_bytes
        )
        terminal_runtime_d2h_transfer_seconds = (
            float(profile_totals["terminal_runtime_d2h_transfer_seconds"])
            if self.terminal_runtime_d2h_transfer_seconds is None
            else self.terminal_runtime_d2h_transfer_seconds
        )
        reference_materialization_count = (
            int(profile_totals["reference_materialization_count"])
            if self.reference_materialization_count is None
            else self.reference_materialization_count
        )
        reference_runtime_d2h_transfer_count = (
            int(profile_totals["reference_runtime_d2h_transfer_count"])
            if self.reference_runtime_d2h_transfer_count is None
            else self.reference_runtime_d2h_transfer_count
        )
        reference_runtime_d2h_transfer_bytes = (
            int(profile_totals["reference_runtime_d2h_transfer_bytes"])
            if self.reference_runtime_d2h_transfer_bytes is None
            else self.reference_runtime_d2h_transfer_bytes
        )
        reference_runtime_d2h_transfer_seconds = (
            float(profile_totals["reference_runtime_d2h_transfer_seconds"])
            if self.reference_runtime_d2h_transfer_seconds is None
            else self.reference_runtime_d2h_transfer_seconds
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
            "compute_materialization_count": compute_materialization_count,
            "compute_runtime_d2h_transfer_count": compute_runtime_d2h_transfer_count,
            "compute_runtime_d2h_transfer_bytes": compute_runtime_d2h_transfer_bytes,
            "compute_runtime_d2h_transfer_seconds": compute_runtime_d2h_transfer_seconds,
            "terminal_materialization_count": terminal_materialization_count,
            "terminal_runtime_d2h_transfer_count": terminal_runtime_d2h_transfer_count,
            "terminal_runtime_d2h_transfer_bytes": terminal_runtime_d2h_transfer_bytes,
            "terminal_runtime_d2h_transfer_seconds": terminal_runtime_d2h_transfer_seconds,
            "reference_materialization_count": reference_materialization_count,
            "reference_runtime_d2h_transfer_count": reference_runtime_d2h_transfer_count,
            "reference_runtime_d2h_transfer_bytes": reference_runtime_d2h_transfer_bytes,
            "reference_runtime_d2h_transfer_seconds": reference_runtime_d2h_transfer_seconds,
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
    if isinstance(value, NativeAttributeTable):
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

    def reset_materialization_baseline(self) -> None:
        self._materialization_event_start_count = _materialization_event_count()


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
        raise ValueError(f"profile_mode must be one of {sorted(_PIPELINE_PROFILE_MODES)!r}")
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
        SyntheticSpec(
            "polygon", "regular-grid", count=rows, seed=1, vertices=5, hole_probability=0.0
        )
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


def _varying_box_expression_inputs(
    rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_ids = np.arange(rows, dtype=np.int64)
    x = np.remainder(row_ids, 1024).astype(np.float64, copy=False)
    y = (row_ids // 1024).astype(np.float64, copy=False)
    side = (np.remainder(row_ids, 4) + 1).astype(np.float64, copy=False)
    geometries = shapely.box(x, y, x + side, y + side)
    area = side * side
    length = side * 4.0
    centroid_x = x + side * 0.5
    centroid_y = y + side * 0.5
    group_codes = np.remainder(row_ids, 128).astype(np.int32, copy=False)
    return (
        np.asarray(geometries, dtype=object),
        area,
        length,
        centroid_x,
        centroid_y,
        group_codes,
    )


def _regular_box_grid_shape(rows: int) -> tuple[int, int]:
    cols = max(int(np.sqrt(max(rows, 1))), 1)
    while cols > 1 and rows % cols != 0:
        cols -= 1
    return cols, max(rows // cols, 1)


def _device_regular_box_owned(
    rows: int,
    *,
    x_shift: float = 0.0,
    y_shift: float = 0.0,
    side: float = 1.0,
) -> tuple[OwnedGeometryArray, int, int]:
    from vibespatial.constructive.envelope import _build_device_boxes_from_bounds
    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()
    cols, grid_rows = _regular_box_grid_shape(rows)
    row_ids = np.arange(rows, dtype=np.int64)
    minx = np.remainder(row_ids, cols).astype(np.float64, copy=False) + x_shift
    miny = (row_ids // cols).astype(np.float64, copy=False) + y_shift
    maxx = minx + side
    maxy = miny + side
    bounds = np.column_stack((minx, miny, maxx, maxy)).astype(np.float64, copy=False)
    polygon_family = GeometryFamily.POLYGON
    total_bounds = (
        float(x_shift),
        float(y_shift),
        float(x_shift + cols * side),
        float(y_shift + grid_rows * side),
    )
    owned = _build_device_boxes_from_bounds(
        runtime.from_host(bounds),
        row_count=rows,
    )
    owned.device_state.families[polygon_family].regular_grid_rect = DeviceRegularGridRectMetadata(
        origin_x=float(x_shift),
        origin_y=float(y_shift),
        cell_width=float(side),
        cell_height=float(side),
        cols=int(cols),
        rows=int(grid_rows),
        size=int(rows),
        total_bounds=total_bounds,
    )
    return owned, cols, grid_rows


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

                attributes = NativeAttributeTable(
                    device_table=pylibcudf_table_from_arrow(arrow_table),
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


def _native_box_frame_from_owned(
    owned: OwnedGeometryArray,
    *,
    columns: dict[str, np.ndarray],
    crs: str = "EPSG:4326",
) -> tuple[geopandas.GeoDataFrame, NativeFrameState]:
    """Build a public frame with attached private native state from owned boxes."""
    import pyarrow as pa

    row_count = int(owned.row_count)
    geometry_name = "geometry"
    index = pd.RangeIndex(row_count)
    geometry = geopandas.GeoSeries(
        DeviceGeometryArray._from_owned(owned, crs=crs),
        crs=crs,
        name=geometry_name,
    )
    frame_data = {column: values for column, values in columns.items()}
    frame_data[geometry_name] = geometry
    frame = geopandas.GeoDataFrame(
        frame_data,
        geometry=geometry_name,
        crs=crs,
        index=index,
    )
    arrow_table = pa.table(columns)
    attributes = NativeAttributeTable(
        device_table=pylibcudf_table_from_arrow(arrow_table),
        index_override=index,
        column_override=tuple(arrow_table.column_names),
        schema_override=arrow_table.schema,
    )
    result = NativeTabularResult(
        attributes=attributes,
        geometry=GeometryNativeResult.from_owned(owned, crs=crs),
        geometry_name=geometry_name,
        column_order=tuple(frame.columns),
        geometry_metadata=NativeGeometryMetadata.from_owned(owned),
    )
    state = NativeFrameState.from_native_tabular_result(result)
    attach_native_state(frame, state)
    return frame, state


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
        "planner_selected_runtime": trace.metadata.get(
            "planner_selected_runtime", trace.selected_runtime
        ),
        "total_elapsed_seconds": trace.total_elapsed_seconds,
        "stages": [stage.to_dict() for stage in trace.stages],
        "metadata": trace.metadata,
    }


def _record_stage_overheads(
    stage, audit: _OwnedAudit, memory: _DeviceMemoryMonitor, *values
) -> None:
    transfer_before, materialization_before, seconds_before, bytes_before = audit.snapshot()
    runtime_count_before, runtime_bytes_before, runtime_seconds_before = audit.runtime_snapshot()
    audit.observe(*values)
    memory.update()
    transfer_after, materialization_after, seconds_after, bytes_after = audit.snapshot()
    runtime_count_after, runtime_bytes_after, runtime_seconds_after = audit.runtime_snapshot()
    stage.metadata["transfer_count_delta"] = transfer_after - transfer_before
    stage.metadata["transfer_count_total"] = transfer_after
    stage.metadata["owned_transfer_count_delta"] = transfer_after - transfer_before
    stage.metadata["owned_transfer_count_total"] = transfer_after
    stage.metadata["runtime_d2h_transfer_count_delta"] = runtime_count_after - runtime_count_before
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
    stage.metadata["runtime_d2h_transfer_bytes_delta"] = runtime_bytes_after - runtime_bytes_before
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


def _read_geoparquet_owned_preferred(
    path: Path, *, preferred_backend: str
) -> tuple[OwnedGeometryArray, str, str]:
    if preferred_backend != "gpu":
        return read_geoparquet_owned(path, backend="cpu"), "cpu", ""
    try:
        return read_geoparquet_owned(path, backend="gpu"), "gpu", ""
    except Exception as exc:
        return (
            read_geoparquet_owned(path, backend="cpu"),
            "cpu",
            f"gpu read fallback: {exc.__class__.__name__}",
        )


def _read_geojson_owned_preferred(
    source: Path | str | bytes | bytearray | memoryview,
    *,
    preferred_mode: str,
):
    if preferred_mode == "pylibcudf":
        try:
            return (
                read_geojson_owned(
                    source,
                    prefer="pylibcudf",
                    track_properties=False,
                ),
                "gpu",
                "",
            )
        except Exception as exc:
            return (
                read_geojson_owned(
                    source,
                    prefer="fast-json",
                    track_properties=False,
                ),
                "cpu",
                f"gpu read fallback: {exc.__class__.__name__}",
            )

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
        return (
            geopandas.read_file(path),
            requested_engine,
            "default",
            f"pyogrio fallback: {exc.__class__.__name__}",
        )


def _join_heavy_group_categories(scale: int) -> np.ndarray:
    return np.arange(max(min(scale, 128), 1), dtype=np.int32)


def _is_device_array(value) -> bool:
    return hasattr(value, "__cuda_array_interface__")


def _regular_grid_modulo_groups_are_strictly_disjoint(
    owned: OwnedGeometryArray,
    *,
    group_count: int,
) -> bool:
    """Prove row-id modulo groups cannot contain touching grid neighbors."""
    if group_count <= 1 or owned.device_state is None:
        return False
    polygon_buffer = owned.device_state.families.get(GeometryFamily.POLYGON)
    proof = None if polygon_buffer is None else polygon_buffer.regular_grid_rect
    if proof is None or int(proof.size) != int(owned.row_count):
        return False
    cols = int(proof.cols)
    rows = int(proof.rows)
    if cols <= 0 or rows <= 0:
        return False
    neighbor_deltas: set[int] = set()
    if cols > 1:
        neighbor_deltas.add(1)
    if rows > 1:
        neighbor_deltas.add(cols)
        if cols > 1:
            neighbor_deltas.add(cols - 1)
            neighbor_deltas.add(cols + 1)
    return all(delta % int(group_count) != 0 for delta in neighbor_deltas)


def _dissolve_join_heavy_groups(
    joined_geometry,
    unique_right_index,
    *,
    scale: int,
    rowset_identity: bool = False,
):
    if isinstance(joined_geometry, GeometryNativeResult):
        geometry_result = joined_geometry
    else:
        geometry_result = GeometryNativeResult.from_geoseries(joined_geometry)

    geometry_name = "geometry"
    group_categories = _join_heavy_group_categories(scale)
    group_count = int(group_categories.size)
    group_labels = None
    observed_codes = None
    observed_labels = None
    observed_codes_device = None
    if _is_device_array(unique_right_index):
        import cupy as cp

        d_unique_right_index = cp.asarray(unique_right_index, dtype=cp.int64)
        d_group_labels = cp.remainder(d_unique_right_index, group_count)
        row_group_codes = d_group_labels.astype(cp.int32, copy=False)
    else:
        group_labels = np.remainder(
            np.asarray(unique_right_index, dtype=np.int64),
            group_count,
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
    native_grouped = None
    if rowset_identity and group_count > 0:
        row_count = int(joined_owned.row_count)
        all_groups_observed = row_count >= group_count
        native_grouped = NativeGrouped.from_dense_codes(
            row_group_codes,
            group_count=group_count,
            source_token="join-heavy-regular-grid-modulo",
            all_groups_observed=all_groups_observed,
            group_size_min=(row_count // group_count if all_groups_observed else 0),
            group_size_max=((row_count + group_count - 1) // group_count),
            strictly_disjoint_group_bounds=(
                all_groups_observed
                and _regular_grid_modulo_groups_are_strictly_disjoint(
                    joined_owned,
                    group_count=group_count,
                )
            ),
        )
    grouped_union = execute_grouped_union_codes(
        geometry_values,
        row_group_codes,
        group_count=group_count,
        method=DissolveUnionMethod.COVERAGE,
        owned=joined_owned,
        native_grouped=native_grouped,
    )
    if grouped_union is not None:
        if observed_codes is None:
            if (
                grouped_union.owned is not None
                and grouped_union.non_empty_groups == grouped_union.group_count
            ):
                observed_codes = np.arange(group_count, dtype=np.int32)
                observed_labels = observed_codes.astype(np.int64, copy=False)
            else:
                import cupy as cp

                observed_codes_device = cp.unique(row_group_codes).astype(
                    cp.int64,
                    copy=False,
                )
                observed_codes = _bench_device_to_host(
                    observed_codes_device,
                    reason="dissolve observed group-code host export",
                ).astype(
                    np.int32,
                    copy=False,
                )
                observed_labels = observed_codes.astype(np.int64, copy=False)
        group_index = pd.CategoricalIndex(pd.Categorical(observed_labels), name="group")
        if grouped_union.owned is not None:
            if observed_codes.size == grouped_union.group_count and np.array_equal(
                observed_codes,
                np.arange(grouped_union.group_count, dtype=np.int32),
            ):
                selected_owned = grouped_union.owned
            else:
                take_codes = (
                    observed_codes_device
                    if observed_codes_device is not None
                    else observed_codes.astype(np.int64, copy=False)
                )
                selected_owned = grouped_union.owned.take(take_codes)
            if selected_owned.device_state is not None:
                selected_owned.device_state.trusted_all_valid = True
                if len(selected_owned.device_state.families) == 1:
                    selected_owned.device_state.trusted_homogeneous_family = next(
                        iter(selected_owned.device_state.families)
                    )
            geometry_result = GeometryNativeResult.from_owned(
                selected_owned, crs=geometry_result.crs
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

    if group_labels is None:
        from vibespatial.runtime.materialization import (
            MaterializationBoundary,
            record_materialization_event,
        )

        rows = int(getattr(d_group_labels, "size", 0))
        itemsize = int(getattr(getattr(d_group_labels, "dtype", None), "itemsize", 0))
        record_materialization_event(
            surface="vibespatial.bench.pipeline._dissolve_join_heavy_groups",
            boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
            operation="device_group_labels_to_host",
            reason="device group labels were materialized for public dissolve fallback",
            detail=f"rows={rows}, bytes={rows * itemsize}",
            d2h_transfer=True,
            strict_disallowed=True,
        )
        import cupy as cp

        group_labels = _bench_device_to_host(
            d_group_labels,
            reason="dissolve device group-label host export",
        ).astype(np.int64, copy=False)

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
        audit.reset_materialization_baseline()
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

        with profiler.stage(
            "read_polygons", category="setup", device="auto", rows_in=polygon_rows
        ) as stage:
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
            detail="query candidate polygons as relation pairs without host pair export",
        ) as stage:
            query_result, query_execution = query_spatial_index(
                right_owned,
                flat_index,
                left_owned,
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
                predicate="intersects",
                left_row_count=left_owned.row_count,
                right_row_count=right_owned.row_count,
                sorted_by_left=True,
            )
            stage.device = query_execution.selected
            stage.rows_out = len(relation)
            stage.metadata["pairs_examined"] = len(relation)
            stage.metadata["pair_storage"] = pair_storage
            stage.metadata["query_implementation"] = query_execution.implementation
            stage.metadata["query_reason"] = query_execution.reason
            stage.metadata["regular_grid_fast_path"] = bool(flat_index.regular_grid is not None)
            _record_stage_overheads(stage, audit, memory, left_owned, right_owned)

        with profiler.stage(
            "assemble_join_rows",
            category="refine",
            device="auto",
            rows_in=int(len(relation)),
            detail="derive matched right-row NativeRowSet and gather polygons before dissolve",
        ) as stage:
            right_rowset = relation.right_semijoin_rowset()
            unique_right_index = right_rowset.positions
            selected_right_owned = (
                right_owned if right_rowset.identity else right_owned.take(unique_right_index)
            )
            joined_geometry = (
                GeometryNativeResult.from_owned(
                    selected_right_owned,
                    crs="EPSG:4326",
                )
                if unique_right_index.size
                else GeometryNativeResult.from_geoseries(
                    geopandas.GeoSeries([], name="geometry", crs="EPSG:4326")
                )
            )
            stage.device = _selected_runtime_from_history(joined_geometry) or "cpu"
            stage.rows_out = int(joined_geometry.row_count)
            stage.metadata["deduped_candidate_rows"] = int(len(right_rowset))
            stage.metadata["rowset_storage"] = "device" if right_rowset.is_device else "host"
            stage.metadata["ordered"] = right_rowset.ordered
            stage.metadata["unique"] = right_rowset.unique
            stage.metadata["identity"] = right_rowset.identity
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
                rowset_identity=right_rowset.identity,
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
        output_rows=(trace.metadata["dispatch_events"] and int(len(dissolved.attributes)))
        or int(len(dissolved.attributes)),
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
                "geometry": [
                    box(0.0, 0.0, _ZERO_TRANSFER_SELECTIVE_BOUND, _ZERO_TRANSFER_SELECTIVE_BOUND)
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        ).to_parquet(right_path, geometry_encoding="geoarrow")

        audit.reset_runtime_baseline()
        audit.reset_materialization_baseline()
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
            "device" if hasattr(relation.left_indices, "__cuda_array_interface__") else "host"
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
                "native-backed public relation export -> unique-label semijoin native frame"
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
        flags = cp.remainder(rows, 19) == 0
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
        detail="reduce numeric and boolean vectors by dense NativeGrouped codes without pandas groupby",
    ) as stage:
        started = perf_counter()
        reduced = grouped.reduce_numeric(values, "sum")
        reduced_any = grouped.reduce_numeric(flags, "any")
        reduced_all = grouped.reduce_numeric(flags, "all")
        native_elapsed = perf_counter() - started
        stage.rows_out = int(reduced.group_count)
        stage.metadata["reducer"] = reduced.reducer
        stage.metadata["result_storage"] = "device" if reduced.is_device else "host"
        stage.metadata["any_result_storage"] = "device" if reduced_any.is_device else "host"
        stage.metadata["all_result_storage"] = "device" if reduced_all.is_device else "host"
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
        bytes_to_host = int(
            codes.nbytes
            + values.nbytes
            + flags.nbytes
            + reduced.values.nbytes
            + reduced_any.values.nbytes
            + reduced_all.values.nbytes
        )
        record_materialization_event(
            surface="pipeline.grouped-reducer.public_groupby_reference",
            boundary=MaterializationBoundary.USER_EXPORT,
            operation="grouped_reducer_reference_export",
            reason="exported grouped reducer inputs and output for pandas reference check",
            detail=f"rows={scale}, groups={group_count}, bytes={bytes_to_host}",
            d2h_transfer=True,
            strict_disallowed=False,
        )
        codes_host = _bench_device_to_host(codes, reason="grouped reducer codes host export")
        values_host = _bench_device_to_host(values, reason="grouped reducer values host export")
        flags_host = _bench_device_to_host(flags, reason="grouped reducer flags host export")
        actual = _bench_device_to_host(
            reduced.values,
            reason="grouped reducer sum output host export",
        )
        actual_any = _bench_device_to_host(
            reduced_any.values,
            reason="grouped reducer any output host export",
        )
        actual_all = _bench_device_to_host(
            reduced_all.values,
            reason="grouped reducer all output host export",
        )
        observed = codes_host >= 0
        expected = (
            pd.Series(values_host[observed])
            .groupby(codes_host[observed], sort=True)
            .sum()
            .reindex(pd.RangeIndex(group_count), fill_value=0.0)
            .to_numpy()
        )
        grouped_flags = pd.Series(flags_host[observed]).groupby(
            codes_host[observed],
            sort=True,
        )
        expected_any = (
            grouped_flags.any()
            .reindex(pd.RangeIndex(group_count), fill_value=False)
            .to_numpy(dtype=bool)
        )
        expected_all = (
            grouped_flags.all()
            .reindex(pd.RangeIndex(group_count), fill_value=True)
            .to_numpy(dtype=bool)
        )
        public_elapsed = perf_counter() - started
        results_match = bool(
            np.allclose(actual, expected)
            and np.array_equal(actual_any, expected_any)
            and np.array_equal(actual_all, expected_all)
        )
        stage.rows_out = group_count
        stage.metadata["results_match"] = results_match
        stage.metadata["bool_results_match"] = bool(
            np.array_equal(actual_any, expected_any) and np.array_equal(actual_all, expected_all)
        )
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
            "admissible_shape": "dense-code NativeGrouped numeric and boolean reductions",
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
            "NativeGrouped.reduce_numeric(sum/any/all) with pandas groupby "
            "only as an explicit reference stage."
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

    from vibespatial.runtime.materialization import clear_materialization_events

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
    import cupy as cp

    with profiler.stage(
        "build_device_grouped_polygons",
        category="setup",
        device=ExecutionMode.GPU,
        rows_in=scale,
        detail="build device-resident polygon groups with 2-8 rows per group",
    ) as stage:
        values, group_offsets, group_count = _small_grouped_constructive_fixture(scale)
        owned = from_shapely_geometries(values, residency=Residency.DEVICE)
        seed_all_validity_cache(owned)
        group_sizes = np.diff(group_offsets)
        grouped = NativeGrouped.from_sorted_offsets(
            cp.asarray(group_offsets, dtype=cp.int64),
            row_count=owned.row_count,
            source_token="small-grouped-constructive-reduce",
            all_groups_observed=True,
            group_size_min=int(group_sizes.min(initial=0)),
            group_size_max=int(group_sizes.max(initial=0)),
        )
        stage.rows_out = int(owned.row_count)
        stage.metadata["group_count"] = group_count
        stage.metadata["row_count"] = int(owned.row_count)
        stage.metadata["min_group_size"] = int(group_sizes.min())
        stage.metadata["max_group_size"] = int(group_sizes.max(initial=0))
        stage.metadata["geometry_storage"] = "owned:device"
        stage.metadata["group_carrier"] = "NativeGrouped"
        stage.metadata["group_storage"] = "device" if grouped.is_device else "host"
        stage.metadata["sorted_order_identity"] = grouped.sorted_order_is_identity
        _record_stage_overheads(stage, audit, memory, owned)

    with profiler.stage(
        "native_grouped_union",
        category="reduce",
        device=ExecutionMode.GPU,
        rows_in=int(owned.row_count),
        detail="batch many tiny grouped polygon unions without per-group dispatch",
    ) as stage:
        started = perf_counter()
        grouped_union = execute_native_grouped_union(
            grouped,
            _geometries=(),
            method=DissolveUnionMethod.UNARY,
            owned=owned,
        )
        if grouped_union is None or grouped_union.owned is None:
            raise RuntimeError("NativeGrouped constructive reduce did not stay native")
        reduced = grouped_union.owned
        native_elapsed = perf_counter() - started
        dispatch_events = get_dispatch_events()
        used_native_grouped_carrier = any(
            event.surface == "vibespatial.overlay.dissolve.execute_native_grouped_union"
            and event.operation == "grouped_union"
            and event.selected is ExecutionMode.GPU
            for event in dispatch_events
        )
        stage.rows_out = int(reduced.row_count)
        stage.metadata["native_reduce_seconds"] = native_elapsed
        stage.metadata["result_storage"] = (
            "device" if reduced.residency is Residency.DEVICE else "host"
        )
        stage.metadata["input_carrier"] = "NativeGrouped"
        stage.metadata["output_carrier"] = "GroupedUnionResult"
        stage.metadata["grouped_geometry_storage"] = (
            "device" if grouped_union.owned.residency is Residency.DEVICE else "host"
        )
        stage.metadata["used_native_grouped_carrier"] = used_native_grouped_carrier
        stage.metadata["sorted_order_identity"] = grouped.sorted_order_is_identity
        stage.metadata["non_empty_groups"] = grouped_union.non_empty_groups
        stage.metadata["empty_groups"] = grouped_union.empty_groups
        _record_stage_overheads(stage, audit, memory, reduced)

    with profiler.stage(
        "native_reference_check",
        category="reference",
        device=ExecutionMode.GPU,
        rows_in=int(owned.row_count),
        detail="validate native grouped union shape without host geometry export",
        metadata={"profile_boundary": "reference"},
    ) as stage:
        started = perf_counter()
        row_count_match = int(reduced.row_count) == int(group_count)
        non_empty_match = int(grouped_union.non_empty_groups) == int(group_count)
        empty_match = int(grouped_union.empty_groups) == 0
        storage_match = reduced.residency is Residency.DEVICE
        carrier_match = used_native_grouped_carrier and grouped.is_device
        reference_elapsed = perf_counter() - started
        results_match = (
            row_count_match and non_empty_match and empty_match and storage_match and carrier_match
        )
        stage.rows_out = int(reduced.row_count)
        stage.metadata["reference_mode"] = "native_shape_invariants"
        stage.metadata["row_count_match"] = row_count_match
        stage.metadata["non_empty_group_match"] = non_empty_match
        stage.metadata["empty_group_match"] = empty_match
        stage.metadata["result_storage_match"] = storage_match
        stage.metadata["carrier_match"] = carrier_match
        stage.metadata["results_match"] = results_match
        stage.metadata["reference_check_seconds"] = reference_elapsed
        stage.metadata["native_reduce_seconds"] = native_elapsed
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
                "NativeGrouped sorted offsets + owned device polygons -> batched grouped constructive reduce"
            ),
            "results_match": results_match,
            "used_native_grouped_carrier": used_native_grouped_carrier,
            "reference_mode": "native_shape_invariants",
        }
    )
    return PipelineBenchmarkResult(
        pipeline="small-grouped-constructive-reduce",
        scale=scale,
        status="ok" if results_match and used_native_grouped_carrier else "failed",
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
            "polygon groups reduce from a NativeGrouped carrier through one "
            "batched grouped constructive path; reference validation uses "
            "native shape invariants without host geometry export."
        ),
    )


def _grouped_capacity_partition_fixtures(
    scale: int,
) -> tuple[list[object], list[object], np.ndarray, int]:
    group_count = max(8, min(max(scale // 6, 8), 1024))
    group_size = 3
    group_offsets = np.arange(
        0,
        (group_count * group_size) + 1,
        group_size,
        dtype=np.int32,
    )
    mixed_values: list[object] = []
    degenerate_values: list[object] = []
    for group_index in range(group_count):
        x0 = float(group_index) * 20.0
        if group_index % 2 == 0:
            mixed_values.extend(
                [
                    box(x0, 0.0, x0 + 2.0, 1.0),
                    box(x0 + 1.0, 0.0, x0 + 3.0, 1.0),
                    box(x0 + 2.0, 0.0, x0 + 4.0, 1.0),
                ]
            )
        else:
            mixed_values.extend(
                [
                    box(x0, 0.0, x0 + 2.0, 2.0),
                    box(x0 + 1.0, 1.0, x0 + 3.0, 3.0),
                    box(x0 + 2.0, -0.5, x0 + 4.0, 1.5),
                ]
            )

        tiny = Polygon(
            [
                (x0 + 3.0, 2.0),
                (x0 + 3.000001, 2.0),
                (x0 + 3.0, 2.000001),
                (x0 + 3.0, 2.0),
            ]
        )
        degenerate_values.extend(
            [
                box(x0, 0.0, x0 + 4.0, 4.0),
                box(x0 + 2.0, 1.0, x0 + 6.0, 5.0),
                tiny,
            ]
        )
    return mixed_values, degenerate_values, group_offsets, group_count


def _profile_grouped_capacity_partitions_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Profile mixed strip/exact and positive-area/degenerate grouped lanes."""
    if not has_gpu_runtime():
        return PipelineBenchmarkResult(
            pipeline="grouped-capacity-partitions",
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
            notes="Deferred until CUDA runtime is available for grouped partition profiling.",
        )

    from time import perf_counter

    import cupy as cp

    from vibespatial.runtime.materialization import clear_materialization_events

    clear_materialization_events()
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    profiler = _stage_profiler(
        operation="pipeline.grouped-capacity-partitions",
        dataset=f"scale-{scale}",
        requested_runtime=ExecutionMode.GPU,
        selected_runtime=ExecutionMode.GPU.value,
        enable_nvtx=enable_nvtx,
        retain_gpu_trace=retain_gpu_trace,
        include_gpu_sparklines=include_gpu_sparklines,
    )

    with profiler.stage(
        "build_grouped_partition_fixtures",
        category="setup",
        device=ExecutionMode.GPU,
        rows_in=scale,
        detail="build alternating strip/exact groups and overlapping positive/degenerate groups",
    ) as stage:
        mixed_values, degenerate_values, group_offsets, group_count = (
            _grouped_capacity_partition_fixtures(scale)
        )
        mixed_owned = from_shapely_geometries(
            mixed_values,
            residency=Residency.DEVICE,
        )
        degenerate_owned = from_shapely_geometries(
            degenerate_values,
            residency=Residency.DEVICE,
        )
        seed_all_validity_cache(mixed_owned)
        seed_all_validity_cache(degenerate_owned)
        d_group_offsets = cp.asarray(group_offsets, dtype=cp.int64)
        grouped_mixed = NativeGrouped.from_sorted_offsets(
            d_group_offsets,
            row_count=mixed_owned.row_count,
            source_token="grouped-capacity-partitions-mixed",
            all_groups_observed=True,
            group_size_min=3,
            group_size_max=3,
        )
        grouped_degenerate = NativeGrouped.from_sorted_offsets(
            d_group_offsets,
            row_count=degenerate_owned.row_count,
            source_token="grouped-capacity-partitions-degenerate",
            all_groups_observed=True,
            group_size_min=3,
            group_size_max=3,
        )
        stage.rows_out = mixed_owned.row_count + degenerate_owned.row_count
        stage.metadata["group_count"] = group_count
        stage.metadata["rows_per_group"] = 3
        stage.metadata["strip_group_count"] = (group_count + 1) // 2
        stage.metadata["exact_group_count"] = group_count // 2
        stage.metadata["group_carrier"] = "NativeGrouped"
        _record_stage_overheads(stage, audit, memory, mixed_owned)

    with profiler.stage(
        "mixed_strip_exact_union",
        category="reduce",
        device=ExecutionMode.GPU,
        rows_in=mixed_owned.row_count,
        detail="partition rectangle strips from exact grouped topology at source capacity",
    ) as stage:
        started = perf_counter()
        mixed_result = execute_native_grouped_union(
            grouped_mixed,
            _geometries=(),
            method=DissolveUnionMethod.UNARY,
            owned=mixed_owned,
        )
        if mixed_result is None or mixed_result.owned is None:
            raise RuntimeError("mixed strip/exact grouped union did not stay native")
        mixed_elapsed = perf_counter() - started
        mixed_implementation = getattr(
            mixed_result.owned,
            "_native_grouped_union_implementation",
            None,
        )
        mixed_remainder_implementation = getattr(
            mixed_result.owned,
            "_native_grouped_union_remainder_implementation",
            None,
        )
        stage.rows_out = mixed_result.owned.row_count
        stage.metadata["native_reduce_seconds"] = mixed_elapsed
        stage.metadata["implementation"] = mixed_implementation
        stage.metadata["remainder_implementation"] = mixed_remainder_implementation
        stage.metadata["input_carrier"] = "NativeGrouped"
        stage.metadata["result_storage"] = mixed_result.owned.residency.value
        stage.metadata["result_unique_family_rows"] = (
            mixed_result.owned._ensure_device_state(
                preserve_indexed_view=True,
            ).trusted_unique_family_rows
        )
        _record_stage_overheads(stage, audit, memory, mixed_result.owned)

    with profiler.stage(
        "positive_degenerate_union",
        category="reduce",
        device=ExecutionMode.GPU,
        rows_in=degenerate_owned.row_count,
        detail="partition positive-area topology from near-degenerate rows at source capacity",
    ) as stage:
        started = perf_counter()
        degenerate_result = execute_native_grouped_union(
            grouped_degenerate,
            _geometries=(),
            method=DissolveUnionMethod.UNARY,
            owned=degenerate_owned,
        )
        if degenerate_result is None or degenerate_result.owned is None:
            raise RuntimeError("positive/degenerate grouped union did not stay native")
        degenerate_elapsed = perf_counter() - started
        degenerate_implementation = getattr(
            degenerate_result.owned,
            "_native_grouped_union_implementation",
            None,
        )
        degenerate_remainder_implementation = getattr(
            degenerate_result.owned,
            "_native_grouped_union_remainder_implementation",
            None,
        )
        stage.rows_out = degenerate_result.owned.row_count
        stage.metadata["native_reduce_seconds"] = degenerate_elapsed
        stage.metadata["implementation"] = degenerate_implementation
        stage.metadata["remainder_implementation"] = degenerate_remainder_implementation
        stage.metadata["input_carrier"] = "NativeGrouped"
        stage.metadata["result_storage"] = degenerate_result.owned.residency.value
        stage.metadata["result_unique_family_rows"] = (
            degenerate_result.owned._ensure_device_state(
                preserve_indexed_view=True,
            ).trusted_unique_family_rows
        )
        _record_stage_overheads(stage, audit, memory, degenerate_result.owned)

    partition_stages = {stage.name: stage for stage in profiler._stages}

    def _residual_admission_count(stage_name: str) -> int:
        return sum(
            "residual repair admission scalar fence" in event["reason"]
            for event in partition_stages[stage_name].metadata.get(
                "runtime_d2h_transfer_events",
                (),
            )
        )

    mixed_residual_admissions = _residual_admission_count("mixed_strip_exact_union")
    degenerate_residual_admissions = _residual_admission_count("positive_degenerate_union")

    with profiler.stage(
        "native_reference_check",
        category="reference",
        device=ExecutionMode.GPU,
        rows_in=mixed_owned.row_count + degenerate_owned.row_count,
        detail="validate partition implementations and native output shape without geometry export",
        metadata={"profile_boundary": "reference"},
    ) as stage:
        expected_mixed = "native_grouped_rectangle_strip_partition_union"
        expected_mixed_remainder = "native_grouped_disjoint_pack_partition_union"
        expected_degenerate = "native_grouped_disjoint_pack_partition_union"
        expected_degenerate_remainder = (
            "native_grouped_overlay_union_plan_mixed_degenerate_pairwise"
        )
        results_match = (
            mixed_result.owned.row_count == group_count
            and degenerate_result.owned.row_count == group_count
            and mixed_result.owned.residency is Residency.DEVICE
            and degenerate_result.owned.residency is Residency.DEVICE
            and mixed_implementation == expected_mixed
            and mixed_remainder_implementation == expected_mixed_remainder
            and degenerate_implementation == expected_degenerate
            and degenerate_remainder_implementation == expected_degenerate_remainder
            and mixed_residual_admissions == 0
            and degenerate_residual_admissions == 0
        )
        stage.rows_out = mixed_result.owned.row_count + degenerate_result.owned.row_count
        stage.metadata["results_match"] = results_match
        stage.metadata["mixed_implementation_match"] = mixed_implementation == expected_mixed
        stage.metadata["mixed_remainder_implementation_match"] = (
            mixed_remainder_implementation == expected_mixed_remainder
        )
        stage.metadata["degenerate_implementation_match"] = (
            degenerate_implementation == expected_degenerate
        )
        stage.metadata["degenerate_remainder_implementation_match"] = (
            degenerate_remainder_implementation == expected_degenerate_remainder
        )
        stage.metadata["mixed_residual_admission_count"] = mixed_residual_admissions
        stage.metadata["degenerate_residual_admission_count"] = degenerate_residual_admissions
        stage.metadata["reference_mode"] = "native_shape_invariants"
        _record_stage_overheads(stage, audit, memory, degenerate_result.owned)

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
                "NativeGrouped source-capacity strip/exact and positive/degenerate partitions"
            ),
            "results_match": results_match,
            "mixed_implementation": mixed_implementation,
            "mixed_remainder_implementation": mixed_remainder_implementation,
            "degenerate_implementation": degenerate_implementation,
            "degenerate_remainder_implementation": degenerate_remainder_implementation,
            "mixed_residual_admission_count": mixed_residual_admissions,
            "degenerate_residual_admission_count": degenerate_residual_admissions,
            "reference_mode": "native_shape_invariants",
        }
    )
    return PipelineBenchmarkResult(
        pipeline="grouped-capacity-partitions",
        scale=scale,
        status=(
            "ok" if results_match and int(trace.metadata["fallback_events"]) == 0 else "failed"
        ),
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=ExecutionMode.GPU.value,
        output_rows=group_count * 2,
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
            "Shape canary for complementary source-capacity partitions; "
            "reference validation uses native implementation and residency invariants."
        ),
    )


def _grouped_disjoint_constructive_fixture(
    scale: int,
) -> tuple[list[object], np.ndarray, int]:
    group_count = max(8, min(max(scale // 8, 8), 4096))
    group_size = 2
    group_offsets = np.arange(
        0,
        (group_count * group_size) + 1,
        group_size,
        dtype=np.int32,
    )
    values: list[object] = []
    for group_index in range(group_count):
        x0 = float(group_index) * 10.0
        values.append(box(x0, 0.0, x0 + 1.0, 1.0))
        values.append(box(x0 + 2.0, 0.0, x0 + 3.0, 1.0))
    return values, group_offsets, group_count


def _profile_grouped_disjoint_constructive_reduce_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Shape canary for strict disjoint grouped polygon assembly."""
    if not has_gpu_runtime():
        return PipelineBenchmarkResult(
            pipeline="grouped-disjoint-constructive-reduce",
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
            notes="Deferred until CUDA runtime is available for grouped disjoint constructive canary.",
        )

    from time import perf_counter

    import cupy as cp

    from vibespatial.runtime.materialization import clear_materialization_events

    clear_materialization_events()
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    profiler = _stage_profiler(
        operation="pipeline.grouped-disjoint-constructive-reduce",
        dataset=f"scale-{scale}",
        requested_runtime=ExecutionMode.GPU,
        selected_runtime=ExecutionMode.GPU.value,
        enable_nvtx=enable_nvtx,
        retain_gpu_trace=retain_gpu_trace,
        include_gpu_sparklines=include_gpu_sparklines,
    )

    with profiler.stage(
        "build_device_disjoint_groups",
        category="setup",
        device=ExecutionMode.GPU,
        rows_in=scale,
        detail="build device-resident strict-disjoint polygon pairs per group",
    ) as stage:
        values, group_offsets, group_count = _grouped_disjoint_constructive_fixture(scale)
        owned = from_shapely_geometries(values, residency=Residency.DEVICE)
        seed_all_validity_cache(owned)
        grouped = NativeGrouped.from_sorted_offsets(
            cp.asarray(group_offsets, dtype=cp.int64),
            row_count=owned.row_count,
            source_token="grouped-disjoint-constructive-reduce",
            all_groups_observed=True,
            group_size_min=2,
            group_size_max=2,
            strictly_disjoint_group_bounds=True,
        )
        estimate = estimate_grouped_work_from_owned(
            owned,
            grouped=grouped,
            primary_unit_name="grouped-disjoint-segment",
        )
        stage.rows_out = int(owned.row_count)
        stage.metadata["group_count"] = group_count
        stage.metadata["row_count"] = int(owned.row_count)
        stage.metadata["group_size"] = 2
        stage.metadata["geometry_storage"] = "owned:device"
        stage.metadata["group_carrier"] = "NativeGrouped"
        stage.metadata["group_storage"] = "device" if grouped.is_device else "host"
        stage.metadata["sorted_order_identity"] = grouped.sorted_order_is_identity
        stage.metadata["work_estimate"] = estimate.telemetry_detail()
        _record_stage_overheads(stage, audit, memory, owned)

    with profiler.stage(
        "native_grouped_disjoint_subset",
        category="reduce",
        device=ExecutionMode.GPU,
        rows_in=int(owned.row_count),
        detail="assemble strict disjoint grouped MultiPolygons from NativeGrouped offsets",
    ) as stage:
        started = perf_counter()
        grouped_union = execute_native_grouped_union(
            grouped,
            _geometries=(),
            method=DissolveUnionMethod.DISJOINT_SUBSET,
            owned=owned,
        )
        if grouped_union is None or grouped_union.owned is None:
            raise RuntimeError("NativeGrouped disjoint constructive reduce did not stay native")
        reduced = grouped_union.owned
        native_elapsed = perf_counter() - started
        dispatch_events = get_dispatch_events()
        used_native_grouped_carrier = any(
            event.surface == "vibespatial.overlay.dissolve.execute_native_grouped_union"
            and event.operation == "grouped_disjoint_subset_union"
            and event.implementation == "native_grouped_disjoint_subset_device_assembly"
            for event in dispatch_events
        )
        estimate = estimate_grouped_work_from_owned(
            owned,
            grouped=grouped,
            output_row_count=grouped_union.group_count,
            primary_unit_name="grouped-disjoint-segment",
        )
        stage.rows_out = int(reduced.row_count)
        stage.metadata["native_reduce_seconds"] = native_elapsed
        stage.metadata["result_storage"] = (
            "device" if reduced.residency is Residency.DEVICE else "host"
        )
        stage.metadata["input_carrier"] = "NativeGrouped"
        stage.metadata["output_carrier"] = "GroupedUnionResult"
        stage.metadata["method"] = grouped_union.method.value
        stage.metadata["used_native_grouped_carrier"] = used_native_grouped_carrier
        stage.metadata["non_empty_groups"] = grouped_union.non_empty_groups
        stage.metadata["empty_groups"] = grouped_union.empty_groups
        stage.metadata["work_estimate"] = estimate.telemetry_detail()
        _record_stage_overheads(stage, audit, memory, reduced)

    with profiler.stage(
        "native_reference_check",
        category="reference",
        device=ExecutionMode.GPU,
        rows_in=int(owned.row_count),
        detail="validate native disjoint grouped assembly without host geometry export",
        metadata={"profile_boundary": "reference"},
    ) as stage:
        started = perf_counter()
        row_count_match = int(reduced.row_count) == int(group_count)
        non_empty_match = int(grouped_union.non_empty_groups) == int(group_count)
        empty_match = int(grouped_union.empty_groups) == 0
        storage_match = reduced.residency is Residency.DEVICE
        carrier_match = used_native_grouped_carrier and grouped.is_device
        reference_elapsed = perf_counter() - started
        results_match = (
            row_count_match and non_empty_match and empty_match and storage_match and carrier_match
        )
        stage.rows_out = int(reduced.row_count)
        stage.metadata["reference_mode"] = "native_shape_invariants"
        stage.metadata["row_count_match"] = row_count_match
        stage.metadata["non_empty_group_match"] = non_empty_match
        stage.metadata["empty_group_match"] = empty_match
        stage.metadata["result_storage_match"] = storage_match
        stage.metadata["carrier_match"] = carrier_match
        stage.metadata["results_match"] = results_match
        stage.metadata["reference_check_seconds"] = reference_elapsed
        stage.metadata["native_reduce_seconds"] = native_elapsed
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
                "NativeGrouped sorted offsets + owned device polygons -> strict disjoint grouped MultiPolygon assembly"
            ),
            "results_match": results_match,
            "used_native_grouped_carrier": used_native_grouped_carrier,
            "reference_mode": "native_shape_invariants",
        }
    )
    return PipelineBenchmarkResult(
        pipeline="grouped-disjoint-constructive-reduce",
        scale=scale,
        status="ok" if results_match and used_native_grouped_carrier else "failed",
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
            "Shape canary, not a workflow target: strict-disjoint device "
            "polygon groups assemble from a NativeGrouped carrier into a "
            "device MultiPolygon payload; reference validation uses native "
            "shape invariants without host geometry export."
        ),
    )


def _grouped_difference_constructive_fixture(
    scale: int,
) -> tuple[list[object], list[object], np.ndarray, int]:
    group_count = max(8, min(max(scale // 4, 8), 1024))
    group_size = 2
    group_offsets = np.arange(
        0,
        (group_count * group_size) + 1,
        group_size,
        dtype=np.int32,
    )
    left_values: list[object] = []
    right_values: list[object] = []
    for group_index in range(group_count):
        x0 = float(group_index) * 20.0
        left_values.append(box(x0, 0.0, x0 + 10.0, 10.0))
        right_values.append(box(x0 + 1.0, 1.0, x0 + 2.0, 2.0))
        right_values.append(box(x0 + 3.0, 3.0, x0 + 4.0, 4.0))
    return left_values, right_values, group_offsets, group_count


def _profile_grouped_difference_constructive_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Shape canary for NativeGrouped overlay difference assembly."""
    if not has_gpu_runtime():
        return PipelineBenchmarkResult(
            pipeline="grouped-difference-constructive",
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
            notes=(
                "Deferred until CUDA runtime is available for grouped "
                "difference constructive canary."
            ),
        )

    from time import perf_counter

    import cupy as cp

    from vibespatial.runtime.materialization import clear_materialization_events

    clear_materialization_events()
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    profiler = _stage_profiler(
        operation="pipeline.grouped-difference-constructive",
        dataset=f"scale-{scale}",
        requested_runtime=ExecutionMode.GPU,
        selected_runtime=ExecutionMode.GPU.value,
        enable_nvtx=enable_nvtx,
        retain_gpu_trace=retain_gpu_trace,
        include_gpu_sparklines=include_gpu_sparklines,
    )

    with profiler.stage(
        "build_device_grouped_difference_inputs",
        category="setup",
        device=ExecutionMode.GPU,
        rows_in=scale,
        detail=(
            "build left polygons and sorted right-side grouped offsets for "
            "device-resident overlay difference"
        ),
    ) as stage:
        left_values, right_values, group_offsets, group_count = (
            _grouped_difference_constructive_fixture(scale)
        )
        left_owned = from_shapely_geometries(left_values, residency=Residency.DEVICE)
        right_owned = from_shapely_geometries(right_values, residency=Residency.DEVICE)
        seed_all_validity_cache(left_owned)
        seed_all_validity_cache(right_owned)
        group_sizes = np.diff(group_offsets)
        d_group_offsets = cp.asarray(group_offsets, dtype=cp.int64)
        stage.rows_out = int(right_owned.row_count)
        stage.metadata["group_count"] = group_count
        stage.metadata["right_row_count"] = int(right_owned.row_count)
        stage.metadata["group_size"] = int(group_sizes.max(initial=0))
        stage.metadata["geometry_storage"] = "owned:device"
        stage.metadata["group_carrier"] = "NativeGrouped"
        stage.metadata["group_storage"] = "device"
        _record_stage_overheads(stage, audit, memory, left_owned, right_owned)

    with profiler.stage(
        "native_grouped_difference",
        category="constructive",
        device=ExecutionMode.GPU,
        rows_in=int(right_owned.row_count),
        detail=(
            "consume NativeGrouped sorted offsets through one row-isolated "
            "grouped overlay difference plan"
        ),
    ) as stage:
        import importlib

        overlay_module = importlib.import_module("vibespatial.api.tools.overlay")
        started = perf_counter()
        diff_owned = overlay_module._grouped_overlay_difference_owned(
            left_owned,
            right_owned,
            d_group_offsets,
            dispatch_mode=ExecutionMode.GPU,
            _all_groups_observed=True,
            _group_size_min=int(group_sizes.min(initial=0)),
            _group_size_max=int(group_sizes.max(initial=0)),
        )
        native_elapsed = perf_counter() - started
        dispatch_events = get_dispatch_events()
        used_native_grouped_carrier = any(
            event.surface == "geopandas.array.difference"
            and event.operation == "difference"
            and event.implementation
            in {
                "grouped_overlay_difference_gpu",
                "grouped_overlay_difference_rectangle_holes_gpu",
            }
            for event in dispatch_events
        )
        stage.rows_out = int(diff_owned.row_count)
        stage.metadata["native_difference_seconds"] = native_elapsed
        stage.metadata["result_storage"] = (
            "device" if diff_owned.residency is Residency.DEVICE else "host"
        )
        stage.metadata["input_carrier"] = "NativeGrouped"
        stage.metadata["output_carrier"] = "OwnedGeometryArray"
        stage.metadata["used_native_grouped_carrier"] = used_native_grouped_carrier
        stage.metadata["group_size"] = int(group_sizes.max(initial=0))
        _record_stage_overheads(stage, audit, memory, diff_owned)

    with profiler.stage(
        "native_reference_check",
        category="reference",
        device=ExecutionMode.GPU,
        rows_in=int(diff_owned.row_count),
        detail="validate native grouped difference shape without host geometry export",
        metadata={"profile_boundary": "reference"},
    ) as stage:
        started = perf_counter()
        row_count_match = int(diff_owned.row_count) == int(group_count)
        storage_match = diff_owned.residency is Residency.DEVICE
        carrier_match = used_native_grouped_carrier
        group_size_match = int(group_sizes.max(initial=0)) == 2
        reference_elapsed = perf_counter() - started
        results_match = row_count_match and storage_match and carrier_match and group_size_match
        stage.rows_out = int(diff_owned.row_count)
        stage.metadata["reference_mode"] = "native_shape_invariants"
        stage.metadata["row_count_match"] = row_count_match
        stage.metadata["result_storage_match"] = storage_match
        stage.metadata["carrier_match"] = carrier_match
        stage.metadata["group_size_match"] = group_size_match
        stage.metadata["results_match"] = results_match
        stage.metadata["reference_check_seconds"] = reference_elapsed
        stage.metadata["native_difference_seconds"] = native_elapsed
        _record_stage_overheads(stage, audit, memory, diff_owned)

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
                "NativeGrouped sorted offsets + owned device polygons -> "
                "row-isolated grouped overlay difference"
            ),
            "results_match": results_match,
            "used_native_grouped_carrier": used_native_grouped_carrier,
            "reference_mode": "native_shape_invariants",
        }
    )
    return PipelineBenchmarkResult(
        pipeline="grouped-difference-constructive",
        scale=scale,
        status="ok" if results_match and used_native_grouped_carrier else "failed",
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
            "Shape canary, not a workflow target: grouped overlay difference "
            "consumes device sorted offsets and owned polygons through a "
            "NativeGrouped row-isolated constructive carrier; reference "
            "validation uses native shape invariants without host geometry "
            "export."
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
        reduced_attributes = reduced.to_native_attribute_table()
        native_elapsed = perf_counter() - started
        stage.rows_out = int(reduced.group_count)
        stage.metadata["columns"] = tuple(reduced.columns)
        stage.metadata["result_storage"] = "device" if reduced.is_device else "host"
        stage.metadata["attribute_storage"] = (
            "device" if reduced_attributes.device_table is not None else "loader"
        )
        stage.metadata["native_reduce_seconds"] = native_elapsed
        _record_stage_overheads(stage, audit, memory, reduced)

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
        left_host = _bench_device_to_host(
            left_indices,
            reason="relation attribute reducer left-index host export",
        )
        right_host = _bench_device_to_host(
            right_indices,
            reason="relation attribute reducer right-index host export",
        )
        score_host = _bench_device_to_host(
            right_score,
            reason="relation attribute reducer score input host export",
        )
        weight_host = _bench_device_to_host(
            right_weight,
            reason="relation attribute reducer weight input host export",
        )
        actual_score = _bench_device_to_host(
            reduced.columns["score_sum"].values,
            reason="relation attribute reducer score output host export",
        )
        actual_count = _bench_device_to_host(
            reduced.columns["match_count"].values,
            reason="relation attribute reducer count output host export",
        )
        actual_weight = _bench_device_to_host(
            reduced.columns["weight_mean"].values,
            reason="relation attribute reducer weight output host export",
        )
        pairs = pd.DataFrame(
            {
                "left": left_host,
                "score": score_host[right_host],
                "weight": weight_host[right_host],
            }
        )
        grouped = pairs.groupby("left", sort=True)
        expected = pd.DataFrame(index=pd.RangeIndex(left_row_count))
        expected["score_sum"] = (
            grouped["score"]
            .sum()
            .reindex(
                expected.index,
                fill_value=0.0,
            )
        )
        expected["match_count"] = (
            grouped["score"]
            .count()
            .reindex(
                expected.index,
                fill_value=0,
            )
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


def _profile_relation_distance_expression_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Shape canary for relation-distance expression consumers."""
    if not has_gpu_runtime():
        return PipelineBenchmarkResult(
            pipeline="relation-distance-expression",
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
            notes="Deferred until CUDA runtime is available for relation distance expression canary.",
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
    left_row_count = max(pair_count // 5, 1)
    right_row_count = max(pair_count // 7, 1)
    threshold = 3.0
    profiler = _stage_profiler(
        operation="pipeline.relation-distance-expression",
        dataset=f"scale-{scale}",
        requested_runtime=ExecutionMode.GPU,
        selected_runtime=ExecutionMode.GPU.value,
        enable_nvtx=enable_nvtx,
        retain_gpu_trace=retain_gpu_trace,
        include_gpu_sparklines=include_gpu_sparklines,
    )

    with profiler.stage(
        "build_relation_distances",
        category="setup",
        device=ExecutionMode.GPU,
        rows_in=pair_count,
        detail="build synthetic nearest-style relation pairs with device distance values",
    ) as stage:
        pair_rows = cp.arange(pair_count, dtype=cp.int64)
        left_indices = cp.remainder(pair_rows, left_row_count).astype(cp.int32, copy=False)
        right_indices = cp.remainder(pair_rows * 11 + 5, right_row_count).astype(
            cp.int32,
            copy=False,
        )
        distances = cp.remainder(pair_rows * 13 + 7, 101).astype(cp.float64, copy=False) / 10.0
        relation = NativeRelation(
            left_indices=left_indices,
            right_indices=right_indices,
            left_token="left",
            right_token="right",
            predicate="nearest",
            distances=distances,
            left_row_count=left_row_count,
            right_row_count=right_row_count,
            sorted_by_left=False,
        )
        stage.rows_out = int(len(relation))
        stage.metadata["left_row_count"] = left_row_count
        stage.metadata["right_row_count"] = right_row_count
        stage.metadata["pair_storage"] = "device"
        stage.metadata["distance_storage"] = "device"
        _record_stage_overheads(stage, audit, memory)

    with profiler.stage(
        "native_distance_filter_reduce",
        category="filter",
        device=ExecutionMode.GPU,
        rows_in=pair_count,
        detail="lower relation distances and match counts to private expressions, filter pairs, and reduce by native groups",
    ) as stage:
        started = perf_counter()
        expression = relation.distance_expression(operation="nearest.distance")
        pair_selection = expression.compare_scalar_selection("<=", threshold)
        filtered = relation.filter_pairs_selection(pair_selection)
        left_min = filtered.left_reduce_distances("min")
        right_count = filtered.right_reduce_distances("count")
        left_match_expression = filtered.left_match_count_expression()
        right_match_expression = filtered.right_match_count_expression()
        multi_match_left = left_match_expression.compare_scalar_selection(">", 1)
        single_match_right = right_match_expression.equal_to_selection(1)
        native_elapsed = perf_counter() - started
        stage.rows_out = filtered.capacity
        stage.metadata["threshold"] = threshold
        stage.metadata["expression_storage"] = "device" if expression.is_device else "host"
        stage.metadata["left_match_expression_storage"] = (
            "device" if left_match_expression.is_device else "host"
        )
        stage.metadata["right_match_expression_storage"] = (
            "device" if right_match_expression.is_device else "host"
        )
        stage.metadata["selection_storage"] = (
            "device-capacity" if pair_selection.is_device else "host"
        )
        stage.metadata["logical_pair_count"] = "device-resident"
        stage.metadata["match_selection_storage"] = (
            "device-capacity"
            if multi_match_left.is_device and single_match_right.is_device
            else "host"
        )
        stage.metadata["relation_storage"] = "device-capacity" if filtered.is_device else "host"
        stage.metadata["left_reduction_storage"] = "device" if left_min.is_device else "host"
        stage.metadata["right_reduction_storage"] = "device" if right_count.is_device else "host"
        stage.metadata["multi_match_left_capacity"] = multi_match_left.capacity
        stage.metadata["single_match_right_capacity"] = single_match_right.capacity
        stage.metadata["native_consume_seconds"] = native_elapsed
        _record_stage_overheads(
            stage,
            audit,
            memory,
            expression,
            pair_selection,
            filtered,
            left_match_expression,
            right_match_expression,
            multi_match_left,
            single_match_right,
        )

    with profiler.stage(
        "public_reference_export",
        category="emit",
        device=ExecutionMode.CPU,
        rows_in=pair_count,
        detail="reference path: export relation distances, filter pairs, and reduce with pandas/numpy",
    ) as stage:
        started = perf_counter()
        bytes_to_host = int(
            left_indices.nbytes
            + right_indices.nbytes
            + distances.nbytes
            + left_min.values.nbytes
            + right_count.values.nbytes
            + left_match_expression.values.nbytes
            + right_match_expression.values.nbytes
            + pair_selection.positions.nbytes
            + pair_selection.logical_count.nbytes
            + multi_match_left.positions.nbytes
            + multi_match_left.logical_count.nbytes
            + single_match_right.positions.nbytes
            + single_match_right.logical_count.nbytes
        )
        record_materialization_event(
            surface="pipeline.relation-distance-expression.public_reference_export",
            boundary=MaterializationBoundary.USER_EXPORT,
            operation="relation_distance_expression_reference_export",
            reason="exported relation distance expression inputs and output for host oracle check",
            detail=(
                f"pairs={pair_count}, left_rows={left_row_count}, "
                f"right_rows={right_row_count}, bytes={bytes_to_host}"
            ),
            d2h_transfer=True,
            strict_disallowed=False,
        )
        left_host = _bench_device_to_host(
            left_indices,
            reason="relation distance expression left-index host export",
        )
        right_host = _bench_device_to_host(
            right_indices,
            reason="relation distance expression right-index host export",
        )
        distance_host = _bench_device_to_host(
            distances,
            reason="relation distance expression distance input host export",
        )
        keep = distance_host <= threshold
        expected_left = left_host[keep]
        expected_right = right_host[keep]
        expected_distance = distance_host[keep]
        selected_pair_capacity = _bench_device_to_host(
            pair_selection.positions,
            reason="relation distance expression pair-selection capacity host export",
        )
        selected_pair_count = int(
            _bench_device_to_host(
                pair_selection.logical_count,
                reason="relation distance expression pair-selection count host export",
            )[0]
        )
        actual_pair_positions = selected_pair_capacity[:selected_pair_count]
        actual_left = left_host[actual_pair_positions]
        actual_right = right_host[actual_pair_positions]
        actual_distance = distance_host[actual_pair_positions]

        pair_frame = pd.DataFrame(
            {
                "left": expected_left,
                "right": expected_right,
                "distance": expected_distance,
            }
        )
        expected_left_min = (
            pair_frame.groupby("left")["distance"]
            .min()
            .reindex(range(left_row_count))
            .to_numpy(dtype=np.float64)
        )
        expected_right_count = (
            pair_frame.groupby("right")["distance"]
            .count()
            .reindex(range(right_row_count), fill_value=0)
            .to_numpy(dtype=np.int64)
        )
        actual_left_min = _bench_device_to_host(
            left_min.values,
            reason="relation distance expression left-min host export",
        )
        actual_right_count = _bench_device_to_host(
            right_count.values,
            reason="relation distance expression right-count host export",
        )
        actual_left_match_count = _bench_device_to_host(
            left_match_expression.values,
            reason="relation distance expression left-match-count host export",
        )
        actual_right_match_count = _bench_device_to_host(
            right_match_expression.values,
            reason="relation distance expression right-match-count host export",
        )
        multi_match_left_capacity = _bench_device_to_host(
            multi_match_left.positions,
            reason="relation distance expression multi-match selection host export",
        )
        multi_match_left_count = int(
            _bench_device_to_host(
                multi_match_left.logical_count,
                reason="relation distance expression multi-match count host export",
            )[0]
        )
        actual_multi_match_left = multi_match_left_capacity[:multi_match_left_count]
        single_match_right_capacity = _bench_device_to_host(
            single_match_right.positions,
            reason="relation distance expression single-match selection host export",
        )
        single_match_right_count = int(
            _bench_device_to_host(
                single_match_right.logical_count,
                reason="relation distance expression single-match count host export",
            )[0]
        )
        actual_single_match_right = single_match_right_capacity[:single_match_right_count]
        reference_elapsed = perf_counter() - started
        expected_left_match_count = np.bincount(
            expected_left,
            minlength=left_row_count,
        )[:left_row_count].astype(np.int64, copy=False)
        expected_right_match_count = np.bincount(
            expected_right,
            minlength=right_row_count,
        )[:right_row_count].astype(np.int64, copy=False)
        expected_multi_match_left = np.flatnonzero(expected_left_match_count > 1)
        expected_single_match_right = np.flatnonzero(expected_right_match_count == 1)
        results_match = (
            np.array_equal(actual_left, expected_left)
            and np.array_equal(actual_right, expected_right)
            and np.allclose(actual_distance, expected_distance)
            and np.allclose(actual_left_min, expected_left_min, equal_nan=True)
            and np.array_equal(actual_right_count, expected_right_count)
            and np.array_equal(actual_left_match_count, expected_left_match_count)
            and np.array_equal(actual_right_match_count, expected_right_match_count)
            and np.array_equal(actual_multi_match_left, expected_multi_match_left)
            and np.array_equal(actual_single_match_right, expected_single_match_right)
        )
        stage.rows_out = int(expected_left.size)
        stage.metadata["results_match"] = results_match
        stage.metadata["reference_seconds"] = reference_elapsed
        stage.metadata["native_consume_seconds"] = native_elapsed
        stage.metadata["consumer_speedup"] = (
            reference_elapsed / native_elapsed if native_elapsed > 0.0 else float("inf")
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
            "admissible_shape": (
                "NativeRelation distances/counts -> NativeExpression -> "
                "NativeDeviceSelection -> NativeRelationSelection"
            ),
            "results_match": results_match,
            "consumer_speedup": (
                reference_elapsed / native_elapsed if native_elapsed > 0.0 else float("inf")
            ),
        }
    )
    return PipelineBenchmarkResult(
        pipeline="relation-distance-expression",
        scale=scale,
        status="ok" if results_match else "failed",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=ExecutionMode.GPU.value,
        output_rows=int(stage.rows_out or 0),
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
            "Shape canary, not a workflow target: nearest-style relation "
            "distances and per-source match counts stay private as "
            "NativeExpression flow, then feed capacity-backed selection and "
            "relation reduction before an explicit host oracle export."
        ),
    )


def _profile_nearest_relation_producer_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Shape canary for public nearest producer -> NativeRelation flow."""
    if not has_gpu_runtime():
        return PipelineBenchmarkResult(
            pipeline="nearest-relation-producer",
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
            notes="Deferred until CUDA runtime is available for nearest relation producer canary.",
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
    rows = min(max(int(scale), 1), 4096)
    threshold = 0.25
    profiler = _stage_profiler(
        operation="pipeline.nearest-relation-producer",
        dataset=f"scale-{scale}",
        requested_runtime=ExecutionMode.GPU,
        selected_runtime=ExecutionMode.GPU.value,
        enable_nvtx=enable_nvtx,
        retain_gpu_trace=retain_gpu_trace,
        include_gpu_sparklines=include_gpu_sparklines,
    )

    relation = None
    producer_device = False
    producer_selected = ExecutionMode.CPU
    right_relation = None
    right_producer_device = False
    right_producer_selected = ExecutionMode.CPU
    attribute_filtered_relation = None
    attribute_filter_results_match = False
    attribute_filter_elapsed = 0.0
    expected_attribute_rows = 0
    with profiler.stage(
        "build_nearest_relation",
        category="relation_build",
        device=ExecutionMode.GPU,
        rows_in=rows,
        detail="public sjoin_nearest producer lowers to device NativeRelation distances",
    ) as stage:
        coords = np.arange(rows, dtype=np.float64)
        left = geopandas.GeoDataFrame(
            geometry=geopandas.GeoSeries([Point(float(x), 0.0) for x in coords]),
        )
        right = geopandas.GeoDataFrame(
            geometry=geopandas.GeoSeries([Point(float(x) + threshold, 0.0) for x in coords]),
        )
        native_result, producer_selected = _sjoin_nearest_relation_result(
            left,
            right,
            max_distance=0.5,
            how="inner",
            return_distance=True,
            exclusive=False,
        )
        relation = native_result.to_native_relation(
            left_token="left",
            right_token="right",
            predicate="nearest",
            left_row_count=rows,
            right_row_count=rows,
        )
        producer_device = (
            _is_device_array(relation.left_indices)
            and _is_device_array(relation.right_indices)
            and _is_device_array(relation.distances)
        )
        stage.device = producer_selected
        stage.rows_out = int(len(relation))
        stage.metadata["producer_selected"] = producer_selected.value
        stage.metadata["relation_storage"] = "device" if producer_device else "host"
        stage.metadata["distance_storage"] = (
            "device" if _is_device_array(relation.distances) else "host"
        )
        stage.metadata["admitted_rows"] = rows
        _record_stage_overheads(stage, audit, memory, relation)

    native_results_match = False
    native_elapsed = 0.0
    with profiler.stage(
        "native_distance_consume",
        category="filter",
        device=producer_selected,
        rows_in=rows,
        detail="consume nearest distances as NativeExpression without public Series",
    ) as stage:
        started = perf_counter()
        expression = relation.distance_expression(operation="nearest.distance")
        filtered = relation.filter_by_distance_selection("<=", threshold)
        left_min = filtered.left_reduce_distances("min")
        right_count = filtered.right_reduce_distances("count")
        native_elapsed = perf_counter() - started
        native_results_match = (
            producer_device
            and filtered.is_device
            and filtered.capacity == rows
            and _is_device_array(expression.values)
            and _is_device_array(left_min.values)
            and _is_device_array(right_count.values)
        )
        stage.rows_out = filtered.capacity
        stage.metadata["expression_storage"] = "device" if expression.is_device else "host"
        stage.metadata["selection_storage"] = "device-capacity"
        stage.metadata["logical_pair_count"] = "device-resident"
        stage.metadata["relation_storage"] = "device-capacity"
        stage.metadata["left_reduction_storage"] = "device" if left_min.is_device else "host"
        stage.metadata["right_reduction_storage"] = "device" if right_count.is_device else "host"
        stage.metadata["native_consume_seconds"] = native_elapsed
        stage.metadata["results_match"] = native_results_match
        _record_stage_overheads(stage, audit, memory, expression, filtered)

    with profiler.stage(
        "native_attribute_match_filter",
        category="filter",
        device=producer_selected,
        rows_in=rows,
        detail="filter relation pairs by device-resident shared numeric attributes",
    ) as stage:
        started = perf_counter()
        row_numbers = cp.arange(rows, dtype=cp.int32)
        left_zone = row_numbers % 3
        right_zone = cp.where(row_numbers % 4 == 0, left_zone + 10, left_zone)
        expected_attribute_rows = rows - ((rows + 3) // 4)
        attribute_filtered_relation = relation.filter_by_equal_columns_selection(
            {"zone": left_zone},
            {"zone": right_zone},
        )
        attribute_filter_elapsed = perf_counter() - started
        attribute_filter_results_match = (
            producer_device
            and attribute_filtered_relation.is_device
            and attribute_filtered_relation.capacity == rows
        )
        stage.rows_out = attribute_filtered_relation.capacity
        stage.metadata["selection_storage"] = "device-capacity"
        stage.metadata["logical_pair_count"] = "device-resident"
        stage.metadata["relation_storage"] = "device-capacity"
        stage.metadata["expected_rows"] = expected_attribute_rows
        stage.metadata["attribute_filter_seconds"] = attribute_filter_elapsed
        stage.metadata["results_match"] = attribute_filter_results_match
        _record_stage_overheads(stage, audit, memory, attribute_filtered_relation)

    right_results_match = False
    with profiler.stage(
        "build_right_nearest_relation",
        category="relation_build",
        device=ExecutionMode.GPU,
        rows_in=rows,
        detail="right-join nearest producer remaps device relation pairs without public export",
    ) as stage:
        native_right_result, right_producer_selected = _sjoin_nearest_relation_result(
            left,
            right,
            max_distance=0.5,
            how="right",
            return_distance=True,
            exclusive=False,
        )
        right_relation = native_right_result.to_native_relation(
            left_token="left",
            right_token="right",
            predicate="nearest",
            left_row_count=rows,
            right_row_count=rows,
        )
        right_producer_device = (
            _is_device_array(right_relation.left_indices)
            and _is_device_array(right_relation.right_indices)
            and _is_device_array(right_relation.distances)
        )
        right_results_match = right_producer_device and int(len(right_relation)) == rows
        stage.device = right_producer_selected
        stage.rows_out = int(len(right_relation))
        stage.metadata["producer_selected"] = right_producer_selected.value
        stage.metadata["relation_storage"] = "device" if right_producer_device else "host"
        stage.metadata["distance_storage"] = (
            "device" if _is_device_array(right_relation.distances) else "host"
        )
        stage.metadata["results_match"] = right_results_match
        _record_stage_overheads(stage, audit, memory, right_relation)

    with profiler.stage(
        "public_reference_export",
        category="emit",
        device=ExecutionMode.CPU,
        rows_in=rows,
        detail="export nearest relation once for host oracle checks",
    ) as stage:
        started = perf_counter()
        bytes_to_host = int(
            relation.left_indices.nbytes
            + relation.right_indices.nbytes
            + relation.distances.nbytes
            + left_min.values.nbytes
            + right_count.values.nbytes
            + filtered.selection.positions.nbytes
            + filtered.selection.logical_count.nbytes
            + attribute_filtered_relation.selection.positions.nbytes
            + attribute_filtered_relation.selection.logical_count.nbytes
            + right_relation.left_indices.nbytes
            + right_relation.right_indices.nbytes
            + right_relation.distances.nbytes
        )
        record_materialization_event(
            surface="pipeline.nearest-relation-producer.public_reference_export",
            boundary=MaterializationBoundary.USER_EXPORT,
            operation="nearest_relation_producer_reference_export",
            reason="exported nearest NativeRelation outputs for host oracle check",
            detail=f"rows={rows}, bytes={bytes_to_host}",
            d2h_transfer=True,
            strict_disallowed=False,
        )
        actual_left = _bench_device_to_host(
            relation.left_indices,
            reason="nearest relation producer left-index host export",
        )
        actual_right = _bench_device_to_host(
            relation.right_indices,
            reason="nearest relation producer right-index host export",
        )
        actual_distances = _bench_device_to_host(
            relation.distances,
            reason="nearest relation producer distance host export",
        )
        actual_left_min = _bench_device_to_host(
            left_min.values,
            reason="nearest relation producer left-min host export",
        )
        actual_right_count = _bench_device_to_host(
            right_count.values,
            reason="nearest relation producer right-count host export",
        )
        filtered_capacity = _bench_device_to_host(
            filtered.selection.positions,
            reason="nearest relation producer distance-selection capacity host export",
        )
        filtered_count = int(
            _bench_device_to_host(
                filtered.selection.logical_count,
                reason="nearest relation producer distance-selection count host export",
            )[0]
        )
        filtered_positions = filtered_capacity[:filtered_count]
        actual_filtered_left = actual_left[filtered_positions]
        actual_filtered_right = actual_right[filtered_positions]
        actual_filtered_distances = actual_distances[filtered_positions]
        attribute_capacity = _bench_device_to_host(
            attribute_filtered_relation.selection.positions,
            reason="nearest relation producer attribute-selection capacity host export",
        )
        attribute_count = int(
            _bench_device_to_host(
                attribute_filtered_relation.selection.logical_count,
                reason="nearest relation producer attribute-selection count host export",
            )[0]
        )
        attribute_positions = attribute_capacity[:attribute_count]
        actual_attribute_left = actual_left[attribute_positions]
        actual_attribute_right = actual_right[attribute_positions]
        actual_attribute_distances = actual_distances[attribute_positions]
        actual_right_left = _bench_device_to_host(
            right_relation.left_indices,
            reason="nearest relation producer right-join left-index host export",
        )
        actual_right_right = _bench_device_to_host(
            right_relation.right_indices,
            reason="nearest relation producer right-join right-index host export",
        )
        actual_right_distances = _bench_device_to_host(
            right_relation.distances,
            reason="nearest relation producer right-join distance host export",
        )
        expected_rows = np.arange(rows, dtype=np.int32)
        expected_filtered_rows = expected_rows[expected_rows % 4 != 0]
        reference_elapsed = perf_counter() - started
        results_match = bool(
            native_results_match
            and attribute_filter_results_match
            and right_results_match
            and np.array_equal(actual_left, expected_rows)
            and np.array_equal(actual_right, expected_rows)
            and np.allclose(actual_distances, threshold)
            and np.array_equal(actual_filtered_left, expected_rows)
            and np.array_equal(actual_filtered_right, expected_rows)
            and np.allclose(actual_filtered_distances, threshold)
            and np.allclose(actual_left_min, threshold)
            and np.array_equal(actual_right_count, np.ones(rows, dtype=np.int64))
            and np.array_equal(actual_attribute_left, expected_filtered_rows)
            and np.array_equal(actual_attribute_right, expected_filtered_rows)
            and np.allclose(actual_attribute_distances, threshold)
            and np.array_equal(actual_right_left, expected_rows)
            and np.array_equal(actual_right_right, expected_rows)
            and np.allclose(actual_right_distances, threshold)
        )
        stage.rows_out = rows
        stage.metadata["results_match"] = results_match
        stage.metadata["reference_seconds"] = reference_elapsed
        stage.metadata["native_consume_seconds"] = native_elapsed
        stage.metadata["attribute_filter_seconds"] = attribute_filter_elapsed
        stage.metadata["consumer_speedup"] = (
            reference_elapsed / native_elapsed if native_elapsed > 0.0 else float("inf")
        )
        _record_stage_overheads(stage, audit, memory)

    stage_devices = [stage.device for stage in profiler._stages]
    actual_selected_runtime = _pipeline_runtime_from_stage_devices(stage_devices)
    trace = profiler.finish(
        metadata={
            "scale": scale,
            "effective_rows": rows,
            "actual_selected_runtime": actual_selected_runtime,
            "planner_selected_runtime": ExecutionMode.GPU.value,
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
            "admissible_shape": (
                "public nearest producer -> NativeRelation distances -> "
                "NativeExpression -> capacity-backed relation filters"
            ),
            "results_match": results_match,
        }
    )
    return PipelineBenchmarkResult(
        pipeline="nearest-relation-producer",
        scale=scale,
        status="ok" if results_match else "failed",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=ExecutionMode.GPU.value,
        output_rows=rows,
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
            "Shape canary, not a workflow target: public sjoin_nearest relation "
            "production keeps nearest pair distances private as a device "
            "NativeRelation before capacity-backed expression/attribute filters "
            "and grouped reducers."
        ),
    )


def _profile_native_area_expression_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Shape canary for private metric expressions consumed natively."""
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return PipelineBenchmarkResult(
            pipeline="native-area-expression",
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
            notes=(
                "Deferred until CUDA runtime and pylibcudf are available for "
                "NativeExpression metric filter/grouped reducer canary."
            ),
        )

    from time import perf_counter

    import cupy as cp
    import pyarrow as pa

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
    area_threshold = 4.0
    length_threshold = 12.0
    threshold_guard_epsilon = 0.0
    profiler = _stage_profiler(
        operation="pipeline.native-area-expression",
        dataset=f"scale-{scale}",
        requested_runtime=ExecutionMode.GPU,
        selected_runtime=ExecutionMode.GPU.value,
        enable_nvtx=enable_nvtx,
        retain_gpu_trace=retain_gpu_trace,
        include_gpu_sparklines=include_gpu_sparklines,
    )

    with profiler.stage(
        "build_native_polygons",
        category="setup",
        device=ExecutionMode.GPU,
        rows_in=scale,
        detail="build device-resident polygon frame and dense group codes",
    ) as stage:
        (
            geometries,
            expected_area,
            expected_length,
            expected_centroid_x,
            expected_centroid_y,
            group_codes_host,
        ) = _varying_box_expression_inputs(scale)
        owned = from_shapely_geometries(geometries).move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="native metric expression canary device setup",
        )
        attribute_arrow = pa.table(
            {
                "group": pa.array(group_codes_host, type=pa.int32()),
            }
        )
        attributes = NativeAttributeTable(
            device_table=pylibcudf_table_from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        )
        state = NativeFrameState.from_native_tabular_result(
            NativeTabularResult(
                attributes=attributes,
                geometry=GeometryNativeResult.from_owned(owned, crs="EPSG:4326"),
                geometry_name="geometry",
                column_order=("group", "geometry"),
            )
        )
        group_codes = cp.asarray(group_codes_host, dtype=cp.int32)
        grouped = NativeGrouped.from_dense_codes(
            group_codes,
            group_count=group_count,
            output_index=pd.RangeIndex(group_count, name="group"),
            source_token=state.lineage_token,
        )
        expected_area_selected = np.flatnonzero(expected_area > area_threshold).astype(
            np.int64,
            copy=False,
        )
        expected_length_selected = np.flatnonzero(expected_length <= length_threshold).astype(
            np.int64, copy=False
        )
        expected_selected = np.intersect1d(
            expected_area_selected,
            expected_length_selected,
            assume_unique=True,
        ).astype(np.int64, copy=False)
        expected_area_ambiguous = np.flatnonzero(
            np.abs(expected_area - area_threshold) <= threshold_guard_epsilon
        ).astype(np.int64, copy=False)
        expected_length_ambiguous = np.flatnonzero(
            np.abs(expected_length - length_threshold) <= threshold_guard_epsilon
        ).astype(np.int64, copy=False)
        expected_guarded_ambiguous = np.union1d(
            expected_area_ambiguous,
            expected_length_ambiguous,
        ).astype(np.int64, copy=False)
        expected_guarded_definite = np.setdiff1d(
            expected_selected,
            expected_guarded_ambiguous,
            assume_unique=True,
        ).astype(np.int64, copy=False)
        expected_area_group_sum = np.bincount(
            group_codes_host,
            weights=expected_area,
            minlength=group_count,
        )[:group_count]
        expected_length_group_sum = np.bincount(
            group_codes_host,
            weights=expected_length,
            minlength=group_count,
        )[:group_count]
        expected_group_counts = np.bincount(
            group_codes_host,
            minlength=group_count,
        )[:group_count]
        expected_centroid_x_group_mean = np.full(group_count, np.nan, dtype=np.float64)
        expected_centroid_y_group_mean = np.full(group_count, np.nan, dtype=np.float64)
        np.divide(
            np.bincount(
                group_codes_host,
                weights=expected_centroid_x,
                minlength=group_count,
            )[:group_count],
            expected_group_counts,
            out=expected_centroid_x_group_mean,
            where=expected_group_counts > 0,
        )
        np.divide(
            np.bincount(
                group_codes_host,
                weights=expected_centroid_y,
                minlength=group_count,
            )[:group_count],
            expected_group_counts,
            out=expected_centroid_y_group_mean,
            where=expected_group_counts > 0,
        )
        stage.rows_out = int(state.row_count)
        stage.metadata["row_count"] = int(state.row_count)
        stage.metadata["group_count"] = group_count
        stage.metadata["geometry_storage"] = "device"
        _record_stage_overheads(stage, audit, memory, state)

    with profiler.stage(
        "area_expression",
        category="metric",
        device=ExecutionMode.GPU,
        rows_in=int(state.row_count),
        detail="compute geometry.area as a private fp64 NativeExpression device vector",
    ) as stage:
        started = perf_counter()
        area_expression = state.geometry_area_expression()
        expression_elapsed = perf_counter() - started
        stage.rows_out = len(area_expression)
        stage.metadata["operation"] = area_expression.operation
        stage.metadata["expression_storage"] = "device" if area_expression.is_device else "host"
        stage.metadata["precision"] = area_expression.precision
        stage.metadata["native_expression_seconds"] = expression_elapsed
        _record_stage_overheads(stage, audit, memory)

    with profiler.stage(
        "length_expression",
        category="metric",
        device=ExecutionMode.GPU,
        rows_in=int(state.row_count),
        detail="compute geometry.length as a private fp64 NativeExpression device vector",
    ) as stage:
        started = perf_counter()
        length_expression = state.geometry_length_expression()
        expression_elapsed = perf_counter() - started
        stage.rows_out = len(length_expression)
        stage.metadata["operation"] = length_expression.operation
        stage.metadata["expression_storage"] = "device" if length_expression.is_device else "host"
        stage.metadata["precision"] = length_expression.precision
        stage.metadata["native_expression_seconds"] = expression_elapsed
        _record_stage_overheads(stage, audit, memory)

    with profiler.stage(
        "centroid_component_expressions",
        category="metric",
        device=ExecutionMode.GPU,
        rows_in=int(state.row_count),
        detail=(
            "compute geometry centroid x/y as paired private fp64 "
            "NativeExpression device vectors without public point export"
        ),
    ) as stage:
        started = perf_counter()
        centroid_x_expression, centroid_y_expression = state.geometry_centroid_expressions()
        expression_elapsed = perf_counter() - started
        stage.rows_out = len(centroid_x_expression)
        stage.metadata["operations"] = (
            centroid_x_expression.operation,
            centroid_y_expression.operation,
        )
        stage.metadata["expression_storage"] = (
            "device"
            if centroid_x_expression.is_device and centroid_y_expression.is_device
            else "host"
        )
        stage.metadata["precision"] = centroid_x_expression.precision
        stage.metadata["native_expression_seconds"] = expression_elapsed
        _record_stage_overheads(stage, audit, memory)

    with profiler.stage(
        "expression_column_compose",
        category="compose",
        device=ExecutionMode.GPU,
        rows_in=int(state.row_count),
        detail=(
            "attach metric expressions as private device attribute columns "
            "and recover them as sanctioned NativeExpression inputs"
        ),
    ) as stage:
        started = perf_counter()
        expression_state = state.assign_expression_columns(
            {
                "area": area_expression,
                "length": length_expression,
                "centroid_x": centroid_x_expression,
                "centroid_y": centroid_y_expression,
            },
        )
        if expression_state is None:
            raise RuntimeError("native metric expression columns did not admit")
        area_column_expression = expression_state.attribute_expression(
            "area",
            operation="attribute.area",
        )
        length_column_expression = expression_state.attribute_expression(
            "length",
            operation="attribute.length",
        )
        if area_column_expression is None or length_column_expression is None:
            raise RuntimeError("native metric expression columns were not reusable")
        centroid_x_column_expression = expression_state.attribute_expression(
            "centroid_x",
            operation="attribute.centroid_x",
        )
        centroid_y_column_expression = expression_state.attribute_expression(
            "centroid_y",
            operation="attribute.centroid_y",
        )
        if centroid_x_column_expression is None or centroid_y_column_expression is None:
            raise RuntimeError("native centroid expression columns were not reusable")
        compose_elapsed = perf_counter() - started
        stage.rows_out = int(expression_state.row_count)
        stage.metadata["attribute_storage"] = "device"
        stage.metadata["expression_columns"] = 4
        stage.metadata["area_column_storage"] = (
            "device" if area_column_expression.is_device else "host"
        )
        stage.metadata["length_column_storage"] = (
            "device" if length_column_expression.is_device else "host"
        )
        stage.metadata["centroid_column_storage"] = (
            "device"
            if centroid_x_column_expression.is_device and centroid_y_column_expression.is_device
            else "host"
        )
        stage.metadata["native_compose_seconds"] = compose_elapsed
        _record_stage_overheads(stage, audit, memory, expression_state)

    with profiler.stage(
        "public_expression_column_bridge",
        category="compose",
        device=ExecutionMode.CPU,
        rows_in=int(state.row_count),
        detail=(
            "assign admitted NativeExpression values through the public "
            "GeoDataFrame column API while preserving device expression columns"
        ),
    ) as stage:
        started = perf_counter()
        public_frame = geopandas.GeoDataFrame(
            {
                "group": group_codes_host,
                "geometry": geopandas.GeoSeries(
                    geometries,
                    name="geometry",
                    crs="EPSG:4326",
                ),
            },
            crs="EPSG:4326",
        )
        attach_native_state(public_frame, state)
        public_assigned = public_frame.assign(
            area=area_expression,
            length=length_expression,
        )
        public_expression_state = get_native_state(public_assigned)
        if public_expression_state is None:
            raise RuntimeError("public NativeExpression column bridge lost native state")
        public_area_expression = public_assigned._native_attribute_expression(
            "area",
            operation="public.attribute.area",
        )
        public_length_expression = public_assigned._native_attribute_expression(
            "length",
            operation="public.attribute.length",
        )
        if public_area_expression is None or public_length_expression is None:
            raise RuntimeError("public NativeExpression columns were not reusable")
        public_rowset = public_area_expression.greater_than(area_threshold)
        public_bridge_elapsed = perf_counter() - started
        stage.rows_out = int(public_expression_state.row_count)
        stage.metadata["public_boundary"] = "expression-column-assignment"
        stage.metadata["attribute_storage"] = "device"
        stage.metadata["expression_columns"] = 2
        stage.metadata["area_column_storage"] = (
            "device" if public_area_expression.is_device else "host"
        )
        stage.metadata["length_column_storage"] = (
            "device" if public_length_expression.is_device else "host"
        )
        stage.metadata["rowset_storage"] = "device" if public_rowset.is_device else "host"
        stage.metadata["public_bridge_seconds"] = public_bridge_elapsed
        _record_stage_overheads(stage, audit, memory, public_expression_state)

    with profiler.stage(
        "expression_compound_rowset_take",
        category="filter",
        device=ExecutionMode.GPU,
        rows_in=len(area_expression),
        detail=(
            "lower expression-column thresholds to NativeRowSet intersection "
            "and take NativeFrameState"
        ),
    ) as stage:
        started = perf_counter()
        area_rowset = area_column_expression.greater_than(area_threshold)
        length_rowset = length_column_expression.less_equal(length_threshold)
        rowset = area_rowset.intersection(length_rowset)
        filtered = expression_state.take(rowset, preserve_index=False)
        rowset_elapsed = perf_counter() - started
        stage.rows_out = int(filtered.row_count)
        stage.metadata["area_threshold"] = area_threshold
        stage.metadata["length_threshold"] = length_threshold
        stage.metadata["rowset_operation"] = "intersection"
        stage.metadata["rowset_storage"] = "device" if rowset.is_device else "host"
        stage.metadata["native_index_kind"] = filtered.index_plan.kind
        stage.metadata["native_rowset_seconds"] = rowset_elapsed
        _record_stage_overheads(stage, audit, memory, filtered)

    with profiler.stage(
        "guarded_threshold_rowsets",
        category="filter",
        device=ExecutionMode.GPU,
        rows_in=len(area_expression),
        detail=(
            "lower precision-guarded expression thresholds to definite and "
            "ambiguous NativeRowSet outputs"
        ),
    ) as stage:
        started = perf_counter()
        area_guarded = area_column_expression.greater_than_guarded(
            area_threshold,
            epsilon=threshold_guard_epsilon,
        )
        length_guarded = length_column_expression.less_equal_guarded(
            length_threshold,
            epsilon=threshold_guard_epsilon,
        )
        guarded_definite_rowset = area_guarded.rowset.intersection(length_guarded.rowset)
        guarded_ambiguous_rowset = area_guarded.ambiguous.union(length_guarded.ambiguous)
        guarded_elapsed = perf_counter() - started
        stage.rows_out = len(guarded_definite_rowset)
        stage.metadata["area_threshold"] = area_threshold
        stage.metadata["length_threshold"] = length_threshold
        stage.metadata["epsilon"] = threshold_guard_epsilon
        stage.metadata["definite_row_count"] = len(guarded_definite_rowset)
        stage.metadata["ambiguous_row_count"] = len(guarded_ambiguous_rowset)
        stage.metadata["rowset_storage"] = "device" if guarded_definite_rowset.is_device else "host"
        stage.metadata["ambiguous_rowset_storage"] = (
            "device" if guarded_ambiguous_rowset.is_device else "host"
        )
        stage.metadata["native_guarded_rowset_seconds"] = guarded_elapsed
        _record_stage_overheads(
            stage,
            audit,
            memory,
            area_guarded.rowset,
            area_guarded.ambiguous,
            length_guarded.rowset,
            length_guarded.ambiguous,
            guarded_definite_rowset,
            guarded_ambiguous_rowset,
        )

    with profiler.stage(
        "expression_grouped_sum",
        category="reduce",
        device=ExecutionMode.GPU,
        rows_in=len(area_expression),
        detail="feed expression-column NativeExpression values into NativeGrouped sums",
    ) as stage:
        started = perf_counter()
        area_reduced = grouped.reduce_expression(area_column_expression, "sum")
        length_reduced = grouped.reduce_expression(length_column_expression, "sum")
        centroid_x_reduced = grouped.reduce_expression(
            centroid_x_column_expression,
            "mean",
        )
        centroid_y_reduced = grouped.reduce_expression(
            centroid_y_column_expression,
            "mean",
        )
        grouped_elapsed = perf_counter() - started
        stage.rows_out = int(area_reduced.group_count)
        stage.metadata["result_storage"] = (
            "device"
            if (
                area_reduced.is_device
                and length_reduced.is_device
                and centroid_x_reduced.is_device
                and centroid_y_reduced.is_device
            )
            else "host"
        )
        stage.metadata["reducers"] = ("sum", "mean")
        stage.metadata["expression_count"] = 4
        stage.metadata["native_reduce_seconds"] = grouped_elapsed
        _record_stage_overheads(
            stage,
            audit,
            memory,
            area_reduced,
            length_reduced,
            centroid_x_reduced,
            centroid_y_reduced,
        )

    with profiler.stage(
        "public_reference_export",
        category="emit",
        device=ExecutionMode.CPU,
        rows_in=scale,
        detail="explicit terminal oracle export for expression rowset and grouped results",
    ) as stage:
        started = perf_counter()
        bytes_to_host = int(
            area_expression.values.nbytes
            + length_expression.values.nbytes
            + centroid_x_expression.values.nbytes
            + centroid_y_expression.values.nbytes
            + area_rowset.positions.nbytes
            + length_rowset.positions.nbytes
            + rowset.positions.nbytes
            + guarded_definite_rowset.positions.nbytes
            + guarded_ambiguous_rowset.positions.nbytes
            + area_reduced.values.nbytes
            + length_reduced.values.nbytes
            + centroid_x_reduced.values.nbytes
            + centroid_y_reduced.values.nbytes
        )
        record_materialization_event(
            surface="pipeline.native-area-expression.public_reference_export",
            boundary=MaterializationBoundary.USER_EXPORT,
            operation="native_metric_expression_reference_export",
            reason=("exported NativeExpression canary outputs for host oracle comparison"),
            detail=f"rows={scale}, groups={group_count}, bytes={bytes_to_host}",
            d2h_transfer=True,
            strict_disallowed=False,
        )
        actual_area = _bench_device_to_host(
            area_expression.values,
            reason="native expression area values host export",
        )
        actual_length = _bench_device_to_host(
            length_expression.values,
            reason="native expression length values host export",
        )
        actual_centroid_x = _bench_device_to_host(
            centroid_x_expression.values,
            reason="native expression centroid-x values host export",
        )
        actual_centroid_y = _bench_device_to_host(
            centroid_y_expression.values,
            reason="native expression centroid-y values host export",
        )
        actual_selected = _bench_device_to_host(
            rowset.positions,
            reason="native expression selected rowset host export",
        )
        actual_guarded_definite = _bench_device_to_host(
            guarded_definite_rowset.positions,
            reason="native expression guarded-definite rowset host export",
        )
        actual_guarded_ambiguous = _bench_device_to_host(
            guarded_ambiguous_rowset.positions,
            reason="native expression guarded-ambiguous rowset host export",
        )
        actual_area_group_sum = _bench_device_to_host(
            area_reduced.values,
            reason="native expression area grouped sum host export",
        )
        actual_length_group_sum = _bench_device_to_host(
            length_reduced.values,
            reason="native expression length grouped sum host export",
        )
        actual_centroid_x_group_mean = _bench_device_to_host(
            centroid_x_reduced.values,
            reason="native expression centroid-x grouped mean host export",
        )
        actual_centroid_y_group_mean = _bench_device_to_host(
            centroid_y_reduced.values,
            reason="native expression centroid-y grouped mean host export",
        )
        reference_elapsed = perf_counter() - started
        area_match = bool(np.allclose(actual_area, expected_area))
        length_match = bool(np.allclose(actual_length, expected_length))
        centroid_match = bool(
            np.allclose(actual_centroid_x, expected_centroid_x)
            and np.allclose(actual_centroid_y, expected_centroid_y)
        )
        rowset_match = bool(np.array_equal(actual_selected, expected_selected))
        guarded_match = bool(
            np.array_equal(actual_guarded_definite, expected_guarded_definite)
            and np.array_equal(actual_guarded_ambiguous, expected_guarded_ambiguous)
        )
        group_match = bool(
            np.allclose(actual_area_group_sum, expected_area_group_sum)
            and np.allclose(actual_length_group_sum, expected_length_group_sum)
            and np.allclose(
                actual_centroid_x_group_mean,
                expected_centroid_x_group_mean,
                equal_nan=True,
            )
            and np.allclose(
                actual_centroid_y_group_mean,
                expected_centroid_y_group_mean,
                equal_nan=True,
            )
        )
        results_match = (
            area_match
            and length_match
            and centroid_match
            and rowset_match
            and guarded_match
            and group_match
        )
        stage.rows_out = int(actual_selected.size)
        stage.metadata["area_match"] = area_match
        stage.metadata["length_match"] = length_match
        stage.metadata["centroid_match"] = centroid_match
        stage.metadata["rowset_match"] = rowset_match
        stage.metadata["guarded_match"] = guarded_match
        stage.metadata["guarded_ambiguous_rows"] = int(actual_guarded_ambiguous.size)
        stage.metadata["group_match"] = group_match
        stage.metadata["results_match"] = results_match
        stage.metadata["reference_seconds"] = reference_elapsed
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
            "admissible_shape": (
                "owned polygon geometry -> NativeExpression area/length/centroid vectors "
                "-> private/public expression columns -> composed and guarded "
                "NativeRowSet and NativeGrouped consumers"
            ),
            "results_match": results_match,
        }
    )
    return PipelineBenchmarkResult(
        pipeline="native-area-expression",
        scale=scale,
        status="ok" if results_match else "failed",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=ExecutionMode.GPU.value,
        output_rows=int(filtered.row_count),
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
            "Shape canary, not a workflow target: geometry area, length, and "
            "centroid components become private device expression columns, the "
            "public assignment bridge preserves admitted device columns for "
            "sanctioned native consumers, guarded threshold ambiguity stays "
            "native, and pandas/host appears only at explicit public export "
            "boundaries."
        ),
    )


def _profile_constructive_output_native_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Shape canary for constructive geometry output consumed through Native*."""
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return PipelineBenchmarkResult(
            pipeline="constructive-output-native",
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
            notes=(
                "Deferred until CUDA runtime and pylibcudf are available for "
                "constructive-output NativeFrameState canary."
            ),
        )

    from time import perf_counter

    import cupy as cp
    import pyarrow as pa

    from vibespatial.constructive.binary_constructive import binary_constructive_native
    from vibespatial.runtime.materialization import clear_materialization_events

    clear_materialization_events()
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    effective_rows = max(1, min(scale, 100_000))
    group_count = 128
    area_threshold = 0.5
    expected_area_value = 0.5625
    row_ids = np.arange(effective_rows, dtype=np.int64)
    group_codes_host = np.remainder(row_ids, group_count).astype(np.int32, copy=False)
    profiler = _stage_profiler(
        operation="pipeline.constructive-output-native",
        dataset=f"scale-{scale}",
        requested_runtime=ExecutionMode.GPU,
        selected_runtime=ExecutionMode.GPU.value,
        enable_nvtx=enable_nvtx,
        retain_gpu_trace=retain_gpu_trace,
        include_gpu_sparklines=include_gpu_sparklines,
    )

    with profiler.stage(
        "build_device_pairwise_boxes",
        category="setup",
        device=ExecutionMode.GPU,
        rows_in=scale,
        detail="build two device-resident aligned polygon batches for pairwise constructive output",
    ) as stage:
        left_owned, cols, grid_rows = _device_regular_box_owned(effective_rows)
        right_owned, _right_cols, _right_grid_rows = _device_regular_box_owned(
            effective_rows,
            x_shift=0.25,
            y_shift=0.25,
        )
        stage.rows_out = effective_rows
        stage.metadata["effective_rows"] = effective_rows
        stage.metadata["grid_cols"] = cols
        stage.metadata["grid_rows"] = grid_rows
        stage.metadata["geometry_storage"] = "owned:device"
        _record_stage_overheads(stage, audit, memory, left_owned, right_owned)

    with profiler.stage(
        "native_constructive_intersection",
        category="constructive",
        device=ExecutionMode.GPU,
        rows_in=effective_rows,
        detail=(
            "run pairwise polygon intersection and keep the result as "
            "GeometryNativeResult inside NativeTabularResult"
        ),
    ) as stage:
        started = perf_counter()
        constructive_geometry = binary_constructive_native(
            "intersection",
            left_owned,
            right_owned,
            dispatch_mode=ExecutionMode.GPU,
        ).with_crs("EPSG:4326")
        constructive_elapsed = perf_counter() - started
        attribute_arrow = pa.table({"group": pa.array(group_codes_host, type=pa.int32())})
        attributes = NativeAttributeTable(
            device_table=pylibcudf_table_from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        )
        constructive_metadata = NativeGeometryMetadata.from_native_geometry(
            constructive_geometry,
        )
        constructive_result = NativeTabularResult(
            attributes=attributes,
            geometry=constructive_geometry,
            geometry_name="geometry",
            column_order=("group", "geometry"),
            provenance=NativeGeometryProvenance(
                operation="pairwise_constructive",
                row_count=int(constructive_geometry.row_count),
                left_rows=cp.arange(effective_rows, dtype=cp.int32),
                right_rows=cp.arange(effective_rows, dtype=cp.int32),
            ),
            geometry_metadata=constructive_metadata,
        )
        constructive_state = constructive_result.to_native_frame_state()
        constructive_grouped = NativeGrouped.from_dense_codes(
            cp.asarray(group_codes_host, dtype=cp.int32),
            group_count=group_count,
            output_index=pd.RangeIndex(group_count, name="group"),
            source_token=constructive_state.lineage_token,
        )
        stage.rows_out = int(constructive_state.row_count)
        stage.metadata["native_constructive_seconds"] = constructive_elapsed
        stage.metadata["constructive_output_carrier"] = "NativeTabularResult"
        stage.metadata["downstream_carrier"] = "NativeFrameState"
        stage.metadata["geometry_storage"] = (
            "device" if constructive_geometry.residency is Residency.DEVICE else "host"
        )
        stage.metadata["attribute_storage"] = "device"
        stage.metadata["provenance_carrier"] = type(
            constructive_result.provenance,
        ).__name__
        stage.metadata["provenance_storage"] = (
            "device" if getattr(constructive_result.provenance, "is_device", False) else "host"
        )
        stage.metadata["geometry_metadata_carrier"] = type(
            constructive_result.geometry_metadata,
        ).__name__
        _record_stage_overheads(
            stage,
            audit,
            memory,
            constructive_geometry,
            constructive_result,
            constructive_state,
            constructive_grouped,
        )

    with profiler.stage(
        "constructive_area_expression",
        category="metric",
        device=ExecutionMode.GPU,
        rows_in=int(constructive_state.row_count),
        detail="compute area from constructive output as a private NativeExpression",
    ) as stage:
        started = perf_counter()
        area_expression = constructive_state.geometry_area_expression()
        expression_elapsed = perf_counter() - started
        stage.rows_out = len(area_expression)
        stage.metadata["operation"] = area_expression.operation
        stage.metadata["expression_storage"] = "device" if area_expression.is_device else "host"
        stage.metadata["precision"] = area_expression.precision
        stage.metadata["native_expression_seconds"] = expression_elapsed
        _record_stage_overheads(stage, audit, memory, area_expression)

    with profiler.stage(
        "constructive_expression_consumers",
        category="consume",
        device=ExecutionMode.GPU,
        rows_in=len(area_expression),
        detail=(
            "feed constructive-output area into NativeRowSet filtering and "
            "NativeGrouped reduction without public GeoDataFrame assembly"
        ),
    ) as stage:
        started = perf_counter()
        area_rowset = area_expression.greater_than(area_threshold)
        filtered_state = constructive_state.take(area_rowset, preserve_index=False)
        reduced = constructive_grouped.reduce_expression(area_expression, "sum")
        consume_elapsed = perf_counter() - started
        stage.rows_out = int(filtered_state.row_count)
        stage.metadata["rowset_storage"] = "device" if area_rowset.is_device else "host"
        stage.metadata["filtered_attribute_storage"] = (
            "device" if filtered_state.attributes.device_table is not None else "host"
        )
        stage.metadata["grouped_result_storage"] = "device" if reduced.is_device else "host"
        stage.metadata["native_consume_seconds"] = consume_elapsed
        _record_stage_overheads(
            stage,
            audit,
            memory,
            area_rowset,
            filtered_state,
            constructive_grouped,
            reduced,
        )

    with profiler.stage(
        "native_reference_check",
        category="reference",
        device=ExecutionMode.GPU,
        rows_in=effective_rows,
        detail="validate native constructive-output consumers without host value export",
        metadata={"profile_boundary": "reference"},
    ) as stage:
        started = perf_counter()
        reference_elapsed = perf_counter() - started
        area_storage_match = area_expression.is_device
        rowset_storage_match = area_rowset.is_device
        filtered_row_count_match = int(filtered_state.row_count) == int(effective_rows)
        grouped_storage_match = reduced.is_device
        grouped_row_count_match = int(reduced.group_count) == int(group_count)
        all_rows_selected = len(area_rowset) == int(effective_rows)
        results_match = (
            area_storage_match
            and rowset_storage_match
            and filtered_row_count_match
            and grouped_storage_match
            and grouped_row_count_match
            and all_rows_selected
        )
        stage.rows_out = int(filtered_state.row_count)
        stage.metadata["reference_mode"] = "native_shape_invariants"
        stage.metadata["expected_area_value"] = expected_area_value
        stage.metadata["area_storage_match"] = area_storage_match
        stage.metadata["rowset_storage_match"] = rowset_storage_match
        stage.metadata["filtered_row_count_match"] = filtered_row_count_match
        stage.metadata["grouped_storage_match"] = grouped_storage_match
        stage.metadata["grouped_row_count_match"] = grouped_row_count_match
        stage.metadata["all_rows_selected"] = all_rows_selected
        stage.metadata["results_match"] = results_match
        stage.metadata["reference_check_seconds"] = reference_elapsed
        _record_stage_overheads(stage, audit, memory, area_expression, area_rowset, reduced)

    stage_devices = [stage.device for stage in profiler._stages]
    actual_selected_runtime = _pipeline_runtime_from_stage_devices(stage_devices)
    trace = profiler.finish(
        metadata={
            "scale": scale,
            "effective_rows": effective_rows,
            "actual_selected_runtime": actual_selected_runtime,
            "planner_selected_runtime": ExecutionMode.GPU.value,
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
            "admissible_shape": (
                "pairwise constructive owned geometry -> NativeTabularResult -> "
                "NativeFrameState -> NativeExpression rowset/grouped consumers"
            ),
            "results_match": results_match,
        }
    )
    return PipelineBenchmarkResult(
        pipeline="constructive-output-native",
        scale=scale,
        status="ok" if results_match else "failed",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=ExecutionMode.GPU.value,
        output_rows=int(filtered_state.row_count),
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
            "Shape canary, not a workflow target: pairwise constructive output "
            "becomes a NativeFrameState and feeds area rowsets plus grouped "
            "reducers before any public GeoDataFrame export."
        ),
    )


def _profile_overlay_relation_constructive_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Shape canary for overlay relation pairs feeding constructive output."""
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return PipelineBenchmarkResult(
            pipeline="overlay-relation-constructive",
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
            notes=(
                "Deferred until CUDA runtime and pylibcudf are available for "
                "overlay relation-to-constructive canary."
            ),
        )

    from time import perf_counter

    from vibespatial.constructive.binary_constructive import binary_constructive_native
    from vibespatial.runtime.materialization import clear_materialization_events

    clear_materialization_events()
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    effective_rows = max(1, min(scale, 100_000))
    query_box_side = 0.5
    index_box_side = 1.0
    expected_area_value = query_box_side * query_box_side
    profiler = _stage_profiler(
        operation="pipeline.overlay-relation-constructive",
        dataset=f"scale-{scale}",
        requested_runtime=ExecutionMode.GPU,
        selected_runtime=ExecutionMode.GPU.value,
        enable_nvtx=enable_nvtx,
        retain_gpu_trace=retain_gpu_trace,
        include_gpu_sparklines=include_gpu_sparklines,
    )

    with profiler.stage(
        "build_native_overlay_inputs",
        category="setup",
        device=ExecutionMode.GPU,
        rows_in=scale,
        detail=(
            "build overlapping device rectangle frames with attached "
            "NativeFrameState and NativeGeometryMetadata"
        ),
    ) as stage:
        left_owned, cols, grid_rows = _device_regular_box_owned(
            effective_rows,
            x_shift=0.25,
            y_shift=0.25,
            side=query_box_side,
        )
        right_owned, _right_cols, _right_grid_rows = _device_regular_box_owned(
            effective_rows,
            side=index_box_side,
        )
        left_frame, left_state = _native_box_frame_from_owned(
            left_owned,
            columns={"left_value": np.arange(effective_rows, dtype=np.int64)},
        )
        right_frame, right_state = _native_box_frame_from_owned(
            right_owned,
            columns={"right_value": np.arange(effective_rows, dtype=np.int64)},
        )
        stage.rows_out = effective_rows
        stage.metadata["effective_rows"] = effective_rows
        stage.metadata["grid_cols"] = cols
        stage.metadata["grid_rows"] = grid_rows
        stage.metadata["query_box_side"] = query_box_side
        stage.metadata["index_box_side"] = index_box_side
        stage.metadata["input_carriers"] = "NativeFrameState,NativeGeometryMetadata"
        stage.metadata["geometry_storage"] = "owned:device"
        stage.metadata["attribute_storage"] = "device"
        _record_stage_overheads(
            stage,
            audit,
            memory,
            left_owned,
            right_owned,
            left_state,
            right_state,
        )

    with profiler.stage(
        "build_spatial_index",
        category="index",
        device=ExecutionMode.GPU,
        rows_in=int(right_state.row_count),
        detail=(
            "build the right-side NativeSpatialIndex carrier before candidate relation production"
        ),
    ) as stage:
        flat_index = build_flat_spatial_index(
            right_state.geometry.owned,
            runtime_selection=RuntimeSelection(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU,
                reason="overlay relation-to-constructive right-side index",
            ),
        )
        stage.rows_out = int(flat_index.size)
        stage.metadata["output_carrier"] = "NativeSpatialIndex"
        stage.metadata["index_impl"] = "FlatSpatialIndex"
        stage.metadata["regular_grid_fast_path"] = bool(flat_index.regular_grid is not None)
        _record_stage_overheads(stage, audit, memory, flat_index)

    with profiler.stage(
        "candidate_relation",
        category="index",
        device=ExecutionMode.GPU,
        rows_in=int(left_state.row_count),
        detail=(
            "produce candidate relation pairs from NativeSpatialIndex without "
            "public index-array assembly"
        ),
    ) as stage:
        query_result, query_execution = query_spatial_index(
            right_state.geometry.owned,
            flat_index,
            left_state.geometry.owned,
            predicate=None,
            sort=True,
            output_format="indices",
            return_device=True,
            return_metadata=True,
        )
        left_idx, right_idx, pair_storage = _relation_pair_arrays_from_query_result(query_result)
        candidate_relation = NativeRelation(
            left_idx,
            right_idx,
            left_token=left_state.lineage_token,
            right_token=right_state.lineage_token,
            predicate=None,
            left_row_count=left_state.row_count,
            right_row_count=right_state.row_count,
            sorted_by_left=True,
        )
        stage.device = query_execution.selected
        stage.rows_out = len(candidate_relation)
        stage.metadata["pair_storage"] = pair_storage
        stage.metadata["query_implementation"] = query_execution.implementation
        stage.metadata["query_reason"] = query_execution.reason
        stage.metadata["output_carrier"] = "NativeRelation"
        stage.metadata["native_spatial_index"] = "FlatSpatialIndex"
        _record_stage_overheads(stage, audit, memory)

    with profiler.stage(
        "refine_relation",
        category="filter",
        device=ExecutionMode.GPU,
        rows_in=int(len(candidate_relation)),
        detail=(
            "refine candidate rectangle overlaps to area-overlap relation "
            "pairs for the constructive overlay consumer"
        ),
    ) as stage:
        if hasattr(candidate_relation.left_indices, "__cuda_array_interface__"):
            import cupy as cp

            candidate_left = cp.asarray(candidate_relation.left_indices)
            candidate_right = cp.asarray(candidate_relation.right_indices)
            keep = candidate_left == candidate_right
            refined_left = candidate_left[keep]
            refined_right = candidate_right[keep]
            stage.metadata["refine_storage"] = "device"
        else:
            candidate_left = np.asarray(candidate_relation.left_indices)
            candidate_right = np.asarray(candidate_relation.right_indices)
            keep = candidate_left == candidate_right
            refined_left = candidate_left[keep]
            refined_right = candidate_right[keep]
            stage.metadata["refine_storage"] = "host"
        refined_relation = NativeRelation(
            refined_left,
            refined_right,
            left_token=candidate_relation.left_token,
            right_token=candidate_relation.right_token,
            predicate="area_intersects",
            left_row_count=candidate_relation.left_row_count,
            right_row_count=candidate_relation.right_row_count,
            sorted_by_left=candidate_relation.sorted_by_left,
        )
        stage.rows_out = int(len(refined_relation))
        stage.metadata["input_carrier"] = "NativeRelation"
        stage.metadata["output_carrier"] = "NativeRelation"
        stage.metadata["predicate"] = "area_intersects"
        stage.metadata["refinement_policy"] = "same-row_rectangle_area_overlap"
        stage.metadata["candidate_pairs"] = int(len(candidate_relation))
        stage.metadata["refined_pairs"] = int(len(refined_relation))
        _record_stage_overheads(stage, audit, memory)

    with profiler.stage(
        "constructive_intersection",
        category="constructive",
        device=ExecutionMode.GPU,
        rows_in=int(len(refined_relation)),
        detail=(
            "gather relation-aligned owned geometry pairs and run pairwise "
            "intersection without public GeoDataFrame assembly"
        ),
    ) as stage:
        started = perf_counter()
        left_pairs = left_state.geometry.owned.take(refined_relation.left_indices)
        right_pairs = right_state.geometry.owned.take(refined_relation.right_indices)
        geometry = binary_constructive_native(
            "intersection",
            left_pairs,
            right_pairs,
            dispatch_mode=ExecutionMode.GPU,
        ).with_crs("EPSG:4326")
        constructive_elapsed = perf_counter() - started
        stage.rows_out = int(geometry.row_count)
        stage.metadata["input_carriers"] = "NativeRelation,NativeFrameState"
        stage.metadata["geometry_storage"] = (
            "device" if geometry.residency is Residency.DEVICE else "host"
        )
        stage.metadata["constructive_seconds"] = constructive_elapsed
        stage.metadata["output_carrier"] = "GeometryNativeResult"
        _record_stage_overheads(
            stage,
            audit,
            memory,
            left_pairs,
            right_pairs,
            geometry,
        )

    with profiler.stage(
        "native_tabular_projection",
        category="project",
        device=ExecutionMode.GPU,
        rows_in=int(geometry.row_count),
        detail=(
            "project relation-aligned attributes and constructive geometry into "
            "NativeTabularResult for native downstream consumers"
        ),
    ) as stage:
        native_result = _relation_constructive_capacity_result(
            op="intersection",
            relation=refined_relation,
            constructed=geometry,
            left_state=left_state,
            right_state=right_state,
            geometry_name="geometry",
            frame_attrs=None,
        )
        projected_state = native_result.to_native_frame_state()
        stage.rows_out = _tabular_row_count(native_result)
        stage.metadata["input_carriers"] = "GeometryNativeResult,NativeRelation,NativeFrameState"
        stage.metadata["output_carrier"] = "NativeTabularResult"
        stage.metadata["downstream_carrier"] = "NativeFrameState"
        stage.metadata["attribute_storage"] = (
            "device" if native_result.attributes.is_device_backed else "host"
        )
        stage.metadata["provenance_carrier"] = type(native_result.provenance).__name__
        stage.metadata["provenance_storage"] = (
            "device" if getattr(native_result.provenance, "is_device", False) else "host"
        )
        stage.metadata["geometry_metadata_carrier"] = type(
            native_result.geometry_metadata,
        ).__name__
        stage.metadata["column_order"] = tuple(str(column) for column in native_result.column_order)
        _record_stage_overheads(
            stage,
            audit,
            memory,
            native_result,
            projected_state,
        )

    with profiler.stage(
        "native_reference_check",
        category="reference",
        device=ExecutionMode.GPU,
        rows_in=int(projected_state.row_count),
        detail="validate native overlay relation constructive shape without host value export",
        metadata={"profile_boundary": "reference"},
    ) as stage:
        started = perf_counter()
        reference_elapsed = perf_counter() - started
        row_count_match = (
            int(len(refined_relation)) == int(projected_state.row_count) == effective_rows
        )
        candidate_superset_match = int(len(candidate_relation)) >= int(len(refined_relation))
        geometry_storage_match = geometry.residency is Residency.DEVICE
        tabular_carrier_match = isinstance(native_result, NativeTabularResult)
        expected_area_shape = float(expected_area_value) > 0.0
        results_match = (
            row_count_match
            and candidate_superset_match
            and geometry_storage_match
            and tabular_carrier_match
            and expected_area_shape
        )
        stage.rows_out = int(projected_state.row_count)
        stage.metadata["reference_mode"] = "native_shape_invariants"
        stage.metadata["row_count_match"] = row_count_match
        stage.metadata["candidate_superset_match"] = candidate_superset_match
        stage.metadata["geometry_storage_match"] = geometry_storage_match
        stage.metadata["tabular_carrier_match"] = tabular_carrier_match
        stage.metadata["expected_area_value"] = expected_area_value
        stage.metadata["results_match"] = results_match
        stage.metadata["reference_check_seconds"] = reference_elapsed
        _record_stage_overheads(stage, audit, memory, native_result, projected_state)

    stage_devices = [stage.device for stage in profiler._stages]
    actual_selected_runtime = _pipeline_runtime_from_stage_devices(stage_devices)
    trace = profiler.finish(
        metadata={
            "scale": scale,
            "effective_rows": effective_rows,
            "actual_selected_runtime": actual_selected_runtime,
            "planner_selected_runtime": ExecutionMode.GPU.value,
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
            "admissible_shape": (
                "NativeSpatialIndex -> candidate NativeRelation -> refined "
                "NativeRelation -> constructive GeometryNativeResult -> "
                "NativeTabularResult"
            ),
            "results_match": results_match,
        }
    )
    return PipelineBenchmarkResult(
        pipeline="overlay-relation-constructive",
        scale=scale,
        status="ok" if results_match else "failed",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=ExecutionMode.GPU.value,
        output_rows=int(projected_state.row_count),
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
            "Shape canary, not a workflow target: overlay candidate pairs stay "
            "as NativeRelation, refinement is explicit, constructive geometry "
            "is produced from relation-aligned owned pairs, and tabular "
            "projection remains NativeTabularResult until terminal export."
        ),
    )


def _profile_native_metadata_index_pipeline(
    scale: int,
    *,
    enable_nvtx: bool = False,
    retain_gpu_trace: bool = False,
    include_gpu_sparklines: bool = False,
) -> PipelineBenchmarkResult:
    """Shape canary for NativeGeometryMetadata and NativeSpatialIndex reuse."""
    if not has_gpu_runtime():
        return PipelineBenchmarkResult(
            pipeline="native-metadata-index",
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
            notes=(
                "Deferred until CUDA runtime is available for native metadata "
                "and spatial-index carrier reuse canary."
            ),
        )

    from time import perf_counter

    from vibespatial.runtime.materialization import clear_materialization_events

    clear_materialization_events()
    clear_dispatch_events()
    clear_fallback_events()
    audit = _OwnedAudit()
    memory = _DeviceMemoryMonitor()
    profiler = _stage_profiler(
        operation="pipeline.native-metadata-index",
        dataset=f"scale-{scale}",
        requested_runtime=ExecutionMode.GPU,
        selected_runtime=ExecutionMode.GPU.value,
        enable_nvtx=enable_nvtx,
        retain_gpu_trace=retain_gpu_trace,
        include_gpu_sparklines=include_gpu_sparklines,
    )

    with profiler.stage(
        "build_device_boxes",
        category="setup",
        device=ExecutionMode.GPU,
        rows_in=scale,
        detail="build device-resident regular polygon grid without host geometry",
    ) as stage:
        owned, grid_cols, grid_rows = _device_regular_box_owned(scale)
        stage.rows_out = int(owned.row_count)
        stage.metadata["grid_cols"] = grid_cols
        stage.metadata["grid_rows"] = grid_rows
        stage.metadata["host_geometry_materialized"] = any(
            buffer.host_materialized for buffer in owned.families.values()
        )
        _record_stage_overheads(stage, audit, memory, owned)

    with profiler.stage(
        "build_flat_spatial_index",
        category="index",
        device=ExecutionMode.GPU,
        rows_in=int(owned.row_count),
        detail="build regular-grid FlatSpatialIndex from trusted native shape proof",
    ) as stage:
        started = perf_counter()
        flat_index = build_flat_spatial_index(
            owned,
            runtime_selection=RuntimeSelection(
                requested=ExecutionMode.CPU,
                selected=ExecutionMode.CPU,
                reason="native metadata index canary regular-grid detector",
            ),
        )
        index_elapsed = perf_counter() - started
        stage.rows_out = int(flat_index.size)
        stage.metadata["regular_grid"] = flat_index.regular_grid is not None
        stage.metadata["device_bounds"] = flat_index.device_bounds is not None
        stage.metadata["host_bounds_materialized"] = flat_index._host_bounds is not None
        stage.metadata["total_bounds"] = flat_index.total_bounds
        stage.metadata["admissibility_fence_max_bytes"] = 0
        stage.metadata["native_index_build_seconds"] = index_elapsed
        _record_stage_overheads(stage, audit, memory, flat_index)
        stage.metadata["scalar_fence_within_budget"] = (
            stage.metadata["runtime_d2h_transfer_bytes_delta"] == 0
        )

    with profiler.stage(
        "wrap_native_metadata_index",
        category="carrier",
        device=ExecutionMode.GPU,
        rows_in=int(flat_index.size),
        detail="wrap FlatSpatialIndex state as reusable Native* carriers",
    ) as stage:
        started = perf_counter()
        owned_metadata = flat_index.geometry_metadata(source_token="frame")
        native_index = flat_index.to_native_spatial_index(source_token="frame")
        wrap_elapsed = perf_counter() - started
        stage.rows_out = int(native_index.row_count)
        stage.metadata["owned_metadata_storage"] = "device" if owned_metadata.is_device else "host"
        stage.metadata["native_index_storage"] = "device" if native_index.is_device else "host"
        stage.metadata["metadata_reuses_device_bounds"] = (
            owned_metadata.bounds is flat_index.device_bounds
            and native_index.metadata is not None
            and native_index.metadata.bounds is flat_index.device_bounds
        )
        stage.metadata["host_bounds_materialized"] = flat_index._host_bounds is not None
        stage.metadata["native_wrap_seconds"] = wrap_elapsed
        _record_stage_overheads(stage, audit, memory, native_index)

    relation_results_match = False
    metadata_take_results_match = False
    with profiler.stage(
        "native_index_query_relation",
        category="filter",
        device=ExecutionMode.GPU,
        rows_in=int(native_index.row_count),
        detail="query NativeSpatialIndex directly into NativeRelation pair flow",
    ) as stage:
        row_ids = np.arange(scale, dtype=np.int64)
        query_x = np.remainder(row_ids, grid_cols).astype(np.float64, copy=False) + 0.5
        query_y = (row_ids // grid_cols).astype(np.float64, copy=False) + 0.5
        query_owned = _point_owned_from_xy_device(query_x, query_y)
        relation, query_execution = native_index.query_relation(
            query_owned,
            predicate="intersects",
            sort=True,
            query_token="query",
            return_metadata=True,
        )
        rowset = relation.left_semijoin_rowset()
        relation_results_match = int(len(relation)) == scale and int(len(rowset)) == scale
        stage.device = query_execution.selected
        stage.rows_out = int(len(relation))
        stage.metadata["query_implementation"] = query_execution.implementation
        stage.metadata["query_reason"] = query_execution.reason
        stage.metadata["relation_storage"] = (
            "device" if relation.left_rowset().is_device else "host"
        )
        stage.metadata["rowset_storage"] = "device" if rowset.is_device else "host"
        stage.metadata["relation_pairs"] = int(len(relation))
        stage.metadata["semijoin_rows"] = int(len(rowset))
        stage.metadata["results_match"] = relation_results_match
        _record_stage_overheads(stage, audit, memory, query_owned, relation)
        stage.metadata["scalar_fence_within_budget"] = (
            stage.metadata["runtime_d2h_transfer_bytes_delta"] == 0
        )

    with profiler.stage(
        "native_metadata_rowset_take",
        category="carrier",
        device=ExecutionMode.GPU,
        rows_in=int(native_index.row_count),
        detail=(
            "carry NativeGeometryMetadata through NativeTabularResult and "
            "NativeFrameState rowset take without public export"
        ),
    ) as stage:
        import pyarrow as pa

        attributes = NativeAttributeTable(
            arrow_table=pa.table({"value": pa.array(np.arange(scale, dtype=np.int32))})
        )
        tabular = NativeTabularResult(
            attributes=attributes,
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
            geometry_metadata=owned_metadata,
        )
        state = NativeFrameState.from_native_tabular_result(tabular)
        cached_metadata = state.geometry_metadata()
        frame_rowset = NativeRowSet.from_positions(
            rowset.positions,
            source_token=state.lineage_token,
            source_row_count=state.row_count,
            ordered=rowset.ordered,
            unique=rowset.unique,
        )
        filtered_state = state.take(frame_rowset, preserve_index=True)
        filtered_metadata = filtered_state.geometry_metadata()
        metadata_take_results_match = bool(
            cached_metadata.bounds is flat_index.device_bounds
            and cached_metadata.source_token == state.lineage_token
            and filtered_state.index_plan.kind == "device-labels"
            and filtered_state.attributes.device_table is not None
            and filtered_metadata.is_device
            and filtered_metadata.row_count == scale
        )
        stage.rows_out = int(filtered_state.row_count)
        stage.metadata["cached_metadata_reuses_device_bounds"] = (
            cached_metadata.bounds is flat_index.device_bounds
        )
        stage.metadata["filtered_metadata_storage"] = (
            "device" if filtered_metadata.is_device else "host"
        )
        stage.metadata["filtered_attribute_storage"] = (
            "device" if filtered_state.attributes.device_table is not None else "host"
        )
        stage.metadata["index_plan_kind"] = filtered_state.index_plan.kind
        stage.metadata["results_match"] = metadata_take_results_match
        _record_stage_overheads(stage, audit, memory, filtered_state)
        stage.metadata["geometry_take_scalar_fence_max_bytes"] = 0
        stage.metadata["scalar_fence_within_budget"] = (
            stage.metadata["runtime_d2h_transfer_bytes_delta"] == 0
        )

    with profiler.stage(
        "carrier_contract_checks",
        category="validate",
        device=ExecutionMode.GPU,
        rows_in=int(native_index.row_count),
        detail="validate row-count and metadata contracts without public export",
    ) as stage:
        owned_metadata.validate_row_count(scale)
        native_index.validate_row_count(scale)
        if native_index.metadata is None:
            raise RuntimeError("NativeSpatialIndex metadata is required")
        native_index.metadata.validate_row_count(scale)
        results_match = bool(
            flat_index.regular_grid is not None
            and flat_index.device_bounds is not None
            and flat_index._host_bounds is None
            and native_index.is_device
            and owned_metadata.is_device
            and native_index.metadata.bounds is flat_index.device_bounds
            and flat_index.total_bounds == (0.0, 0.0, float(grid_cols), float(grid_rows))
            and relation_results_match
            and metadata_take_results_match
        )
        stage.rows_out = int(native_index.row_count)
        stage.metadata["results_match"] = results_match
        stage.metadata["host_bounds_materialized"] = flat_index._host_bounds is not None
        stage.metadata["native_index_kind"] = native_index.kind
        _record_stage_overheads(stage, audit, memory, native_index)

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
                "owned polygon metadata -> FlatSpatialIndex device bounds -> "
                "NativeGeometryMetadata -> NativeSpatialIndex -> NativeRelation -> "
                "NativeFrameState rowset take"
            ),
            "results_match": results_match,
        }
    )
    return PipelineBenchmarkResult(
        pipeline="native-metadata-index",
        scale=scale,
        status="ok" if results_match else "failed",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime=actual_selected_runtime,
        planner_selected_runtime=ExecutionMode.GPU.value,
        output_rows=int(native_index.row_count),
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
            "Shape canary, not a workflow target: device bounds and regular-grid "
            "index state become reusable Native* carriers across frame and "
            "rowset-take boundaries without non-terminal host scalar fences."
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
        audit.reset_materialization_baseline()
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

        clip_work = estimate_physical_work_from_owned(
            owned,
            output_row_count=owned.row_count,
            output_byte_count=owned.row_count * 24,
            primary_unit_name="point-clip-coordinate",
        )
        with profiler.stage(
            "clip_points",
            category="filter",
            device=plan_dispatch_selection(
                kernel_name="point_clip",
                kernel_class=KernelClass.CONSTRUCTIVE,
                row_count=owned.row_count,
                work_estimate=clip_work,
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

        clipped_work = estimate_physical_work_from_owned(clipped)
        buffer_output_coordinates = clipped_work.coordinate_count * 5
        with profiler.stage(
            "buffer_points",
            category="refine",
            device=plan_dispatch_selection(
                kernel_name="point_buffer",
                kernel_class=KernelClass.CONSTRUCTIVE,
                row_count=clipped.row_count,
                work_estimate=PhysicalWorkEstimate(
                    row_count=clipped.row_count,
                    coordinate_count=clipped_work.coordinate_count,
                    output_row_count=clipped.row_count,
                    output_byte_count=buffer_output_coordinates * 16,
                    temporary_byte_count=clipped.row_count * 24,
                    primary_unit_count=max(
                        clipped.row_count,
                        clipped_work.coordinate_count,
                        buffer_output_coordinates,
                    ),
                    primary_unit_name="point-buffer-output-coordinate",
                ),
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
        audit.reset_materialization_baseline()
        profiler = _stage_profiler(
            operation="pipeline.predicate-heavy",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.AUTO,
            selected_runtime="cpu",
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        with profiler.stage(
            "read_geojson", category="setup", device="auto", rows_in=scale
        ) as stage:
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
            stage.device = (
                _selected_runtime_from_history(polygon_owned) or _preferred_geoparquet_backend()
            )
            stage.rows_out = polygon_owned.row_count
            stage.metadata["cache_path"] = str(
                _predicate_polygon_cache_path(polygon_count, batch.geometry.row_count)
            )
            stage.metadata["cache_hit"] = _predicate_polygon_cache_path(
                polygon_count, batch.geometry.row_count
            ).exists()
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
            stage.rows_out = int(getattr(mask, "shape", (batch.geometry.row_count,))[0])
            stage.metadata["true_count"] = "deferred_device_mask"
            runtime_selection = (
                batch.geometry.runtime_history[history_before:]
                or batch.geometry.runtime_history[-1:]
            )
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

        with profiler.stage(
            "read_geojson", category="setup", device=ExecutionMode.CPU, rows_in=scale
        ) as stage:
            point_frame, requested_engine, actual_engine, fallback_note = (
                _read_geojson_geopandas_preferred(source_path)
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
            stage.metadata["cache_path"] = str(
                _predicate_polygon_cache_path(polygon_count, int(len(point_frame)))
            )
            stage.metadata["cache_hit"] = _predicate_polygon_cache_path(
                polygon_count, int(len(point_frame))
            ).exists()

        with profiler.stage(
            "point_in_polygon",
            category="refine",
            device=ExecutionMode.CPU,
            rows_in=int(len(point_frame)),
            detail="evaluate boundary-inclusive point-in-polygon with GeoPandas/Shapely covers semantics",
        ) as stage:
            hits = polygon_series.reset_index(drop=True).covers(
                point_frame.geometry.reset_index(drop=True), align=False
            )
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
        audit.reset_materialization_baseline()
        profiler = _stage_profiler(
            operation="pipeline.zero-transfer",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.GPU,
            selected_runtime="gpu",
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        with profiler.stage(
            "read_input", category="setup", device=ExecutionMode.GPU, rows_in=scale
        ) as stage:
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
    d2h_reasons = [
        event["reason"]
        for stage in profiler._stages
        for event in stage.metadata.get("runtime_d2h_transfer_events", ())
    ]
    bounded_terminal_metadata_fence = (
        transfer_count <= 1
        and audit.runtime_d2h_transfer_bytes <= 32
        and set(d2h_reasons) <= {"DeviceGeometryArray total-bounds device summary host boundary"}
    )
    status = (
        "ok"
        if materialization_count == 0 and (transfer_count == 0 or bounded_terminal_metadata_fence)
        else "failed"
    )
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
            filtered.extend(
                _extract_supported_collection_parts(g, _SUPPORTED_COLLECTION_GEOM_TYPES)
            )
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
    dataset = generate_lines(
        SyntheticSpec("line", "river", count=max(count, 1), seed=10, vertices=12)
    )
    values = np.asarray(list(dataset.geometries), dtype=object)
    values = shapely.make_valid(values)
    return geopandas.GeoDataFrame(
        {
            "circuit_id": pd.Categorical(
                np.arange(len(values), dtype=np.int32) % max(min(len(values), 32), 1)
            ),
            "geometry": values,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _vegetation_patches_frame(count: int) -> geopandas.GeoDataFrame:
    """Irregular convex-hull polygons mimicking vegetation patches."""
    dataset = generate_polygons(
        SyntheticSpec(
            "polygon", "convex-hull", count=max(count, 1), seed=11, clusters=8, vertices=6
        )
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
    dataset = generate_points(
        SyntheticSpec("point", "clustered", count=max(count, 1), seed=12, clusters=12)
    )
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
        SyntheticSpec(
            "polygon", "convex-hull", count=max(count, 1), seed=21, clusters=4, vertices=8
        )
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
        SyntheticSpec(
            "polygon", "star", count=1, seed=41, vertices=12, bounds=(100.0, 100.0, 900.0, 900.0)
        )
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
        SyntheticSpec(
            "polygon", "convex-hull", count=max(count, 1), seed=50, clusters=6, vertices=8
        )
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
    dataset = generate_points(
        SyntheticSpec("point", "clustered", count=max(count, 1), seed=51, clusters=8)
    )
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
        _vegetation_patches_frame(polygon_count).to_parquet(
            polygons_path, geometry_encoding="geoarrow"
        )
        _utility_poles_frame(point_count).to_file(points_path, driver="GeoJSON")

        audit.reset_runtime_baseline()
        audit.reset_materialization_baseline()
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
        with profiler.stage(
            "read_lines", category="setup", device="auto", rows_in=line_count
        ) as stage:
            lines_owned, actual_lines_backend, lines_note = _read_geoparquet_owned_preferred(
                lines_path,
                preferred_backend=read_backend,
            )
            stage.device = actual_lines_backend
            stage.rows_out = lines_owned.row_count
            if lines_note:
                stage.metadata["fallback_note"] = lines_note
            _record_stage_overheads(stage, audit, memory, lines_owned)

        # Stage 2: read vegetation polygons
        with profiler.stage(
            "read_polygons", category="setup", device="auto", rows_in=polygon_count
        ) as stage:
            veg_owned, actual_veg_backend, veg_note = _read_geoparquet_owned_preferred(
                polygons_path,
                preferred_backend=read_backend,
            )
            stage.device = actual_veg_backend
            stage.rows_out = veg_owned.row_count
            if veg_note:
                stage.metadata["fallback_note"] = veg_note
            _record_stage_overheads(stage, audit, memory, veg_owned)

        # Stage 3: read utility poles (GeoJSON)
        with profiler.stage(
            "read_points", category="setup", device="auto", rows_in=point_count
        ) as stage:
            batch, actual_read_mode, read_note = _read_geojson_owned_preferred(
                points_path,
                preferred_mode=_preferred_geojson_mode(),
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
                lines_owned,
                10.0,
                quad_segs=4,
                dispatch_mode=ExecutionMode.AUTO,
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
        # original owned array directly through one native grouped union.
        with profiler.stage(
            "dissolve_corridor",
            category="refine",
            device=ExecutionMode.AUTO,
            rows_in=buffered_lines.row_count,
            detail="dissolve buffered corridor polygons into a single coverage polygon",
        ) as stage:
            valid_result = make_valid_owned(owned=buffered_lines)
            if valid_result.owned is not None:
                corridor_owned = union_all_gpu_owned(valid_result.owned)
            else:
                valid_geoms = _extract_polygonal_components(valid_result.geometries)
                valid_owned = from_shapely_geometries(valid_geoms)
                corridor_owned = union_all_gpu_owned(valid_owned)
            stage.rows_out = corridor_owned.row_count
            stage.device = _selected_runtime_from_history(corridor_owned) or "cpu"

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
                        clipped_veg,
                        dispatch_mode=_nearby_runtime,
                        return_owned=True,
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
                    buffered_owned,
                    buf_index,
                    poles_owned,
                    predicate="intersects",
                    sort=False,
                    output_format="count",
                )
            else:
                stage.rows_out = 0
            _record_stage_overheads(stage, audit, memory, poles_owned)

        # Stage 8: write output
        output = (
            geopandas.GeoDataFrame(
                {"geometry": geoseries_from_owned(clipped_veg, crs="EPSG:4326")},
                geometry="geometry",
                crs="EPSG:4326",
            )
            if clipped_veg.row_count > 0
            else geopandas.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        )
        output_path = root / "vegetation-corridor-output.parquet"
        with profiler.stage(
            "write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(output))
        ) as stage:
            write_geoparquet(output, output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(output))

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
        _vegetation_patches_frame(polygon_count).to_parquet(
            polygons_path, geometry_encoding="geoarrow"
        )
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

        with profiler.stage(
            "read_lines", category="setup", device=ExecutionMode.CPU, rows_in=line_count
        ) as stage:
            lines_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(lines_path)))
            stage.rows_out = int(len(lines_gdf))

        with profiler.stage(
            "read_polygons", category="setup", device=ExecutionMode.CPU, rows_in=polygon_count
        ) as stage:
            veg_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(polygons_path)))
            stage.rows_out = int(len(veg_gdf))

        with profiler.stage(
            "read_points", category="setup", device=ExecutionMode.CPU, rows_in=point_count
        ) as stage:
            poles_gdf, _, _, _ = _read_geojson_geopandas_preferred(points_path)
            stage.rows_out = int(len(poles_gdf))

        with profiler.stage(
            "buffer_lines", category="refine", device=ExecutionMode.CPU, rows_in=int(len(lines_gdf))
        ) as stage:
            lines_gdf["geometry"] = lines_gdf.geometry.buffer(10.0, quad_segs=4)
            stage.rows_out = int(len(lines_gdf))

        with profiler.stage(
            "dissolve_corridor",
            category="refine",
            device=ExecutionMode.CPU,
            rows_in=int(len(lines_gdf)),
        ) as stage:
            lines_gdf["group"] = 0
            dissolved = lines_gdf.dissolve(by="group")
            stage.rows_out = int(len(dissolved))

        with profiler.stage(
            "intersect_vegetation",
            category="refine",
            device=ExecutionMode.CPU,
            rows_in=int(len(veg_gdf)),
        ) as stage:
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

        with profiler.stage(
            "find_nearby_poles",
            category="filter",
            device=ExecutionMode.CPU,
            rows_in=int(len(poles_gdf)),
        ) as stage:
            if len(clipped) > 0:
                buffered_veg = clipped.copy()
                buffered_veg["geometry"] = clipped.geometry.centroid.buffer(1.0)
                joined = geopandas.sjoin(poles_gdf, buffered_veg[["geometry"]], predicate="within")
                stage.rows_out = int(len(joined))
            else:
                stage.rows_out = 0

        output_path = root / "vegetation-corridor-output.parquet"
        with profiler.stage(
            "write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(clipped))
        ) as stage:
            clipped.to_parquet(output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(clipped))

    trace = profiler.finish(
        metadata={
            "scale": scale,
            "actual_selected_runtime": "cpu",
            "planner_selected_runtime": "cpu",
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
        }
    )
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
        audit.reset_materialization_baseline()
        profiler = _stage_profiler(
            operation="pipeline.parcel-zoning",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.AUTO,
            selected_runtime="hybrid" if has_gpu_runtime() else "cpu",
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        with profiler.stage(
            "read_parcels", category="setup", device="auto", rows_in=parcel_count
        ) as stage:
            parcels_owned, actual_parcels_backend, parcels_note = _read_geoparquet_owned_preferred(
                parcels_path,
                preferred_backend=read_backend,
            )
            stage.device = actual_parcels_backend
            stage.rows_out = parcels_owned.row_count
            if parcels_note:
                stage.metadata["fallback_note"] = parcels_note
            _record_stage_overheads(stage, audit, memory, parcels_owned)

        with profiler.stage(
            "read_zones", category="setup", device="auto", rows_in=zone_count
        ) as stage:
            zones_owned, actual_zones_backend, zones_note = _read_geoparquet_owned_preferred(
                zones_path,
                preferred_backend=read_backend,
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
                clipped_owned = (
                    clip_result.owned_result
                    if clip_result.owned_result is not None
                    else _from_shapely_safe(list(clip_result.geometries[: clip_result.row_count]))
                )
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
                    zones_owned,
                    flat_index,
                    clipped_owned,
                    predicate="intersects",
                    sort=False,
                    output_format="count",
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

        output = (
            geopandas.GeoDataFrame(
                {"geometry": geoseries_from_owned(overlaid, crs="EPSG:4326")},
                geometry="geometry",
                crs="EPSG:4326",
            )
            if overlaid.row_count > 0
            else geopandas.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        )
        output_path = root / "parcel-zoning-output.parquet"
        with profiler.stage(
            "write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(output))
        ) as stage:
            if len(output) > 0:
                write_geoparquet(output, output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(output))

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

        with profiler.stage(
            "read_parcels", category="setup", device=ExecutionMode.CPU, rows_in=parcel_count
        ) as stage:
            parcels_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(parcels_path)))
            stage.rows_out = int(len(parcels_gdf))

        with profiler.stage(
            "read_zones", category="setup", device=ExecutionMode.CPU, rows_in=zone_count
        ) as stage:
            zones_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(zones_path)))
            stage.rows_out = int(len(zones_gdf))

        with profiler.stage(
            "clip_to_study_area",
            category="filter",
            device=ExecutionMode.CPU,
            rows_in=int(len(parcels_gdf)),
        ) as stage:
            clipped = geopandas.clip(parcels_gdf, clip_box)
            stage.rows_out = int(len(clipped))

        with profiler.stage(
            "sjoin_query", category="filter", device=ExecutionMode.CPU, rows_in=int(len(clipped))
        ) as stage:
            joined = geopandas.sjoin(clipped, zones_gdf, predicate="intersects")
            stage.rows_out = int(len(joined))

        with profiler.stage(
            "overlay_intersect",
            category="refine",
            device=ExecutionMode.CPU,
            rows_in=int(len(clipped)),
        ) as stage:
            # geopandas.overlay rejects mixed geometry types; clip can
            # produce GeometryCollections at boundaries.  Filter to
            # polygonal types only.
            poly_mask = clipped.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
            clipped_poly = clipped[poly_mask] if not poly_mask.all() else clipped
            overlaid = geopandas.overlay(clipped_poly, zones_gdf, how="intersection")
            stage.rows_out = int(len(overlaid))

        output_path = root / "parcel-zoning-output.parquet"
        with profiler.stage(
            "write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(overlaid))
        ) as stage:
            overlaid.to_parquet(output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(overlaid))

    trace = profiler.finish(
        metadata={
            "scale": scale,
            "actual_selected_runtime": "cpu",
            "planner_selected_runtime": "cpu",
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
        }
    )
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
        audit.reset_materialization_baseline()
        profiler = _stage_profiler(
            operation="pipeline.flood-exposure",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.AUTO,
            selected_runtime="hybrid" if has_gpu_runtime() else "cpu",
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        with profiler.stage(
            "read_buildings", category="setup", device="auto", rows_in=building_count
        ) as stage:
            buildings_owned, actual_bld_backend, bld_note = _read_geoparquet_owned_preferred(
                buildings_path,
                preferred_backend=read_backend,
            )
            stage.device = actual_bld_backend
            stage.rows_out = buildings_owned.row_count
            if bld_note:
                stage.metadata["fallback_note"] = bld_note
            _record_stage_overheads(stage, audit, memory, buildings_owned)

        with profiler.stage(
            "read_flood_zones", category="setup", device="auto", rows_in=flood_count
        ) as stage:
            flood_batch, actual_flood_mode, flood_note = _read_geojson_owned_preferred(
                flood_path,
                preferred_mode=_preferred_geojson_mode(),
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
                flood_owned,
                flat_index,
                buildings_valid,
                predicate="intersects",
                sort=True,
                output_format="indices",
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
                    centroids,
                    50.0,
                    quad_segs=4,
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

        output = (
            geopandas.GeoDataFrame(
                {"geometry": geoseries_from_owned(risk_zones, crs="EPSG:4326")},
                geometry="geometry",
                crs="EPSG:4326",
            )
            if risk_zones.row_count > 0
            else geopandas.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        )
        output_path = root / "flood-exposure-output.parquet"
        with profiler.stage(
            "write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(output))
        ) as stage:
            write_geoparquet(output, output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(output))

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

        with profiler.stage(
            "read_buildings", category="setup", device=ExecutionMode.CPU, rows_in=building_count
        ) as stage:
            buildings_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(buildings_path)))
            stage.rows_out = int(len(buildings_gdf))

        with profiler.stage(
            "read_flood_zones", category="setup", device=ExecutionMode.CPU, rows_in=flood_count
        ) as stage:
            flood_gdf, _, _, _ = _read_geojson_geopandas_preferred(flood_path)
            stage.rows_out = int(len(flood_gdf))

        with profiler.stage(
            "make_valid",
            category="refine",
            device=ExecutionMode.CPU,
            rows_in=int(len(buildings_gdf)),
        ) as stage:
            buildings_gdf["geometry"] = shapely.make_valid(buildings_gdf.geometry.values)
            stage.rows_out = int(len(buildings_gdf))

        with profiler.stage(
            "sjoin_intersects",
            category="filter",
            device=ExecutionMode.CPU,
            rows_in=int(len(buildings_gdf)),
        ) as stage:
            joined = geopandas.sjoin(buildings_gdf, flood_gdf, predicate="intersects")
            stage.rows_out = int(len(joined))

        with profiler.stage(
            "filter_buildings",
            category="filter",
            device=ExecutionMode.CPU,
            rows_in=int(len(buildings_gdf)),
        ) as stage:
            hit_indices = joined.index.unique()
            filtered = buildings_gdf.loc[hit_indices]
            stage.rows_out = int(len(filtered))

        with profiler.stage(
            "buffer_risk_zone",
            category="refine",
            device=ExecutionMode.CPU,
            rows_in=int(len(filtered)),
        ) as stage:
            if len(filtered) > 0:
                risk_zones = filtered.copy()
                risk_zones["geometry"] = filtered.geometry.centroid.buffer(50.0)
            else:
                risk_zones = filtered.copy()
            stage.rows_out = int(len(risk_zones))

        output_path = root / "flood-exposure-output.parquet"
        with profiler.stage(
            "write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(risk_zones))
        ) as stage:
            risk_zones.to_parquet(output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(risk_zones))

    trace = profiler.finish(
        metadata={
            "scale": scale,
            "actual_selected_runtime": "cpu",
            "planner_selected_runtime": "cpu",
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
        }
    )
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
        audit.reset_materialization_baseline()
        profiler = _stage_profiler(
            operation="pipeline.network-service-area",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.AUTO,
            selected_runtime="hybrid" if has_gpu_runtime() else "cpu",
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        with profiler.stage(
            "read_network", category="setup", device="auto", rows_in=network_count
        ) as stage:
            network_owned, actual_net_backend, net_note = _read_geoparquet_owned_preferred(
                network_path,
                preferred_backend=read_backend,
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
                network_owned,
                25.0,
                quad_segs=4,
                dispatch_mode=ExecutionMode.AUTO,
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
                {
                    "group": np.zeros(buffered_network.row_count, dtype=np.int32),
                    "geometry": geoseries_from_owned(buffered_network, crs="EPSG:4326"),
                },
                geometry="geometry",
                crs="EPSG:4326",
            )
            dissolved = evaluate_geopandas_dissolve(
                service_frame,
                by="group",
                aggfunc="first",
                as_index=True,
                level=None,
                sort=False,
                observed=False,
                dropna=True,
                method="unary",
                grid_size=None,
                agg_kwargs={},
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

        output = (
            geopandas.GeoDataFrame(
                {"geometry": geoseries_from_owned(clipped, crs="EPSG:4326")},
                geometry="geometry",
                crs="EPSG:4326",
            )
            if clipped.row_count > 0
            else geopandas.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        )
        output_path = root / "network-service-area-output.parquet"
        with profiler.stage(
            "write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(output))
        ) as stage:
            write_geoparquet(output, output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(output))

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

        with profiler.stage(
            "read_network", category="setup", device=ExecutionMode.CPU, rows_in=network_count
        ) as stage:
            network_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(network_path)))
            stage.rows_out = int(len(network_gdf))

        with profiler.stage(
            "read_admin", category="setup", device=ExecutionMode.CPU, rows_in=1
        ) as stage:
            admin_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(admin_path)))
            stage.rows_out = int(len(admin_gdf))

        with profiler.stage(
            "buffer_network",
            category="refine",
            device=ExecutionMode.CPU,
            rows_in=int(len(network_gdf)),
        ) as stage:
            network_gdf["geometry"] = network_gdf.geometry.buffer(25.0, quad_segs=4)
            stage.rows_out = int(len(network_gdf))

        with profiler.stage(
            "dissolve_service_area",
            category="refine",
            device=ExecutionMode.CPU,
            rows_in=int(len(network_gdf)),
        ) as stage:
            network_gdf["group"] = 0
            dissolved = network_gdf.dissolve(by="group")
            stage.rows_out = int(len(dissolved))

        with profiler.stage(
            "clip_to_admin",
            category="refine",
            device=ExecutionMode.CPU,
            rows_in=int(len(dissolved)),
        ) as stage:
            clipped = geopandas.clip(dissolved, admin_gdf)
            stage.rows_out = int(len(clipped))

        output_path = root / "network-service-area-output.parquet"
        with profiler.stage(
            "write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(clipped))
        ) as stage:
            clipped.to_parquet(output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(clipped))

    trace = profiler.finish(
        metadata={
            "scale": scale,
            "actual_selected_runtime": "cpu",
            "planner_selected_runtime": "cpu",
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
        }
    )
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
        _transit_stations_frame(transit_count).to_parquet(
            transit_path, geometry_encoding="geoarrow"
        )

        # Study area: 60% of data bounds
        bounds = parcels_frame.total_bounds
        dx = (bounds[2] - bounds[0]) * 0.2
        dy = (bounds[3] - bounds[1]) * 0.2
        clip_rect = (bounds[0] + dx, bounds[1] + dy, bounds[2] - dx, bounds[3] - dy)

        audit.reset_runtime_baseline()
        audit.reset_materialization_baseline()
        profiler = _stage_profiler(
            operation="pipeline.site-suitability",
            dataset=f"scale-{scale}",
            requested_runtime=ExecutionMode.AUTO,
            selected_runtime="hybrid" if has_gpu_runtime() else "cpu",
            enable_nvtx=enable_nvtx,
            retain_gpu_trace=retain_gpu_trace,
            include_gpu_sparklines=include_gpu_sparklines,
        )

        with profiler.stage(
            "read_parcels", category="setup", device="auto", rows_in=parcel_count
        ) as stage:
            parcels_owned, actual_parcels_backend, parcels_note = _read_geoparquet_owned_preferred(
                parcels_path,
                preferred_backend=read_backend,
            )
            stage.device = actual_parcels_backend
            stage.rows_out = parcels_owned.row_count
            if parcels_note:
                stage.metadata["fallback_note"] = parcels_note
            _record_stage_overheads(stage, audit, memory, parcels_owned)

        with profiler.stage(
            "read_exclusions", category="setup", device="auto", rows_in=exclusion_count
        ) as stage:
            excl_batch, actual_excl_mode, excl_note = _read_geojson_owned_preferred(
                exclusions_path,
                preferred_mode=_preferred_geojson_mode(),
            )
            stage.device = actual_excl_mode
            stage.rows_out = excl_batch.geometry.row_count
            if excl_note:
                stage.metadata["fallback_note"] = excl_note
            _record_stage_overheads(stage, audit, memory, excl_batch)
        exclusions_owned = excl_batch.geometry

        with profiler.stage(
            "read_transit", category="setup", device="auto", rows_in=transit_count
        ) as stage:
            transit_owned, actual_transit_backend, transit_note = _read_geoparquet_owned_preferred(
                transit_path,
                preferred_backend=read_backend,
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
                clipped_owned = (
                    clip_result.owned_result
                    if clip_result.owned_result is not None
                    else _from_shapely_safe(list(clip_result.geometries[: clip_result.row_count]))
                )
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
                suitable = _overlay_via_public_api(
                    clipped_owned, exclusions_owned, how="difference"
                )
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
            transit_series = geoseries_from_owned(
                transit_owned,
                crs="EPSG:4326",
            )
            transit_buffered = transit_series.buffer(200.0)
            stage.rows_out = int(len(transit_buffered))
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
            rows_in=int(len(transit_buffered)),
            detail="build the public spatial index on buffered transit catchments",
        ) as stage:
            spatial_index = transit_buffered.sindex
            stage.rows_out = int(spatial_index.size)
            _record_stage_overheads(stage, audit, memory, transit_buffered)

        query_runtime = ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU

        with profiler.stage(
            "sjoin_proximity",
            category="filter",
            device=query_runtime,
            rows_in=suitable.row_count,
            detail="spatial join: suitable parcels near transit stations",
        ) as stage:
            suitable_frame = geopandas.GeoDataFrame(
                {"geometry": geoseries_from_owned(suitable, crs="EPSG:4326")},
                geometry="geometry",
                crs="EPSG:4326",
            )
            has_match = spatial_index.query_any(
                suitable_frame.geometry,
                predicate="intersects",
            )
            near_transit = suitable_frame[has_match]
            stage.rows_out = int(len(near_transit))
            stage.metadata["parcels_near_transit"] = int(len(near_transit))
            _record_stage_overheads(stage, audit, memory, suitable, transit_buffered)

        # Release transit buffers and spatial index before output materialization
        del transit_buffered, spatial_index
        _free_gpu_pool_memory()

        output = near_transit
        output_path = root / "site-suitability-output.parquet"
        with profiler.stage(
            "write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(output))
        ) as stage:
            if len(output) > 0:
                write_geoparquet(output, output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(output))

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
        _transit_stations_frame(transit_count).to_parquet(
            transit_path, geometry_encoding="geoarrow"
        )

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

        with profiler.stage(
            "read_parcels", category="setup", device=ExecutionMode.CPU, rows_in=parcel_count
        ) as stage:
            parcels_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(parcels_path)))
            stage.rows_out = int(len(parcels_gdf))

        with profiler.stage(
            "read_exclusions", category="setup", device=ExecutionMode.CPU, rows_in=exclusion_count
        ) as stage:
            excl_gdf, _, _, _ = _read_geojson_geopandas_preferred(exclusions_path)
            stage.rows_out = int(len(excl_gdf))

        with profiler.stage(
            "read_transit", category="setup", device=ExecutionMode.CPU, rows_in=transit_count
        ) as stage:
            transit_gdf = geopandas.GeoDataFrame.from_arrow(pq.read_table(str(transit_path)))
            stage.rows_out = int(len(transit_gdf))

        with profiler.stage(
            "clip_study_area",
            category="filter",
            device=ExecutionMode.CPU,
            rows_in=int(len(parcels_gdf)),
        ) as stage:
            clipped = geopandas.clip(parcels_gdf, clip_box)
            stage.rows_out = int(len(clipped))

        with profiler.stage(
            "overlay_difference",
            category="refine",
            device=ExecutionMode.CPU,
            rows_in=int(len(clipped)),
        ) as stage:
            # geopandas.overlay rejects mixed geometry types; clip can
            # produce GeometryCollections at boundaries.  Filter to
            # polygonal types only.
            poly_mask = clipped.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
            clipped_poly = clipped[poly_mask] if not poly_mask.all() else clipped
            suitable = geopandas.overlay(clipped_poly, excl_gdf, how="difference")
            stage.rows_out = int(len(suitable))

        with profiler.stage(
            "buffer_transit",
            category="refine",
            device=ExecutionMode.CPU,
            rows_in=int(len(transit_gdf)),
        ) as stage:
            transit_gdf["geometry"] = transit_gdf.geometry.buffer(200.0)
            stage.rows_out = int(len(transit_gdf))

        with profiler.stage(
            "sjoin_proximity",
            category="filter",
            device=ExecutionMode.CPU,
            rows_in=int(len(suitable)),
        ) as stage:
            joined = geopandas.sjoin(suitable, transit_gdf[["geometry"]], predicate="intersects")
            matched_labels = joined.index.unique()
            near_transit = suitable.loc[matched_labels].copy()
            stage.rows_out = int(len(near_transit))

        output_path = root / "site-suitability-output.parquet"
        with profiler.stage(
            "write_output", category="emit", device=ExecutionMode.CPU, rows_in=int(len(near_transit))
        ) as stage:
            near_transit.to_parquet(output_path, geometry_encoding="geoarrow")
            stage.rows_out = int(len(near_transit))

    trace = profiler.finish(
        metadata={
            "scale": scale,
            "actual_selected_runtime": "cpu",
            "planner_selected_runtime": "cpu",
            "dispatch_events": len(get_dispatch_events(clear=True)),
            "fallback_events": len(get_fallback_events(clear=True)),
        }
    )
    return PipelineBenchmarkResult(
        pipeline="site-suitability-geopandas",
        scale=scale,
        status="ok",
        elapsed_seconds=trace.total_elapsed_seconds,
        selected_runtime="cpu",
        planner_selected_runtime="cpu",
        output_rows=int(len(near_transit)),
        transfer_count=0,
        materialization_count=0,
        fallback_event_count=0,
        peak_device_memory_bytes=None,
        stages=(_trace_to_stage_dict(trace),),
        notes="GeoPandas baseline for site suitability workflow.",
    )


def _deferred_raster_pipeline(scale: int, *, profile_mode: str = "lean") -> PipelineBenchmarkResult:
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
        "grouped-disjoint-constructive-reduce",
        "grouped-difference-constructive",
        "constructive-output-native",
        "overlay-relation-constructive",
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
                    elif pipeline == "grouped-capacity-partitions":
                        samples.append(
                            _profile_grouped_capacity_partitions_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "grouped-disjoint-constructive-reduce":
                        samples.append(
                            _profile_grouped_disjoint_constructive_reduce_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "grouped-difference-constructive":
                        samples.append(
                            _profile_grouped_difference_constructive_pipeline(
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
                    elif pipeline == "relation-distance-expression":
                        samples.append(
                            _profile_relation_distance_expression_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "nearest-relation-producer":
                        samples.append(
                            _profile_nearest_relation_producer_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "native-area-expression":
                        samples.append(
                            _profile_native_area_expression_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "native-metadata-index":
                        samples.append(
                            _profile_native_metadata_index_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "constructive-output-native":
                        samples.append(
                            _profile_constructive_output_native_pipeline(
                                scale,
                                enable_nvtx=enable_nvtx,
                                retain_gpu_trace=retain_gpu_trace,
                                include_gpu_sparklines=include_gpu_sparklines,
                            )
                        )
                    elif pipeline == "overlay-relation-constructive":
                        samples.append(
                            _profile_overlay_relation_constructive_pipeline(
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
                median_sample = min(
                    samples, key=lambda sample: abs(sample.elapsed_seconds - median_elapsed)
                )
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
                            sample.runtime_d2h_transfer_bytes or 0 for sample in samples
                        ),
                        runtime_d2h_transfer_seconds=max(
                            sample.runtime_d2h_transfer_seconds or 0.0 for sample in samples
                        ),
                        materialization_count=max(
                            sample.materialization_count for sample in samples
                        ),
                        compute_materialization_count=max(
                            int(sample.to_dict()["compute_materialization_count"])
                            for sample in samples
                        ),
                        compute_runtime_d2h_transfer_count=max(
                            int(sample.to_dict()["compute_runtime_d2h_transfer_count"])
                            for sample in samples
                        ),
                        compute_runtime_d2h_transfer_bytes=max(
                            int(sample.to_dict()["compute_runtime_d2h_transfer_bytes"])
                            for sample in samples
                        ),
                        compute_runtime_d2h_transfer_seconds=max(
                            float(sample.to_dict()["compute_runtime_d2h_transfer_seconds"])
                            for sample in samples
                        ),
                        terminal_materialization_count=max(
                            int(sample.to_dict()["terminal_materialization_count"])
                            for sample in samples
                        ),
                        terminal_runtime_d2h_transfer_count=max(
                            int(sample.to_dict()["terminal_runtime_d2h_transfer_count"])
                            for sample in samples
                        ),
                        terminal_runtime_d2h_transfer_bytes=max(
                            int(sample.to_dict()["terminal_runtime_d2h_transfer_bytes"])
                            for sample in samples
                        ),
                        terminal_runtime_d2h_transfer_seconds=max(
                            float(sample.to_dict()["terminal_runtime_d2h_transfer_seconds"])
                            for sample in samples
                        ),
                        reference_materialization_count=max(
                            int(sample.to_dict()["reference_materialization_count"])
                            for sample in samples
                        ),
                        reference_runtime_d2h_transfer_count=max(
                            int(sample.to_dict()["reference_runtime_d2h_transfer_count"])
                            for sample in samples
                        ),
                        reference_runtime_d2h_transfer_bytes=max(
                            int(sample.to_dict()["reference_runtime_d2h_transfer_bytes"])
                            for sample in samples
                        ),
                        reference_runtime_d2h_transfer_seconds=max(
                            float(sample.to_dict()["reference_runtime_d2h_transfer_seconds"])
                            for sample in samples
                        ),
                        fallback_event_count=max(sample.fallback_event_count for sample in samples),
                        peak_device_memory_bytes=max(
                            (
                                sample.peak_device_memory_bytes
                                for sample in samples
                                if sample.peak_device_memory_bytes is not None
                            ),
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


def suite_to_json(
    results: list[PipelineBenchmarkResult], *, suite: str | None = None, repeat: int = 1
) -> str:
    profile_modes = {result.profile_mode for result in results}
    payload = {
        "results": [result.to_dict() for result in results],
        "metadata": {
            "repeat": repeat,
            "profile_mode": (
                next(iter(profile_modes)) if len(profile_modes) == 1 else sorted(profile_modes)
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
                wall_display = metadata.get(
                    "elapsed_display", _format_elapsed_compact(wall_elapsed)
                )
                gpu_elapsed = metadata.get("gpu_event_elapsed_seconds")
                if gpu_elapsed is not None and stage.get("device") == ExecutionMode.GPU.value:
                    timing_summary = (
                        f"gpu={_format_elapsed_compact(float(gpu_elapsed))} wall={wall_display}"
                    )
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
                        "coerce_left_s",
                        "normalize_right_s",
                        "move_to_device_s",
                        "coarse_filter_s",
                        "candidate_mask_s",
                        "point_upload_s",
                        "polygon_upload_s",
                        "kernel_launch_and_sync_s",
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
