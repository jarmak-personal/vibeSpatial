from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class GeoParquetMetadataSummary:
    source: str
    row_group_rows: np.ndarray
    xmin: np.ndarray
    ymin: np.ndarray
    xmax: np.ndarray
    ymax: np.ndarray

    @property
    def row_group_count(self) -> int:
        return int(self.row_group_rows.size)

    @property
    def total_rows(self) -> int:
        return int(self.row_group_rows.sum(dtype=np.int64))


@dataclass(frozen=True)
class GeoParquetPruneResult:
    strategy: str
    selected_row_groups: tuple[int, ...]
    decoded_row_count: int
    decoded_row_fraction: float
    pruned_row_group_fraction: float
    total_row_groups: int
    total_rows: int
    metadata_source: str


@dataclass(frozen=True)
class GeoParquetPlannerBenchmark:
    strategy: str
    elapsed_seconds: float
    selected_row_groups: int
    decoded_row_fraction: float
    pruned_row_group_fraction: float


def build_geoparquet_metadata_summary(
    *,
    source: str,
    row_group_rows: list[int] | tuple[int, ...] | np.ndarray,
    xmin: list[float] | tuple[float, ...] | np.ndarray,
    ymin: list[float] | tuple[float, ...] | np.ndarray,
    xmax: list[float] | tuple[float, ...] | np.ndarray,
    ymax: list[float] | tuple[float, ...] | np.ndarray,
) -> GeoParquetMetadataSummary:
    rows = np.asarray(row_group_rows, dtype=np.int64)
    xmin_arr = np.asarray(xmin, dtype=np.float64)
    ymin_arr = np.asarray(ymin, dtype=np.float64)
    xmax_arr = np.asarray(xmax, dtype=np.float64)
    ymax_arr = np.asarray(ymax, dtype=np.float64)
    size = rows.size
    if size == 0:
        raise ValueError("GeoParquet metadata summary requires at least one row group")
    if not all(array.size == size for array in (xmin_arr, ymin_arr, xmax_arr, ymax_arr)):
        raise ValueError("row_group_rows and bbox arrays must have matching sizes")
    return GeoParquetMetadataSummary(
        source=source,
        row_group_rows=rows,
        xmin=xmin_arr,
        ymin=ymin_arr,
        xmax=xmax_arr,
        ymax=ymax_arr,
    )


def _row_groups_from_mask(summary: GeoParquetMetadataSummary, mask: np.ndarray, *, strategy: str) -> GeoParquetPruneResult:
    selected = tuple(np.flatnonzero(mask).tolist())
    decoded_rows = int(summary.row_group_rows[np.asarray(mask, dtype=bool)].sum(dtype=np.int64))
    total_rows = summary.total_rows
    decoded_fraction = 0.0 if total_rows == 0 else decoded_rows / total_rows
    pruned_fraction = 1.0 - (len(selected) / summary.row_group_count)
    return GeoParquetPruneResult(
        strategy=strategy,
        selected_row_groups=selected,
        decoded_row_count=decoded_rows,
        decoded_row_fraction=decoded_fraction,
        pruned_row_group_fraction=pruned_fraction,
        total_row_groups=summary.row_group_count,
        total_rows=total_rows,
        metadata_source=summary.source,
    )


def select_row_groups_full_scan(summary: GeoParquetMetadataSummary) -> GeoParquetPruneResult:
    mask = np.ones(summary.row_group_count, dtype=bool)
    return _row_groups_from_mask(summary, mask, strategy="full_scan")


def select_row_groups_loop(summary: GeoParquetMetadataSummary, bbox: BBox) -> GeoParquetPruneResult:
    xmin, ymin, xmax, ymax = bbox
    selected: list[int] = []
    for index in range(summary.row_group_count):
        if not (
            summary.xmin[index] > xmax
            or summary.ymin[index] > ymax
            or summary.xmax[index] < xmin
            or summary.ymax[index] < ymin
        ):
            selected.append(index)
    mask = np.zeros(summary.row_group_count, dtype=bool)
    if selected:
        mask[np.asarray(selected, dtype=np.int64)] = True
    return _row_groups_from_mask(summary, mask, strategy="loop")


def select_row_groups_vectorized(summary: GeoParquetMetadataSummary, bbox: BBox) -> GeoParquetPruneResult:
    xmin, ymin, xmax, ymax = bbox
    mask = ~(
        (summary.xmin > xmax)
        | (summary.ymin > ymax)
        | (summary.xmax < xmin)
        | (summary.ymax < ymin)
    )
    return _row_groups_from_mask(summary, mask, strategy="vectorized")


def select_row_groups(
    summary: GeoParquetMetadataSummary,
    bbox: BBox,
    *,
    strategy: str = "auto",
) -> GeoParquetPruneResult:
    if strategy == "full_scan":
        return select_row_groups_full_scan(summary)
    if strategy == "loop":
        return select_row_groups_loop(summary, bbox)
    if strategy == "vectorized":
        return select_row_groups_vectorized(summary, bbox)
    if strategy != "auto":
        raise ValueError(f"Unsupported GeoParquet planner strategy: {strategy}")
    if summary.row_group_count < 64:
        return select_row_groups_loop(summary, bbox)
    return select_row_groups_vectorized(summary, bbox)


def benchmark_geoparquet_planner(
    summary: GeoParquetMetadataSummary,
    bbox: BBox,
    *,
    repeat: int = 5,
) -> tuple[GeoParquetPlannerBenchmark, ...]:
    strategies = ("full_scan", "loop", "vectorized", "auto")
    results: list[GeoParquetPlannerBenchmark] = []
    for strategy in strategies:
        elapsed_values: list[float] = []
        last_result: GeoParquetPruneResult | None = None
        for _ in range(max(repeat, 1)):
            start = perf_counter()
            last_result = select_row_groups(summary, bbox, strategy=strategy)
            elapsed_values.append(perf_counter() - start)
        assert last_result is not None
        results.append(
            GeoParquetPlannerBenchmark(
                strategy=strategy,
                elapsed_seconds=min(elapsed_values),
                selected_row_groups=len(last_result.selected_row_groups),
                decoded_row_fraction=last_result.decoded_row_fraction,
                pruned_row_group_fraction=last_result.pruned_row_group_fraction,
            )
        )
    return tuple(results)
