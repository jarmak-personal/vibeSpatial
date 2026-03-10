from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
from shapely import box, to_wkb

from .synthetic import SyntheticSpec, generate_lines, generate_points, generate_polygons


class GeometryFamily(IntEnum):
    POINT = 0
    LINE = 1
    POLYGON = 2


@dataclass(frozen=True)
class PayloadModel:
    point_bytes: int
    line_bytes: int
    polygon_bytes: int
    promoted_point_bytes: int
    promoted_line_bytes: int
    promoted_polygon_bytes: int

    def bytes_for_family(self, family: GeometryFamily) -> int:
        return {
            GeometryFamily.POINT: self.point_bytes,
            GeometryFamily.LINE: self.line_bytes,
            GeometryFamily.POLYGON: self.polygon_bytes,
        }[family]

    def promoted_bytes_for_family(self, family: GeometryFamily) -> int:
        return {
            GeometryFamily.POINT: self.promoted_point_bytes,
            GeometryFamily.LINE: self.promoted_line_bytes,
            GeometryFamily.POLYGON: self.promoted_polygon_bytes,
        }[family]


@dataclass(frozen=True)
class LayoutBenchmarkResult:
    dataset_name: str
    scale: int
    family_counts: dict[str, int]
    payload_bytes: int
    tagged_bytes: int
    separate_bytes: int
    sort_partition_bytes: int
    promoted_bytes: int
    tagged_prep_ms: float
    separate_prep_ms: float
    sort_partition_prep_ms: float
    promoted_prep_ms: float
    tagged_warp_purity: float
    separate_warp_purity: float
    sort_partition_warp_purity: float
    recommended_default: str


DATASET_MIXES: dict[str, tuple[tuple[GeometryFamily, float], ...]] = {
    "point-dominated": (
        (GeometryFamily.POINT, 0.90),
        (GeometryFamily.LINE, 0.08),
        (GeometryFamily.POLYGON, 0.02),
    ),
    "polygon-dominated": (
        (GeometryFamily.POLYGON, 0.80),
        (GeometryFamily.LINE, 0.15),
        (GeometryFamily.POINT, 0.05),
    ),
    "mixed": (
        (GeometryFamily.POINT, 0.40),
        (GeometryFamily.LINE, 0.30),
        (GeometryFamily.POLYGON, 0.30),
    ),
}


def _median_wkb_bytes(geometries) -> int:
    values = to_wkb(list(geometries))
    return int(np.median([len(value) for value in values]))


def build_payload_model(seed: int = 0, sample_size: int = 512) -> PayloadModel:
    point_dataset = generate_points(
        SyntheticSpec("point", "uniform", count=sample_size, seed=seed)
    )
    line_dataset = generate_lines(
        SyntheticSpec("line", "random-walk", count=sample_size, seed=seed, vertices=8)
    )
    polygon_dataset = generate_polygons(
        SyntheticSpec("polygon", "star", count=sample_size, seed=seed, vertices=6, hole_probability=0.2)
    )

    promoted_points = [box(*geometry.bounds) for geometry in point_dataset.geometries]
    promoted_lines = [box(*geometry.bounds) for geometry in line_dataset.geometries]
    promoted_polygons = [box(*geometry.bounds) for geometry in polygon_dataset.geometries]

    return PayloadModel(
        point_bytes=_median_wkb_bytes(point_dataset.geometries),
        line_bytes=_median_wkb_bytes(line_dataset.geometries),
        polygon_bytes=_median_wkb_bytes(polygon_dataset.geometries),
        promoted_point_bytes=_median_wkb_bytes(promoted_points),
        promoted_line_bytes=_median_wkb_bytes(promoted_lines),
        promoted_polygon_bytes=_median_wkb_bytes(promoted_polygons),
    )


def generate_family_codes(
    scale: int,
    mix: tuple[tuple[GeometryFamily, float], ...],
    *,
    seed: int = 0,
) -> np.ndarray:
    remaining = scale
    families: list[int] = []
    for index, (family, ratio) in enumerate(mix):
        if index == len(mix) - 1:
            count = remaining
        else:
            count = min(int(round(scale * ratio)), remaining)
            remaining -= count
        families.extend([int(family)] * count)

    array = np.array(families, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    rng.shuffle(array)
    return array


def _payload_total(tags: np.ndarray, payload_model: PayloadModel) -> int:
    counts = np.bincount(tags, minlength=3)
    return int(
        counts[GeometryFamily.POINT] * payload_model.point_bytes
        + counts[GeometryFamily.LINE] * payload_model.line_bytes
        + counts[GeometryFamily.POLYGON] * payload_model.polygon_bytes
    )


def _promoted_total(tags: np.ndarray, payload_model: PayloadModel) -> int:
    counts = np.bincount(tags, minlength=3)
    return int(
        counts[GeometryFamily.POINT] * payload_model.promoted_point_bytes
        + counts[GeometryFamily.LINE] * payload_model.promoted_line_bytes
        + counts[GeometryFamily.POLYGON] * payload_model.promoted_polygon_bytes
    )


def _warp_purity(tags: np.ndarray, warp_size: int = 32) -> float:
    if tags.size == 0:
        return 1.0
    total = 0.0
    warps = 0
    for start in range(0, tags.size, warp_size):
        chunk = tags[start : start + warp_size]
        counts = np.bincount(chunk, minlength=3)
        total += counts.max() / chunk.size
        warps += 1
    return total / warps


def _bench_tagged(tags: np.ndarray) -> float:
    start = time.perf_counter()
    counts = np.bincount(tags, minlength=3)
    _ = counts
    return (time.perf_counter() - start) * 1_000


def _bench_separate(tags: np.ndarray) -> float:
    start = time.perf_counter()
    buckets = [np.flatnonzero(tags == code) for code in range(3)]
    order = np.concatenate(buckets)
    positions = np.empty_like(order)
    positions[order] = np.arange(order.size, dtype=np.int64)
    _ = order, positions
    return (time.perf_counter() - start) * 1_000


def _bench_sort_partition(tags: np.ndarray) -> float:
    start = time.perf_counter()
    order = np.argsort(tags, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size, dtype=np.int64)
    sorted_tags = tags[order]
    _ = sorted_tags, inverse
    return (time.perf_counter() - start) * 1_000


def _bench_promote(tags: np.ndarray, payload_model: PayloadModel) -> float:
    start = time.perf_counter()
    _ = _promoted_total(tags, payload_model)
    return (time.perf_counter() - start) * 1_000


def _recommend(tags: np.ndarray) -> str:
    purity = _warp_purity(tags)
    size = tags.size
    if purity >= 0.88:
        return "tagged-union storage with direct execution"
    if purity >= 0.70 or size < 10_000:
        return "tagged-union storage with optional late partitioning"
    return "dense tagged storage with sort-partition execution"


def benchmark_mixed_layout(
    dataset_name: str,
    scale: int,
    payload_model: PayloadModel,
    *,
    seed: int = 0,
) -> LayoutBenchmarkResult:
    tags = generate_family_codes(scale, DATASET_MIXES[dataset_name], seed=seed)
    counts = np.bincount(tags, minlength=3)
    payload_bytes = _payload_total(tags, payload_model)

    tagged_bytes = payload_bytes + scale * (1 + 4)
    separate_bytes = payload_bytes + scale * (1 + 4)
    sort_partition_bytes = payload_bytes + scale * (1 + 4 + 4 + 4)
    promoted_bytes = _promoted_total(tags, payload_model)

    return LayoutBenchmarkResult(
        dataset_name=dataset_name,
        scale=scale,
        family_counts={
            "point": int(counts[GeometryFamily.POINT]),
            "line": int(counts[GeometryFamily.LINE]),
            "polygon": int(counts[GeometryFamily.POLYGON]),
        },
        payload_bytes=payload_bytes,
        tagged_bytes=tagged_bytes,
        separate_bytes=separate_bytes,
        sort_partition_bytes=sort_partition_bytes,
        promoted_bytes=promoted_bytes,
        tagged_prep_ms=_bench_tagged(tags),
        separate_prep_ms=_bench_separate(tags),
        sort_partition_prep_ms=_bench_sort_partition(tags),
        promoted_prep_ms=_bench_promote(tags, payload_model),
        tagged_warp_purity=_warp_purity(tags),
        separate_warp_purity=1.0,
        sort_partition_warp_purity=1.0,
        recommended_default=_recommend(tags),
    )


def benchmark_matrix(
    scales: tuple[int, ...] = (100_000, 1_000_000),
    *,
    seed: int = 0,
) -> list[LayoutBenchmarkResult]:
    payload_model = build_payload_model(seed=seed)
    results = []
    for dataset_name in DATASET_MIXES:
        for scale in scales:
            results.append(
                benchmark_mixed_layout(dataset_name, scale, payload_model, seed=seed)
            )
    return results


def write_benchmark_report(path: str | Path, results: list[LayoutBenchmarkResult]) -> Path:
    output = Path(path)
    payload = [asdict(result) for result in results]
    output.write_text(json.dumps(payload, indent=2))
    return output
