"""Constructive operation benchmarks: clip-rect, gpu-constructive."""
from __future__ import annotations

from typing import Any

from vibespatial.bench.catalog import benchmark_operation
from vibespatial.bench.schema import (
    BenchmarkResult,
    timing_from_samples,
)


@benchmark_operation(
    name="clip-rect",
    description="Rectangle clipping fast-path benchmark",
    category="constructive",
    geometry_types=("line", "polygon"),
    default_scale=5_000,
    tier=4,
    legacy_script="benchmark_clip_rect.py",
    tags=("gpu", "compare"),
)
def bench_clip_rect(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    nvtx: bool = False,
    gpu_sparkline: bool = False,
    trace: bool = False,
    **kwargs: Any,
) -> BenchmarkResult:
    from vibespatial import benchmark_clip_by_rect, from_shapely_geometries
    from vibespatial.testing.synthetic import SyntheticSpec, generate_lines

    dataset = generate_lines(SyntheticSpec("line", "grid", count=scale, seed=0))
    owned = from_shapely_geometries(list(dataset.geometries))

    rect = (100.0, 100.0, 700.0, 700.0)
    result = benchmark_clip_by_rect(owned, *rect, dataset=f"line-{scale}")

    timing = timing_from_samples([result.owned_elapsed_seconds])

    speedup = None
    baseline_timing = None
    baseline_name = None
    if result.shapely_elapsed_seconds and result.shapely_elapsed_seconds > 0:
        baseline_timing = timing_from_samples([result.shapely_elapsed_seconds])
        baseline_name = "shapely"
        if timing.median_seconds > 0:
            speedup = result.shapely_elapsed_seconds / timing.median_seconds

    return BenchmarkResult(
        operation="clip-rect",
        tier=1,
        scale=scale,
        geometry_type="line",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        baseline_name=baseline_name,
        baseline_timing=baseline_timing,
        speedup=speedup,
        metadata={
            "candidate_rows": result.candidate_rows,
            "fast_rows": result.fast_rows,
            "fallback_rows": result.fallback_rows,
        },
    )


@benchmark_operation(
    name="gpu-constructive",
    description="GPU constructive operations (clip + buffer pipeline)",
    category="constructive",
    geometry_types=("polygon",),
    default_scale=10_000,
    tier=4,
    legacy_script="benchmark_gpu_constructive.py",
    tags=("gpu", "compare"),
)
def bench_gpu_constructive(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    nvtx: bool = False,
    gpu_sparkline: bool = False,
    trace: bool = False,
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from vibespatial import from_shapely_geometries, has_gpu_runtime
    from vibespatial.constructive.clip_rect import clip_by_rect_owned
    from vibespatial.testing.synthetic import SyntheticSpec, generate_polygons

    if not has_gpu_runtime():
        return BenchmarkResult(
            operation="gpu-constructive",
            tier=1,
            scale=scale,
            geometry_type="polygon",
            precision=precision,
            status="skip",
            status_reason="GPU runtime not available",
            timing=timing_from_samples([]),
        )

    dataset = generate_polygons(
        SyntheticSpec("polygon", "regular-grid", count=scale, seed=0)
    )
    owned = from_shapely_geometries(list(dataset.geometries))
    rect = (100.0, 100.0, 700.0, 700.0)

    # Warmup
    clip_by_rect_owned(owned, *rect)

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        clip_by_rect_owned(owned, *rect)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    return BenchmarkResult(
        operation="gpu-constructive",
        tier=1,
        scale=scale,
        geometry_type="polygon",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        metadata={"repeat": repeat, "rect": list(rect)},
    )


@benchmark_operation(
    name="make-valid",
    description="Compact-invalid-row make_valid benchmark",
    category="constructive",
    geometry_types=("polygon",),
    default_scale=10_000,
    tier=4,
    legacy_script="benchmark_make_valid.py",
    tags=("gpu", "compare"),
)
def bench_make_valid(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    **kwargs: Any,
) -> BenchmarkResult:
    from shapely.geometry import Polygon

    from vibespatial import benchmark_make_valid

    invalid_every = 20
    values = [
        Polygon([(float(i), 0.0), (float(i) + 1.0, 1.0), (float(i) + 1.0, 2.0),
                 (float(i) + 1.0, 1.0), (float(i), 0.0)])
        if i % max(invalid_every, 1) == 0
        else Polygon([(float(i), 0.0), (float(i), 1.0), (float(i) + 1.0, 1.0),
                      (float(i) + 1.0, 0.0)])
        for i in range(scale)
    ]
    result = benchmark_make_valid(values)

    timing = timing_from_samples([result.compact_elapsed_seconds])
    baseline_timing = timing_from_samples([result.baseline_elapsed_seconds])
    speedup = result.speedup_vs_baseline

    return BenchmarkResult(
        operation="make-valid",
        tier=1,
        scale=scale,
        geometry_type="polygon",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        baseline_name="baseline",
        baseline_timing=baseline_timing,
        speedup=speedup,
        metadata={"repaired_rows": result.repaired_rows},
    )


@benchmark_operation(
    name="gpu-dissolve",
    description="Grouped GPU dissolve for rectangle coverages",
    category="constructive",
    geometry_types=("polygon",),
    default_scale=50_000,
    tier=2,
    legacy_script="benchmark_gpu_dissolve.py",
    tags=("gpu",),
)
def bench_gpu_dissolve(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    import numpy as np
    from shapely.geometry import box

    import vibespatial.api as geopandas
    from vibespatial.overlay.dissolve import execute_grouped_box_union_gpu

    groups = 100
    rows_per_group = scale // groups
    geometries = []
    group_values = []
    for group in range(groups):
        base_y = float(group * 10.0)
        for idx in range(rows_per_group):
            base_x = float(idx)
            geometries.append(box(base_x, base_y, base_x + 1.0, base_y + 1.0))
            group_values.append(group)

    frame = geopandas.GeoDataFrame(
        {"group": group_values, "geometry": geometries}, crs="EPSG:3857",
    )
    values = np.asarray(frame.geometry.array, dtype=object)
    grouped = frame.groupby("group", sort=True, observed=False, dropna=True)[
        frame.geometry.name
    ]
    group_positions = [
        np.asarray(positions, dtype=np.int32)
        for _, positions in grouped.indices.items()
    ]

    # Warmup
    execute_grouped_box_union_gpu(values, group_positions)

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        execute_grouped_box_union_gpu(values, group_positions)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    return BenchmarkResult(
        operation="gpu-dissolve",
        tier=1,
        scale=scale,
        geometry_type="polygon",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        metadata={"repeat": repeat, "groups": groups},
    )


@benchmark_operation(
    name="stroke-kernels",
    description="Point buffer and offset curve stroke kernels",
    category="constructive",
    geometry_types=("point", "line"),
    default_scale=1_000,
    tier=4,
    legacy_script="benchmark_stroke_kernels.py",
    tags=("gpu", "compare"),
)
def bench_stroke_kernels(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    **kwargs: Any,
) -> BenchmarkResult:
    from shapely.geometry import Point

    from vibespatial import benchmark_point_buffer

    values = [Point(float(i), float(i) * 0.25) for i in range(scale)]
    result = benchmark_point_buffer(values, distance=1.0, quad_segs=8)

    timing = timing_from_samples([result.owned_elapsed_seconds])
    speedup = result.speedup_vs_shapely
    baseline_timing = None
    if result.shapely_elapsed_seconds:
        baseline_timing = timing_from_samples([result.shapely_elapsed_seconds])

    return BenchmarkResult(
        operation="stroke-kernels",
        tier=1,
        scale=scale,
        geometry_type="point",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        baseline_name="shapely" if baseline_timing else None,
        baseline_timing=baseline_timing,
        speedup=speedup,
        metadata={"fast_rows": result.fast_rows, "fallback_rows": result.fallback_rows},
    )
