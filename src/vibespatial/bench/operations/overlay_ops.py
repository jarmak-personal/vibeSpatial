"""Overlay operation benchmarks: gpu-overlay, segment-filters, segment-intersections, segment-primitives."""
from __future__ import annotations

from typing import Any

from vibespatial.bench.catalog import benchmark_operation
from vibespatial.bench.schema import (
    BenchmarkResult,
    timing_from_samples,
)


@benchmark_operation(
    name="gpu-overlay",
    description="GPU polygon overlay intersection",
    category="overlay",
    geometry_types=("polygon",),
    default_scale=1_000,
    tier=2,
    legacy_script="benchmark_gpu_overlay.py",
    tags=("gpu",),
)
def bench_gpu_overlay(
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

    from shapely.affinity import translate

    from vibespatial import from_shapely_geometries, has_gpu_runtime
    from vibespatial.overlay.gpu import overlay_intersection_owned
    from vibespatial.testing.synthetic import SyntheticSpec, generate_polygons

    if not has_gpu_runtime():
        return BenchmarkResult(
            operation="gpu-overlay",
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
    left_geoms = list(dataset.geometries)
    right_geoms = [translate(g, xoff=0.3, yoff=0.3) for g in left_geoms]

    left_owned = from_shapely_geometries(left_geoms)
    right_owned = from_shapely_geometries(right_geoms)

    # Warmup
    try:
        overlay_intersection_owned(left_owned, right_owned)
    except Exception:
        pass

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        overlay_intersection_owned(left_owned, right_owned)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    return BenchmarkResult(
        operation="gpu-overlay",
        tier=1,
        scale=scale,
        geometry_type="polygon",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        metadata={"repeat": repeat},
    )


@benchmark_operation(
    name="segment-filters",
    description="Segment-level MBR filtering benchmark",
    category="overlay",
    geometry_types=("polygon",),
    default_scale=200,
    tier=3,
    legacy_script="benchmark_segment_filters.py",
    tags=("gpu",),
)
def bench_segment_filters(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    **kwargs: Any,
) -> BenchmarkResult:
    from shapely.affinity import translate

    from vibespatial import benchmark_segment_filter, from_shapely_geometries
    from vibespatial.testing.synthetic import SyntheticSpec, generate_polygons

    base = list(generate_polygons(
        SyntheticSpec("polygon", "star", count=scale, seed=0)
    ).geometries)
    shifted = [translate(g, xoff=5.0, yoff=0.0) for g in base]

    left = from_shapely_geometries(base)
    right = from_shapely_geometries(shifted)
    result = benchmark_segment_filter(left, right, tile_size=512)

    timing = timing_from_samples([result.elapsed_seconds])

    return BenchmarkResult(
        operation="segment-filters",
        tier=1,
        scale=scale,
        geometry_type="polygon",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        metadata={
            "naive_segment_pairs": result.naive_segment_pairs,
            "filtered_segment_pairs": result.filtered_segment_pairs,
        },
    )


@benchmark_operation(
    name="segment-intersections",
    description="Segment intersection classification benchmark",
    category="overlay",
    geometry_types=("line",),
    default_scale=200,
    tier=3,
    legacy_script="benchmark_segment_intersections.py",
    tags=("gpu",),
)
def bench_segment_intersections(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from shapely.affinity import translate

    from vibespatial import (
        classify_segment_intersections,
        from_shapely_geometries,
        generate_segment_candidates,
    )
    from vibespatial.testing.synthetic import SyntheticSpec, generate_lines

    base = list(generate_lines(
        SyntheticSpec("line", "grid", count=scale, seed=0)
    ).geometries)
    shifted = [translate(g, xoff=0.5, yoff=0.5) for g in base]

    left = from_shapely_geometries(base)
    right = from_shapely_geometries(shifted)
    candidates = generate_segment_candidates(left, right)

    # Warmup
    classify_segment_intersections(left, right, candidate_pairs=candidates)

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        classify_segment_intersections(left, right, candidate_pairs=candidates)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    return BenchmarkResult(
        operation="segment-intersections",
        tier=1,
        scale=scale,
        geometry_type="line",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        metadata={"repeat": repeat},
    )


@benchmark_operation(
    name="segment-primitives",
    description="Segment intersection primitive benchmark",
    category="overlay",
    geometry_types=("line",),
    default_scale=200,
    tier=3,
    legacy_script="benchmark_segment_primitives.py",
    tags=("gpu",),
)
def bench_segment_primitives(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    **kwargs: Any,
) -> BenchmarkResult:
    from shapely.affinity import translate

    from vibespatial import benchmark_segment_intersections, from_shapely_geometries
    from vibespatial.testing.synthetic import SyntheticSpec, generate_lines

    base = list(generate_lines(
        SyntheticSpec("line", "grid", count=scale, seed=0)
    ).geometries)
    shifted = [translate(g, xoff=0.5, yoff=0.5) for g in base]

    left = from_shapely_geometries(base)
    right = from_shapely_geometries(shifted)
    result = benchmark_segment_intersections(left, right, tile_size=512)

    timing = timing_from_samples([result.elapsed_seconds])

    return BenchmarkResult(
        operation="segment-primitives",
        tier=1,
        scale=scale,
        geometry_type="line",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        metadata={
            "candidate_pairs": result.candidate_pairs,
            "proper_pairs": result.proper_pairs,
        },
    )
