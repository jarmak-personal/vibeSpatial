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
    max_scale=10_000,
)
def bench_gpu_overlay(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from vibespatial import has_gpu_runtime
    from vibespatial.bench.fixture_loader import load_owned
    from vibespatial.bench.fixtures import (
        InputFormat,
        ensure_shifted_fixture,
        resolve_fixture_spec,
    )
    from vibespatial.overlay.gpu import overlay_intersection_owned

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
            input_format=input_format,
        )

    fmt = InputFormat(input_format)
    left_spec = resolve_fixture_spec("polygon", "regular-grid", scale)
    right_spec, _ = ensure_shifted_fixture(left_spec, xoff=0.3, yoff=0.3, fmt=fmt)

    left_owned, read_s1 = load_owned(left_spec, fmt)
    right_owned, read_s2 = load_owned(right_spec, fmt)
    read_seconds = read_s1 + read_s2

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
        input_format=input_format,
        read_seconds=read_seconds,
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
    max_scale=5_000,
)
def bench_segment_filters(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from vibespatial import benchmark_segment_filter
    from vibespatial.bench.fixture_loader import load_owned
    from vibespatial.bench.fixtures import (
        InputFormat,
        ensure_shifted_fixture,
        resolve_fixture_spec,
    )

    fmt = InputFormat(input_format)
    left_spec = resolve_fixture_spec("polygon", "regular-grid", scale)
    right_spec, _ = ensure_shifted_fixture(left_spec, xoff=5.0, yoff=0.0, fmt=fmt)

    left_owned, read_s1 = load_owned(left_spec, fmt)
    right_owned, read_s2 = load_owned(right_spec, fmt)
    read_seconds = read_s1 + read_s2

    result = benchmark_segment_filter(left_owned, right_owned, tile_size=512)

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
        input_format=input_format,
        read_seconds=read_seconds,
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
    max_scale=5_000,
)
def bench_segment_intersections(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from vibespatial import classify_segment_intersections, generate_segment_candidates
    from vibespatial.bench.fixture_loader import load_owned
    from vibespatial.bench.fixtures import (
        InputFormat,
        ensure_shifted_fixture,
        resolve_fixture_spec,
    )

    fmt = InputFormat(input_format)
    left_spec = resolve_fixture_spec("line", "random-walk", scale)
    right_spec, _ = ensure_shifted_fixture(left_spec, xoff=0.5, yoff=0.5, fmt=fmt)

    left_owned, read_s1 = load_owned(left_spec, fmt)
    right_owned, read_s2 = load_owned(right_spec, fmt)
    read_seconds = read_s1 + read_s2

    candidates = generate_segment_candidates(left_owned, right_owned)

    classify_segment_intersections(left_owned, right_owned, candidate_pairs=candidates)

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        classify_segment_intersections(left_owned, right_owned, candidate_pairs=candidates)
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
        input_format=input_format,
        read_seconds=read_seconds,
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
    max_scale=5_000,
)
def bench_segment_primitives(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from vibespatial import benchmark_segment_intersections
    from vibespatial.bench.fixture_loader import load_owned
    from vibespatial.bench.fixtures import (
        InputFormat,
        ensure_shifted_fixture,
        resolve_fixture_spec,
    )

    fmt = InputFormat(input_format)
    left_spec = resolve_fixture_spec("line", "random-walk", scale)
    right_spec, _ = ensure_shifted_fixture(left_spec, xoff=0.5, yoff=0.5, fmt=fmt)

    left_owned, read_s1 = load_owned(left_spec, fmt)
    right_owned, read_s2 = load_owned(right_spec, fmt)
    read_seconds = read_s1 + read_s2

    result = benchmark_segment_intersections(left_owned, right_owned, tile_size=512)

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
        input_format=input_format,
        read_seconds=read_seconds,
        metadata={
            "candidate_pairs": result.candidate_pairs,
            "proper_pairs": result.proper_pairs,
        },
    )
