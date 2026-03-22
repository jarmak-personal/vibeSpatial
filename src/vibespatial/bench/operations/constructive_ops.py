"""Constructive operation benchmarks: clip-rect, gpu-constructive, make-valid, gpu-dissolve, stroke-kernels."""
from __future__ import annotations

import math
from typing import Any

from vibespatial.bench.catalog import benchmark_operation
from vibespatial.bench.schema import (
    BenchmarkResult,
    timing_from_samples,
)


def _clip_rect_from_owned(owned: Any, fraction: float = 0.6) -> tuple[float, float, float, float]:
    """Derive a clip rectangle from the actual geometry bounds of an OwnedGeometryArray.

    Returns the central ``fraction`` of the total bounding box, guaranteeing
    overlap with the input data.  Falls back to (0, 0, 1000, 1000) inner 60%
    if bounds cannot be computed.
    """
    from vibespatial.kernels.core.geometry_analysis import compute_total_bounds

    xmin, ymin, xmax, ymax = compute_total_bounds(owned)
    if math.isnan(xmin) or math.isnan(ymin) or math.isnan(xmax) or math.isnan(ymax):
        # Fallback: use default fixture bounds (0..1000) with the given fraction
        margin = (1.0 - fraction) / 2.0
        return (1000.0 * margin, 1000.0 * margin, 1000.0 * (1.0 - margin), 1000.0 * (1.0 - margin))
    margin = (1.0 - fraction) / 2.0
    dx = (xmax - xmin) * margin
    dy = (ymax - ymin) * margin
    return (xmin + dx, ymin + dy, xmax - dx, ymax - dy)


@benchmark_operation(
    name="clip-rect",
    description="Rectangle clipping fast-path benchmark",
    category="constructive",
    geometry_types=("line", "polygon"),
    default_scale=5_000,
    tier=4,
    legacy_script="benchmark_clip_rect.py",
    tags=("gpu", "compare"),
    max_scale=50_000,
)
def bench_clip_rect(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from vibespatial import benchmark_clip_by_rect
    from vibespatial.bench.fixture_loader import load_owned
    from vibespatial.bench.fixtures import InputFormat, resolve_fixture_spec

    spec = resolve_fixture_spec("line", "random-walk", scale)
    owned, read_seconds = load_owned(spec, InputFormat(input_format))

    # Guard: if fixture produced 0 rows, report as error.
    if owned.row_count == 0:
        return BenchmarkResult(
            operation="clip-rect",
            tier=1,
            scale=scale,
            geometry_type="line",
            precision=precision,
            status="error",
            status_reason="fixture produced 0 rows",
            timing=timing_from_samples([]),
            input_format=input_format,
            read_seconds=read_seconds,
        )

    # Derive clip rectangle from actual data bounds (central 60%) to
    # guarantee overlap regardless of scale or fixture distribution.
    rect = _clip_rect_from_owned(owned, fraction=0.6)

    try:
        result = benchmark_clip_by_rect(owned, *rect, dataset=f"line-{scale}")
    except (IndexError, ValueError) as exc:
        return BenchmarkResult(
            operation="clip-rect",
            tier=1,
            scale=scale,
            geometry_type="line",
            precision=precision,
            status="error",
            status_reason=f"clip_by_rect crashed: {exc}",
            timing=timing_from_samples([]),
            input_format=input_format,
            read_seconds=read_seconds,
        )

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
        input_format=input_format,
        read_seconds=read_seconds,
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
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from vibespatial import has_gpu_runtime
    from vibespatial.bench.fixture_loader import load_owned
    from vibespatial.bench.fixtures import InputFormat, resolve_fixture_spec
    from vibespatial.constructive.clip_rect import clip_by_rect_owned

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
            input_format=input_format,
        )

    spec = resolve_fixture_spec("polygon", "regular-grid", scale)
    owned, read_seconds = load_owned(spec, InputFormat(input_format))

    # Guard: if fixture produced 0 rows, report as error.
    if owned.row_count == 0:
        return BenchmarkResult(
            operation="gpu-constructive",
            tier=1,
            scale=scale,
            geometry_type="polygon",
            precision=precision,
            status="error",
            status_reason="fixture produced 0 rows",
            timing=timing_from_samples([]),
            input_format=input_format,
            read_seconds=read_seconds,
        )

    # Derive clip rectangle from actual data bounds (central 60%) to
    # guarantee overlap regardless of scale or fixture distribution.
    rect = _clip_rect_from_owned(owned, fraction=0.6)

    # Warmup run with guard for kernel-level crashes.
    try:
        warmup_result = clip_by_rect_owned(owned, *rect)
    except (IndexError, ValueError) as exc:
        return BenchmarkResult(
            operation="gpu-constructive",
            tier=1,
            scale=scale,
            geometry_type="polygon",
            precision=precision,
            status="error",
            status_reason=f"clip warmup crashed: {exc}",
            timing=timing_from_samples([]),
            input_format=input_format,
            read_seconds=read_seconds,
        )

    # Guard: if the clip produces 0 rows even with data-derived rect,
    # report as error rather than crashing downstream.
    if warmup_result.owned_result is not None and warmup_result.owned_result.row_count == 0:
        return BenchmarkResult(
            operation="gpu-constructive",
            tier=1,
            scale=scale,
            geometry_type="polygon",
            precision=precision,
            status="error",
            status_reason="clip produced 0 rows with data-derived rect",
            timing=timing_from_samples([]),
            input_format=input_format,
            read_seconds=read_seconds,
        )

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
        input_format=input_format,
        read_seconds=read_seconds,
        metadata={"repeat": repeat, "rect": list(rect)},
    )


@benchmark_operation(
    name="make-valid",
    description="make_valid benchmark on fixture with ~5% invalid polygons",
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
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from vibespatial import benchmark_make_valid
    from vibespatial.bench.fixture_loader import load_geodataframe
    from vibespatial.bench.fixtures import InputFormat, ensure_invalids_fixture

    fmt = InputFormat(input_format)
    spec, _ = ensure_invalids_fixture(scale, fmt=fmt)

    # Load via GeoDataFrame since benchmark_make_valid expects Shapely list
    gdf, read_seconds = load_geodataframe(spec, fmt)
    values = gdf.geometry.tolist()

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
        input_format=input_format,
        read_seconds=read_seconds,
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
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    import numpy as np

    from vibespatial.bench.fixture_loader import load_geodataframe
    from vibespatial.bench.fixtures import InputFormat, ensure_grouped_boxes_fixture
    from vibespatial.overlay.dissolve import execute_grouped_box_union_gpu

    fmt = InputFormat(input_format)
    groups = 100
    spec, _ = ensure_grouped_boxes_fixture(scale, groups=groups, fmt=fmt)
    frame, read_seconds = load_geodataframe(spec, fmt)

    values = np.asarray(frame.geometry.array, dtype=object)
    grouped = frame.groupby("group", sort=True, observed=False, dropna=True)[
        frame.geometry.name
    ]
    group_positions = [
        np.asarray(positions, dtype=np.int32)
        for _, positions in grouped.indices.items()
    ]

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
        input_format=input_format,
        read_seconds=read_seconds,
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
    max_scale=50_000,
)
def bench_stroke_kernels(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from vibespatial import benchmark_point_buffer
    from vibespatial.bench.fixture_loader import load_geodataframe
    from vibespatial.bench.fixtures import InputFormat, resolve_fixture_spec

    spec = resolve_fixture_spec("point", "grid", scale)
    gdf, read_seconds = load_geodataframe(spec, InputFormat(input_format))
    values = gdf.geometry.tolist()

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
        input_format=input_format,
        read_seconds=read_seconds,
        metadata={"fast_rows": result.fast_rows, "fallback_rows": result.fallback_rows},
    )
