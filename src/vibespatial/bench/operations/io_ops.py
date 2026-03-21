"""IO operation benchmarks: io-arrow, io-file, gpu-decode, mixed-layouts."""
from __future__ import annotations

from typing import Any

from vibespatial.bench.catalog import benchmark_operation
from vibespatial.bench.schema import (
    BenchmarkResult,
    timing_from_samples,
)


@benchmark_operation(
    name="io-arrow",
    description="GeoArrow / WKB / GeoParquet decode benchmark",
    category="io",
    geometry_types=("point", "polygon"),
    default_scale=100_000,
    tier=1,
    legacy_script="benchmark_io_arrow.py",
    tags=("io",),
)
def bench_io_arrow(
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

    from vibespatial import from_shapely_geometries
    from vibespatial.io.arrow import owned_to_wkb
    from vibespatial.testing.synthetic import SyntheticSpec, generate_points

    dataset = generate_points(SyntheticSpec("point", "grid", count=scale, seed=0))
    owned = from_shapely_geometries(list(dataset.geometries))

    # Benchmark WKB encoding roundtrip
    # Warmup
    owned_to_wkb(owned)

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        owned_to_wkb(owned)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    return BenchmarkResult(
        operation="io-arrow",
        tier=1,
        scale=scale,
        geometry_type="point",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        metadata={"repeat": repeat, "format": "wkb-encode"},
    )


@benchmark_operation(
    name="io-file",
    description="File-format IO benchmark (GeoJSON, Shapefile)",
    category="io",
    geometry_types=("point",),
    default_scale=100_000,
    tier=1,
    legacy_script="benchmark_io_file.py",
    tags=("io",),
)
def bench_io_file(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    **kwargs: Any,
) -> BenchmarkResult:
    from vibespatial import benchmark_geojson_ingest

    results = benchmark_geojson_ingest(
        geometry_type="point",
        rows=scale,
        repeat=max(1, repeat),
    )

    # Use the first (fastest) implementation's timing
    best = min(results, key=lambda r: r.elapsed_seconds)
    timing = timing_from_samples([best.elapsed_seconds])

    return BenchmarkResult(
        operation="io-file",
        tier=1,
        scale=scale,
        geometry_type="point",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        metadata={
            "format": "geojson",
            "implementation": best.implementation,
            "rows_per_second": best.rows_per_second,
        },
    )


@benchmark_operation(
    name="gpu-decode",
    description="GPU-accelerated WKB/GeoParquet decode benchmark",
    category="io",
    geometry_types=("point",),
    default_scale=100_000,
    tier=1,
    legacy_script="benchmark_gpu_decode.py",
    tags=("gpu", "io"),
)
def bench_gpu_decode(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from vibespatial import decode_wkb_owned, from_shapely_geometries
    from vibespatial.io.arrow import owned_to_wkb
    from vibespatial.testing.synthetic import SyntheticSpec, generate_points

    dataset = generate_points(SyntheticSpec("point", "grid", count=scale, seed=0))
    owned = from_shapely_geometries(list(dataset.geometries))
    wkb_data = owned_to_wkb(owned)

    # Warmup
    decode_wkb_owned(wkb_data)

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        decode_wkb_owned(wkb_data)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    return BenchmarkResult(
        operation="gpu-decode",
        tier=1,
        scale=scale,
        geometry_type="point",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        metadata={"repeat": repeat, "format": "wkb-decode"},
    )


@benchmark_operation(
    name="mixed-layouts",
    description="Mixed geometry layout benchmark (tagged vs separate)",
    category="io",
    geometry_types=("mixed",),
    default_scale=100_000,
    tier=3,
    legacy_script="benchmark_mixed_layouts.py",
    tags=("layout",),
)
def bench_mixed_layouts(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    **kwargs: Any,
) -> BenchmarkResult:
    from vibespatial.testing.mixed_layouts import benchmark_matrix

    results = benchmark_matrix((scale,), seed=0)
    if not results:
        return BenchmarkResult(
            operation="mixed-layouts",
            tier=1,
            scale=scale,
            geometry_type="mixed",
            precision=precision,
            status="error",
            status_reason="no results from benchmark_matrix",
            timing=timing_from_samples([]),
        )

    # Use the first result
    r = results[0]
    timing = timing_from_samples([r.tagged_prep_ms / 1000.0])

    return BenchmarkResult(
        operation="mixed-layouts",
        tier=1,
        scale=scale,
        geometry_type="mixed",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        metadata={
            "dataset_name": r.dataset_name,
            "warp_purity": r.tagged_warp_purity,
            "tagged_ms": r.tagged_prep_ms,
            "separate_ms": r.separate_prep_ms,
            "recommended_default": r.recommended_default,
        },
    )
