from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

import geopandas
import vibespatial.io_arrow as io_arrow
from vibespatial import has_gpu_runtime, has_pylibcudf_support
from vibespatial.cccl_primitives import (
    CompactionStrategy,
    ScanStrategy,
    compact_indices,
    exclusive_sum,
    has_cccl_primitives,
)
from vibespatial.cuda_runtime import get_cuda_runtime
from vibespatial.testing.synthetic import SyntheticSpec, generate_points


def _benchmark_wkb_decode_frame(frame: geopandas.GeoDataFrame, *, repeat: int) -> dict[str, object]:
    runtime = get_cuda_runtime()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "decode-wkb.parquet"
        frame.to_parquet(path, geometry_encoding="WKB")
        filesystem, normalized_path, _, geo_metadata = io_arrow._load_geoparquet_metadata(path)
        primary = geo_metadata["primary_column"]
        gpu_table = io_arrow._read_geoparquet_table_with_pylibcudf(
            normalized_path,
            columns=[primary],
            filesystem=filesystem,
        )
        host_table, _, _ = io_arrow._read_geoparquet_table_with_pyarrow(
            path,
            columns=[primary],
            bbox=None,
            row_groups=None,
        )

        io_arrow._decode_pylibcudf_geoparquet_table_to_owned(gpu_table, geo_metadata)
        runtime.synchronize()
        io_arrow._decode_geoparquet_table_to_owned(host_table, geo_metadata)

        gpu_elapsed = 0.0
        host_elapsed = 0.0
        for _ in range(repeat):
            start = perf_counter()
            io_arrow._decode_pylibcudf_geoparquet_table_to_owned(gpu_table, geo_metadata)
            runtime.synchronize()
            gpu_elapsed += perf_counter() - start

            start = perf_counter()
            io_arrow._decode_geoparquet_table_to_owned(host_table, geo_metadata)
            host_elapsed += perf_counter() - start

    gpu_elapsed /= repeat
    host_elapsed /= repeat
    rows = len(frame)
    return {
        "status": "ok",
        "rows": rows,
        "repeat": repeat,
        "gpu_elapsed_seconds": gpu_elapsed,
        "gpu_rows_per_second": rows / gpu_elapsed if gpu_elapsed else float("inf"),
        "host_elapsed_seconds": host_elapsed,
        "host_rows_per_second": rows / host_elapsed if host_elapsed else float("inf"),
        "speedup_vs_host": (host_elapsed / gpu_elapsed) if gpu_elapsed else float("inf"),
    }


def _benchmark_point_geoarrow_decode(*, scale: int, repeat: int, seed: int) -> dict[str, object]:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return {
            "status": "unavailable",
            "reason": "CUDA runtime and pylibcudf are required for GPU decode benchmarking",
        }

    dataset = generate_points(SyntheticSpec("point", "uniform", count=scale, seed=seed))
    frame = dataset.to_geodataframe()
    runtime = get_cuda_runtime()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "points.parquet"
        frame.to_parquet(path, geometry_encoding="geoarrow")
        filesystem, normalized_path, _, geo_metadata = io_arrow._load_geoparquet_metadata(path)
        primary = geo_metadata["primary_column"]
        gpu_table = io_arrow._read_geoparquet_table_with_pylibcudf(
            normalized_path,
            columns=[primary],
            filesystem=filesystem,
        )
        host_table, _, _ = io_arrow._read_geoparquet_table_with_pyarrow(
            path,
            columns=[primary],
            bbox=None,
            row_groups=None,
        )

        io_arrow._decode_pylibcudf_geoparquet_table_to_owned(gpu_table, geo_metadata)
        runtime.synchronize()
        io_arrow._decode_geoparquet_table_to_owned(host_table, geo_metadata)

        gpu_elapsed = 0.0
        host_elapsed = 0.0
        for _ in range(repeat):
            start = perf_counter()
            io_arrow._decode_pylibcudf_geoparquet_table_to_owned(gpu_table, geo_metadata)
            runtime.synchronize()
            gpu_elapsed += perf_counter() - start

            start = perf_counter()
            io_arrow._decode_geoparquet_table_to_owned(host_table, geo_metadata)
            host_elapsed += perf_counter() - start

    gpu_elapsed /= repeat
    host_elapsed /= repeat
    return {
        "status": "ok",
        "rows": scale,
        "repeat": repeat,
        "gpu_elapsed_seconds": gpu_elapsed,
        "gpu_rows_per_second": scale / gpu_elapsed if gpu_elapsed else float("inf"),
        "host_elapsed_seconds": host_elapsed,
        "host_rows_per_second": scale / host_elapsed if host_elapsed else float("inf"),
        "speedup_vs_host": (host_elapsed / gpu_elapsed) if gpu_elapsed else float("inf"),
    }


def _benchmark_point_wkb_decode(*, scale: int, repeat: int, seed: int) -> dict[str, object]:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return {
            "status": "unavailable",
            "reason": "CUDA runtime and pylibcudf are required for GPU decode benchmarking",
        }

    dataset = generate_points(SyntheticSpec("point", "uniform", count=scale, seed=seed))
    return _benchmark_wkb_decode_frame(dataset.to_geodataframe(), repeat=repeat)


def _benchmark_linestring_wkb_decode(*, scale: int, repeat: int) -> dict[str, object]:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return {
            "status": "unavailable",
            "reason": "CUDA runtime and pylibcudf are required for GPU decode benchmarking",
        }

    from shapely.geometry import LineString

    geometries = [
        LineString(
            [
                (float(index), 0.0),
                (float(index) + 1.0, 1.0),
                (float(index) + 2.0, 0.0),
            ]
        )
        for index in range(scale)
    ]
    return _benchmark_wkb_decode_frame(
        geopandas.GeoDataFrame({"geometry": geometries}, geometry="geometry"),
        repeat=repeat,
    )


def _benchmark_mixed_point_linestring_wkb_decode(*, scale: int, repeat: int) -> dict[str, object]:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return {
            "status": "unavailable",
            "reason": "CUDA runtime and pylibcudf are required for GPU decode benchmarking",
        }

    from shapely.geometry import LineString, Point

    geometries = []
    for index in range(scale):
        if index % 4 == 0:
            geometries.append(Point(float(index), float(index)))
        elif index % 4 == 1:
            geometries.append(LineString([(float(index), 0.0), (float(index) + 1.0, 1.0)]))
        elif index % 4 == 2:
            geometries.append(Point())
        else:
            geometries.append(LineString())
    return _benchmark_wkb_decode_frame(
        geopandas.GeoDataFrame({"geometry": geometries}, geometry="geometry"),
        repeat=repeat,
    )


def _benchmark_decode_primitives(*, scale: int, repeat: int, seed: int) -> dict[str, object]:
    if not has_gpu_runtime():
        return {"status": "unavailable", "reason": "CUDA runtime is required"}

    import cupy as cp

    rng = cp.random.RandomState(seed)
    mask = (rng.random_sample(scale) > 0.2).astype(cp.uint8)
    lengths = rng.randint(0, 5, size=scale, dtype=cp.int32)
    selected_lengths = lengths[mask.astype(bool)]

    def _cupy_select():
        return cp.flatnonzero(mask).astype(cp.int32, copy=False)

    def _cupy_exscan():
        out = cp.cumsum(selected_lengths, dtype=cp.int32)
        out -= selected_lengths
        return out

    cases: list[tuple[str, callable]] = [
        ("cupy_select", _cupy_select),
        ("cupy_exscan", _cupy_exscan),
    ]
    if has_cccl_primitives():
        cases.extend(
            [
                ("cccl_select", lambda: compact_indices(mask, strategy=CompactionStrategy.CCCL_SELECT).values),
                (
                    "cccl_exscan",
                    lambda: exclusive_sum(
                        selected_lengths,
                        strategy=ScanStrategy.CCCL_EXCLUSIVE_SCAN,
                    ),
                ),
            ]
        )

    results: dict[str, object] = {"status": "ok", "selected_rows": int(selected_lengths.size)}
    for _, fn in cases:
        fn()
        cp.cuda.Stream.null.synchronize()
    for label, fn in cases:
        elapsed = 0.0
        for _ in range(repeat):
            start = perf_counter()
            out = fn()
            cp.cuda.Stream.null.synchronize()
            elapsed += perf_counter() - start
        elapsed /= repeat
        results[label] = {
            "elapsed_seconds": elapsed,
            "rows_per_second": scale / elapsed if elapsed else float("inf"),
            "output_size": int(out.size),
        }
    return results


def _benchmark_wkb_header_scan(*, scale: int, repeat: int, seed: int) -> dict[str, object]:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return {
            "status": "unavailable",
            "reason": "CUDA runtime and pylibcudf are required for GPU decode benchmarking",
        }

    from shapely.geometry import LineString, Point, Polygon

    rng = np.random.default_rng(seed)
    geometries = []
    for index in range(scale):
        selector = index % 4
        if selector == 0:
            geometries.append(Point(float(index), float(index)))
        elif selector == 1:
            geometries.append(LineString([(float(index), 0.0), (float(index) + 1.0, 1.0)]))
        elif selector == 2:
            base = float(index)
            geometries.append(Polygon([(base, 0.0), (base + 1.0, 0.0), (base + 1.0, 1.0), (base, 0.0)]))
        else:
            geometries.append(None if rng.random() < 0.5 else Point(float(index), -float(index)))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "mixed-wkb.parquet"
        frame = geopandas.GeoDataFrame({"geometry": geometries}, geometry="geometry")
        frame.to_parquet(path, geometry_encoding="WKB")
        table = io_arrow._read_geoparquet_table_with_pylibcudf(path, columns=["geometry"])
        column = table.columns()[0]

        io_arrow._scan_pylibcudf_wkb_headers(column)
        get_cuda_runtime().synchronize()

        elapsed = 0.0
        scan = None
        for _ in range(repeat):
            start = perf_counter()
            scan = io_arrow._scan_pylibcudf_wkb_headers(column)
            get_cuda_runtime().synchronize()
            elapsed += perf_counter() - start
        elapsed /= repeat

    assert scan is not None
    return {
        "status": "ok",
        "rows": scale,
        "repeat": repeat,
        "elapsed_seconds": elapsed,
        "rows_per_second": scale / elapsed if elapsed else float("inf"),
        "native_rows": scan.native_count,
        "fallback_rows": scan.fallback_count,
    }


def _benchmark_wkb_family_select_choices(*, scale: int, repeat: int) -> dict[str, object]:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        return {
            "status": "unavailable",
            "reason": "CUDA runtime and pylibcudf are required for GPU decode benchmarking",
        }
    if not has_cccl_primitives():
        return {
            "status": "unavailable",
            "reason": "CCCL Python bindings are required for WKB family select comparisons",
        }

    import cupy as cp
    from shapely.geometry import LineString, Point

    geometries = []
    for index in range(scale):
        if index % 4 == 0:
            geometries.append(Point(float(index), float(index)))
        elif index % 4 == 1:
            geometries.append(LineString([(float(index), 0.0), (float(index) + 1.0, 1.0)]))
        elif index % 4 == 2:
            geometries.append(Point())
        else:
            geometries.append(LineString())

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "family-select-wkb.parquet"
        geopandas.GeoDataFrame({"geometry": geometries}, geometry="geometry").to_parquet(
            path,
            geometry_encoding="WKB",
        )
        table = io_arrow._read_geoparquet_table_with_pylibcudf(path, columns=["geometry"])
        scan = io_arrow._scan_pylibcudf_wkb_headers(table.columns()[0])

    masks = {
        "point_mask": scan.point_mask,
        "linestring_mask": scan.family_tags
        == np.int8(io_arrow.FAMILY_TAGS[io_arrow.GeometryFamily.LINESTRING]),
    }
    runtime = get_cuda_runtime()
    results: dict[str, object] = {"status": "ok", "rows": scale, "repeat": repeat}
    for mask_name, mask in masks.items():
        cases = [
            ("cupy_bool", lambda m=mask: cp.flatnonzero(m).astype(cp.int32, copy=False)),
            ("cupy_u8", lambda m=mask: cp.flatnonzero(m.astype(cp.uint8, copy=False)).astype(cp.int32, copy=False)),
            (
                "cccl_u8",
                lambda m=mask: compact_indices(
                    m.astype(cp.uint8, copy=False),
                    strategy=CompactionStrategy.CCCL_SELECT,
                ).values,
            ),
        ]
        family_results: dict[str, object] = {}
        for _, fn in cases:
            fn()
            runtime.synchronize()
        for label, fn in cases:
            elapsed = 0.0
            out_size = 0
            for _ in range(repeat):
                start = perf_counter()
                out = fn()
                runtime.synchronize()
                elapsed += perf_counter() - start
                out_size = int(out.size)
            elapsed /= repeat
            family_results[label] = {
                "elapsed_seconds": elapsed,
                "rows_per_second": scale / elapsed if elapsed else float("inf"),
                "output_size": out_size,
            }
        results[mask_name] = family_results
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark device-side geometry decode stages.")
    parser.add_argument("--scale", type=int, default=1_000_000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    payload = {
        "geoarrow_point_decode": _benchmark_point_geoarrow_decode(
            scale=args.scale,
            repeat=args.repeat,
            seed=args.seed,
        ),
        "wkb_point_decode": _benchmark_point_wkb_decode(
            scale=args.scale,
            repeat=args.repeat,
            seed=args.seed,
        ),
        "wkb_linestring_decode": _benchmark_linestring_wkb_decode(
            scale=args.scale,
            repeat=args.repeat,
        ),
        "primitive_choices": _benchmark_decode_primitives(
            scale=args.scale,
            repeat=args.repeat,
            seed=args.seed,
        ),
        "wkb_mixed_header_scan": _benchmark_wkb_header_scan(
            scale=args.scale,
            repeat=args.repeat,
            seed=args.seed,
        ),
        "wkb_family_select_choices": _benchmark_wkb_family_select_choices(
            scale=args.scale,
            repeat=args.repeat,
        ),
        "wkb_mixed_decode": _benchmark_mixed_point_linestring_wkb_decode(
            scale=args.scale,
            repeat=args.repeat,
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
