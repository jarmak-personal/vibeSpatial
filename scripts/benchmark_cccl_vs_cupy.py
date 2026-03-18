"""Benchmark CuPy vs CCCL Python primitives (cold, warm, and make_* reusable).

Measures the three overlapping primitives — exclusive scan, compaction,
and reduction — at multiple scales. Reports cold-call, warmed, and
make_*-reusable timings so we can decide whether to change AUTO defaults
before the polygon-family GPU push.

Usage:
    uv run python scripts/benchmark_cccl_vs_cupy.py
    uv run python scripts/benchmark_cccl_vs_cupy.py --scales 10000 100000 1000000 --json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np


@dataclass(frozen=True, slots=True)
class PrimitiveResult:
    primitive: str
    scale: int
    cupy_cold_ms: float
    cupy_warm_ms: float
    cccl_cold_ms: float
    cccl_warm_ms: float
    cccl_make_warm_ms: float
    cupy_matches_cccl: bool
    winner_warm: str
    winner_make: str
    cupy_vs_cccl_warm_ratio: float
    cupy_vs_cccl_make_ratio: float


def _require_gpu():
    try:
        import cupy as cp

        _ = cp.cuda.Device(0).compute_capability
    except Exception:
        print("ERROR: No GPU available. This benchmark requires CUDA.", file=sys.stderr)
        sys.exit(1)


def _sync():
    import cupy as cp

    cp.cuda.Stream.null.synchronize()


def _time_ms(fn, *, warmup: int = 0, repeat: int = 7) -> tuple[float, float]:
    """Return (first_call_ms, best_of_repeat_ms)."""
    _sync()
    t0 = perf_counter()
    fn()
    _sync()
    cold = (perf_counter() - t0) * 1000.0

    for _ in range(warmup):
        fn()
        _sync()

    best = float("inf")
    for _ in range(repeat):
        _sync()
        t0 = perf_counter()
        fn()
        _sync()
        best = min(best, (perf_counter() - t0) * 1000.0)
    return cold, best


def _build_result(
    primitive: str,
    scale: int,
    cupy_cold: float,
    cupy_warm: float,
    cccl_cold: float,
    cccl_warm: float,
    cccl_make_warm: float,
    match: bool,
) -> PrimitiveResult:
    return PrimitiveResult(
        primitive=primitive,
        scale=scale,
        cupy_cold_ms=round(cupy_cold, 3),
        cupy_warm_ms=round(cupy_warm, 3),
        cccl_cold_ms=round(cccl_cold, 3),
        cccl_warm_ms=round(cccl_warm, 3),
        cccl_make_warm_ms=round(cccl_make_warm, 3),
        cupy_matches_cccl=match,
        winner_warm="cupy" if cupy_warm <= cccl_warm else "cccl",
        winner_make="cupy" if cupy_warm <= cccl_make_warm else "cccl_make",
        cupy_vs_cccl_warm_ratio=round(cccl_warm / max(cupy_warm, 0.001), 2),
        cupy_vs_cccl_make_ratio=round(cccl_make_warm / max(cupy_warm, 0.001), 2),
    )


# ---------------------------------------------------------------------------
# make_* helper: allocate temp storage for a CCCL reusable callable
# ---------------------------------------------------------------------------

def _alloc_temp_storage(make_obj, *call_args):
    """Call the make_* object with None temp storage to get required size,
    allocate, and return (temp_storage, callable_args)."""
    import cupy as cp

    temp_bytes = make_obj(None, *call_args)
    if isinstance(temp_bytes, int) and temp_bytes > 0:
        return cp.empty(temp_bytes, dtype=cp.uint8)
    # Some versions return the size differently; try a small default
    return cp.empty(max(int(temp_bytes) if temp_bytes else 1, 1), dtype=cp.uint8)


# ---------------------------------------------------------------------------
# Exclusive scan
# ---------------------------------------------------------------------------

def _bench_exclusive_scan(scale: int) -> PrimitiveResult:
    import cupy as cp
    from cuda.compute import algorithms as cccl

    values = cp.ones(scale, dtype=cp.int32)
    out_cupy = cp.empty_like(values)
    out_cccl = cp.empty_like(values)
    init = np.asarray(0, dtype=np.int32)

    def _sum_op(a, b):  # pragma: no cover
        return a + b

    # CuPy path
    def cupy_scan():
        r = cp.cumsum(values, dtype=values.dtype)
        r -= values
        out_cupy[:] = r

    cupy_cold, cupy_warm = _time_ms(cupy_scan)

    # CCCL one-shot path
    def cccl_scan():
        cccl.exclusive_scan(values, out_cccl, _sum_op, init, int(values.size))

    cccl_cold, cccl_warm = _time_ms(cccl_scan)

    # CCCL make_* reusable path
    scanner = cccl.make_exclusive_scan(values, out_cccl, _sum_op, init)
    # Query temp storage size
    temp_bytes = scanner(None, values, out_cccl, _sum_op, int(values.size), init)
    temp = cp.empty(max(int(temp_bytes), 1), dtype=cp.uint8) if temp_bytes else cp.empty(1, dtype=cp.uint8)

    def cccl_make_scan():
        scanner(temp, values, out_cccl, _sum_op, int(values.size), init)

    _, cccl_make_warm = _time_ms(cccl_make_scan)

    _sync()
    match = bool(cp.allclose(out_cupy, out_cccl))

    return _build_result("exclusive_scan", scale, cupy_cold, cupy_warm,
                         cccl_cold, cccl_warm, cccl_make_warm, match)


# ---------------------------------------------------------------------------
# Compaction (bool mask → indices)
# ---------------------------------------------------------------------------

def _bench_compaction(scale: int) -> PrimitiveResult:
    import cupy as cp
    from cuda.compute import algorithms as cccl

    rng = cp.random.default_rng(42)
    mask = rng.integers(0, 2, size=scale, dtype=cp.int32)
    out_cccl = cp.empty(scale, dtype=cp.int32)
    count_cccl = cp.empty(1, dtype=cp.int32)
    indices = cp.arange(scale, dtype=cp.int32)

    def _selected(idx):  # pragma: no cover
        return mask[idx] != 0

    # CuPy path
    cupy_result = None

    def cupy_compact():
        nonlocal cupy_result
        cupy_result = cp.flatnonzero(mask).astype(cp.int32, copy=False)

    cupy_cold, cupy_warm = _time_ms(cupy_compact)

    # CCCL one-shot path
    def cccl_compact():
        cccl.select(indices, out_cccl, count_cccl, _selected, int(mask.size))

    cccl_cold, cccl_warm = _time_ms(cccl_compact)

    # CCCL make_* reusable path — select doesn't have make_* in all versions,
    # so fall back to one-shot timing if make_select is unavailable
    cccl_make_warm = cccl_warm
    if hasattr(cccl, "make_select"):
        try:
            selector = cccl.make_select(indices, out_cccl, count_cccl, _selected)
            temp_bytes = selector(None, indices, out_cccl, count_cccl, _selected, int(mask.size))
            temp = cp.empty(max(int(temp_bytes), 1), dtype=cp.uint8) if temp_bytes else cp.empty(1, dtype=cp.uint8)

            def cccl_make_compact():
                selector(temp, indices, out_cccl, count_cccl, _selected, int(mask.size))

            _, cccl_make_warm = _time_ms(cccl_make_compact)
        except Exception:
            pass  # Fall back to one-shot timing

    _sync()
    cccl_count = int(count_cccl.get()[0])
    match = bool(cp.array_equal(cupy_result, out_cccl[:cccl_count]))

    return _build_result("compaction", scale, cupy_cold, cupy_warm,
                         cccl_cold, cccl_warm, cccl_make_warm, match)


# ---------------------------------------------------------------------------
# Reduction (sum)
# ---------------------------------------------------------------------------

def _bench_reduction(scale: int) -> PrimitiveResult:
    import cupy as cp
    from cuda.compute import algorithms as cccl

    values = cp.ones(scale, dtype=cp.float64)
    out_cupy = cp.empty(1, dtype=cp.float64)
    out_cccl = cp.empty(1, dtype=cp.float64)
    h_init = np.asarray(0.0, dtype=np.float64)

    def _add(a, b):  # pragma: no cover
        return a + b

    # CuPy path
    def cupy_reduce():
        out_cupy[0] = cp.sum(values)

    cupy_cold, cupy_warm = _time_ms(cupy_reduce)

    # CCCL one-shot path
    def cccl_reduce():
        cccl.reduce_into(values, out_cccl, _add, int(values.size), h_init)

    cccl_cold, cccl_warm = _time_ms(cccl_reduce)

    # CCCL make_* reusable path
    reducer = cccl.make_reduce_into(values, out_cccl, _add, h_init)
    temp_bytes = reducer(None, values, out_cccl, _add, int(values.size), h_init)
    temp = cp.empty(max(int(temp_bytes), 1), dtype=cp.uint8) if temp_bytes else cp.empty(1, dtype=cp.uint8)

    def cccl_make_reduce():
        reducer(temp, values, out_cccl, _add, int(values.size), h_init)

    _, cccl_make_warm = _time_ms(cccl_make_reduce)

    _sync()
    match = bool(abs(float(out_cupy.get()[0]) - float(out_cccl.get()[0])) < 1.0)

    return _build_result("reduction", scale, cupy_cold, cupy_warm,
                         cccl_cold, cccl_warm, cccl_make_warm, match)


# ---------------------------------------------------------------------------
# Segmented reduce (new primitive — no CuPy equivalent, compare vs workaround)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SegmentedResult:
    primitive: str
    scale: int
    segments: int
    cupy_workaround_warm_ms: float
    cccl_cold_ms: float
    cccl_warm_ms: float
    cccl_make_warm_ms: float
    results_match: bool
    cccl_make_vs_cupy_ratio: float


def _bench_segmented_reduce(scale: int) -> SegmentedResult:
    import cupy as cp
    from cuda.compute import algorithms as cccl

    # Simulate per-polygon vertex reduction: ~100 vertices per polygon
    segment_size = 100
    n_segments = max(1, scale // segment_size)
    actual_scale = n_segments * segment_size
    values = cp.ones(actual_scale, dtype=cp.float64)
    starts = cp.arange(0, actual_scale, segment_size, dtype=cp.int32)
    ends = starts + segment_size
    out_cccl = cp.empty(n_segments, dtype=cp.float64)
    h_init = np.asarray(0.0, dtype=np.float64)

    def _add(a, b):  # pragma: no cover
        return a + b

    # CuPy workaround: cumsum + fancy indexing
    out_cupy = cp.empty(n_segments, dtype=cp.float64)

    def cupy_seg_reduce():
        prefix = cp.cumsum(values, dtype=cp.float64)
        seg_sums = prefix[ends.astype(cp.int64) - 1]
        shifted = cp.zeros_like(seg_sums)
        if n_segments > 1:
            shifted[1:] = prefix[starts[1:].astype(cp.int64) - 1]
        out_cupy[:] = seg_sums - shifted

    _, cupy_warm = _time_ms(cupy_seg_reduce)

    # CCCL one-shot
    def cccl_seg_reduce():
        cccl.segmented_reduce(
            values, out_cccl, starts, ends, _add, h_init, n_segments,
        )

    cccl_cold, cccl_warm = _time_ms(cccl_seg_reduce)

    # CCCL make_* reusable
    cccl_make_warm = cccl_warm
    if hasattr(cccl, "make_segmented_reduce"):
        try:
            seg_reducer = cccl.make_segmented_reduce(
                values, out_cccl, starts, ends, _add, h_init,
            )
            temp_bytes = seg_reducer(
                None, values, out_cccl, starts, ends, _add, n_segments, h_init,
            )
            temp = cp.empty(max(int(temp_bytes), 1), dtype=cp.uint8) if temp_bytes else cp.empty(1, dtype=cp.uint8)

            def cccl_make_seg_reduce():
                seg_reducer(
                    temp, values, out_cccl, starts, ends, _add, n_segments, h_init,
                )

            _, cccl_make_warm = _time_ms(cccl_make_seg_reduce)
        except Exception:
            pass  # Fall back to one-shot timing

    _sync()
    match = bool(cp.allclose(out_cupy, out_cccl, atol=1e-6))

    return SegmentedResult(
        primitive="segmented_reduce",
        scale=actual_scale,
        segments=n_segments,
        cupy_workaround_warm_ms=round(cupy_warm, 3),
        cccl_cold_ms=round(cccl_cold, 3),
        cccl_warm_ms=round(cccl_warm, 3),
        cccl_make_warm_ms=round(cccl_make_warm, 3),
        results_match=match,
        cccl_make_vs_cupy_ratio=round(cccl_make_warm / max(cupy_warm, 0.001), 2),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_BENCHMARKS = {
    "exclusive_scan": _bench_exclusive_scan,
    "compaction": _bench_compaction,
    "reduction": _bench_reduction,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="CuPy vs CCCL primitive benchmark")
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[10_000, 100_000, 1_000_000, 10_000_000],
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument(
        "--primitives",
        nargs="+",
        default=list(_BENCHMARKS.keys()) + ["segmented_reduce"],
        help="Which primitives to benchmark",
    )
    args = parser.parse_args()
    _require_gpu()

    all_results: list[dict] = []

    for primitive in args.primitives:
        for scale in args.scales:
            if primitive == "segmented_reduce":
                result = _bench_segmented_reduce(scale)
                all_results.append(asdict(result))
                if not args.json:
                    print(
                        f"{result.primitive:>20s}  "
                        f"n={result.scale:>10,d}  "
                        f"segs={result.segments:>8,d}  "
                        f"cupy_wkrnd={result.cupy_workaround_warm_ms:>8.3f}ms  "
                        f"cccl_cold={result.cccl_cold_ms:>8.3f}ms  "
                        f"cccl_warm={result.cccl_warm_ms:>8.3f}ms  "
                        f"cccl_make={result.cccl_make_warm_ms:>8.3f}ms  "
                        f"ratio={result.cccl_make_vs_cupy_ratio:>5.2f}x  "
                        f"match={result.results_match}"
                    )
            elif primitive in _BENCHMARKS:
                result = _BENCHMARKS[primitive](scale)
                all_results.append(asdict(result))
                if not args.json:
                    print(
                        f"{result.primitive:>20s}  "
                        f"n={result.scale:>10,d}  "
                        f"cupy_cold={result.cupy_cold_ms:>8.3f}ms  "
                        f"cupy_warm={result.cupy_warm_ms:>8.3f}ms  "
                        f"cccl_cold={result.cccl_cold_ms:>8.3f}ms  "
                        f"cccl_warm={result.cccl_warm_ms:>8.3f}ms  "
                        f"cccl_make={result.cccl_make_warm_ms:>8.3f}ms  "
                        f"warm_winner={result.winner_warm:>10s}  "
                        f"make_winner={result.winner_make:>10s}  "
                        f"match={result.cupy_matches_cccl}"
                    )

    if args.json:
        print(json.dumps(all_results, indent=2))
    elif not args.json:
        print()
        print("Legend:")
        print("  cupy_cold/warm  = CuPy built-in (first call / best-of-7)")
        print("  cccl_cold/warm  = CCCL one-shot (first call / best-of-7)")
        print("  cccl_make       = CCCL make_* reusable callable (best-of-7)")
        print("  warm_winner     = faster between cupy_warm and cccl_warm")
        print("  make_winner     = faster between cupy_warm and cccl_make_warm")
        print("  ratio           = cccl_time / cupy_time (< 1.0 = CCCL wins)")


if __name__ == "__main__":
    main()
