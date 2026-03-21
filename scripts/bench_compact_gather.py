"""Benchmark compact+gather patterns: CuPy vs CCCL vs NVRTC gather.

Compares three approaches for the simplify pipeline's compact+gather step:
  1. CuPy:       cp.flatnonzero + CuPy fancy indexing  (current simplify code)
  2. CCCL+CuPy:  compact_indices(AUTO=CuPy) + CuPy fancy indexing
  3. CuPy+NVRTC: cp.flatnonzero + custom NVRTC gather kernel

Uses cp.cuda.Event pairs for GPU-accurate timing (not wall clock).
Reports median over 50 iterations after 10 warmup reps per scale.

Usage:
    uv run python scripts/bench_compact_gather.py
"""
from __future__ import annotations

import statistics
import sys

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    print("ERROR: CuPy not available. Requires GPU.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# NVRTC gather kernel source
# ---------------------------------------------------------------------------

_GATHER_KERNEL_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(256, 4)
gather_f64(
    const double* __restrict__ src,
    const int* __restrict__ indices,
    double* __restrict__ dst,
    int n
) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = src[indices[i]];
    }
}
"""

_GATHER_KERNEL_NAMES = ("gather_f64",)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync():
    cp.cuda.Stream.null.synchronize()


def _gpu_time_us(fn, *, warmup: int = 10, repeat: int = 50) -> list[float]:
    """Time *fn* using CUDA events, return list of durations in microseconds.

    Runs *warmup* untimed iterations first to stabilize GPU clocks and
    cache state, then times *repeat* iterations.
    """
    # Warmup (untimed)
    for _ in range(warmup):
        _sync()
        fn()
        _sync()

    # Timed iterations
    times = []
    for _ in range(repeat):
        _sync()
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        fn()
        end.record()
        end.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(start, end)
        times.append(elapsed_ms * 1000.0)  # ms -> us
    return times


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _p10_p90(values: list[float]) -> tuple[float, float]:
    s = sorted(values)
    n = len(s)
    return s[max(0, n // 10)], s[min(n - 1, n * 9 // 10)]


# ---------------------------------------------------------------------------
# Approach 1: Pure CuPy compact + gather
# ---------------------------------------------------------------------------

def cupy_compact_gather(d_keep, d_x, d_y):
    d_kept_indices = cp.flatnonzero(d_keep).astype(cp.int32)
    d_x_out = d_x[d_kept_indices]
    d_y_out = d_y[d_kept_indices]
    return d_x_out, d_y_out, d_kept_indices


# ---------------------------------------------------------------------------
# Approach 2: CCCL compact_indices + CuPy gather
# ---------------------------------------------------------------------------

def cccl_compact_cupy_gather(d_keep, d_x, d_y):
    from vibespatial.cuda.cccl_primitives import compact_indices
    result = compact_indices(d_keep)
    d_kept_indices = result.values[:result.count]
    d_x_out = d_x[d_kept_indices]
    d_y_out = d_y[d_kept_indices]
    return d_x_out, d_y_out, d_kept_indices


# ---------------------------------------------------------------------------
# Approach 3: CuPy compact + NVRTC gather
# ---------------------------------------------------------------------------

_gather_kernel_cache = {}


def _get_gather_kernel():
    if "kernel" not in _gather_kernel_cache:
        from vibespatial.cuda._runtime import compile_kernel_group
        kernels = compile_kernel_group(
            "bench-gather-f64", _GATHER_KERNEL_SOURCE, _GATHER_KERNEL_NAMES,
        )
        _gather_kernel_cache["kernel"] = kernels["gather_f64"]
    return _gather_kernel_cache["kernel"]


def nvrtc_gather(d_src, d_indices, n):
    """Launch NVRTC gather kernel, return output array."""
    from vibespatial.cuda._runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
    )
    runtime = get_cuda_runtime()
    kernel = _get_gather_kernel()
    d_out = cp.empty(n, dtype=cp.float64)
    ptr = runtime.pointer
    params = (
        (ptr(d_src), ptr(d_indices), ptr(d_out), n),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
    )
    grid, block = runtime.launch_config(kernel, n)
    runtime.launch(kernel, grid=grid, block=block, params=params)
    return d_out


def cupy_compact_nvrtc_gather(d_keep, d_x, d_y):
    d_kept_indices = cp.flatnonzero(d_keep).astype(cp.int32)
    n = int(d_kept_indices.shape[0])
    if n == 0:
        return cp.empty(0, dtype=cp.float64), cp.empty(0, dtype=cp.float64), d_kept_indices
    d_x_out = nvrtc_gather(d_x, d_kept_indices, n)
    d_y_out = nvrtc_gather(d_y, d_kept_indices, n)
    return d_x_out, d_y_out, d_kept_indices


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def _verify_results(d_keep, d_x, d_y):
    """Verify all three approaches produce identical output."""
    x1, y1, idx1 = cupy_compact_gather(d_keep, d_x, d_y)
    _sync()
    x2, y2, idx2 = cccl_compact_cupy_gather(d_keep, d_x, d_y)
    _sync()
    x3, y3, idx3 = cupy_compact_nvrtc_gather(d_keep, d_x, d_y)
    _sync()

    np.testing.assert_array_equal(cp.asnumpy(idx1), cp.asnumpy(idx2))
    np.testing.assert_array_equal(cp.asnumpy(idx1), cp.asnumpy(idx3))
    np.testing.assert_array_equal(cp.asnumpy(x1), cp.asnumpy(x2))
    np.testing.assert_array_equal(cp.asnumpy(x1), cp.asnumpy(x3))
    np.testing.assert_array_equal(cp.asnumpy(y1), cp.asnumpy(y2))
    np.testing.assert_array_equal(cp.asnumpy(y1), cp.asnumpy(y3))


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

SCALES = [10_000, 100_000, 1_000_000]
KEEP_RATIO = 0.6
WARMUP = 10
REPEAT = 50


def run_benchmark():
    print("=" * 80)
    print("Compact+Gather Benchmark: CuPy vs CCCL vs NVRTC")
    print("=" * 80)
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"GPU: {gpu_name}")
    print(f"Keep ratio: {KEEP_RATIO:.0%}   |   Warmup: {WARMUP}   |   Repeat: {REPEAT} (median)")
    print()

    # Global warmup: compile NVRTC kernel + warm CuPy/CCCL code paths
    print("Compiling and warming up ... ", end="", flush=True)
    d_warmup_keep = cp.random.random(1000) < KEEP_RATIO
    d_warmup_x = cp.random.random(1000)
    d_warmup_y = cp.random.random(1000)
    for _ in range(3):
        cupy_compact_gather(d_warmup_keep, d_warmup_x, d_warmup_y)
        _sync()
        cccl_compact_cupy_gather(d_warmup_keep, d_warmup_x, d_warmup_y)
        _sync()
        cupy_compact_nvrtc_gather(d_warmup_keep, d_warmup_x, d_warmup_y)
        _sync()
    print("done.")

    # Verify correctness at small scale
    print("Verifying correctness ... ", end="", flush=True)
    _verify_results(d_warmup_keep, d_warmup_x, d_warmup_y)
    print("passed.")
    print()

    # -----------------------------------------------------------------------
    # Part 1: Combined compact+gather timing
    # -----------------------------------------------------------------------
    print("PART 1: Combined compact+gather (end-to-end)")
    print(f"{'Scale':>10s}  "
          f"{'CuPy':>10s}  "
          f"{'CCCL+CuPy':>12s}  "
          f"{'CuPy+NVRTC':>12s}  "
          f"{'Fastest':>14s}")
    print("-" * 68)

    for scale in SCALES:
        # Generate test data on device with fixed seed
        rng = cp.random.default_rng(42)
        d_x = rng.standard_normal(scale, dtype=cp.float64)
        d_y = rng.standard_normal(scale, dtype=cp.float64)
        d_keep = rng.random(scale) < KEEP_RATIO
        _sync()

        # Bind via default args to avoid late-binding closure issues
        def _fn_cupy(k=d_keep, x=d_x, y=d_y):
            return cupy_compact_gather(k, x, y)

        def _fn_cccl(k=d_keep, x=d_x, y=d_y):
            return cccl_compact_cupy_gather(k, x, y)

        def _fn_nvrtc(k=d_keep, x=d_x, y=d_y):
            return cupy_compact_nvrtc_gather(k, x, y)

        t_cupy = _gpu_time_us(_fn_cupy, warmup=WARMUP, repeat=REPEAT)
        t_cccl = _gpu_time_us(_fn_cccl, warmup=WARMUP, repeat=REPEAT)
        t_nvrtc = _gpu_time_us(_fn_nvrtc, warmup=WARMUP, repeat=REPEAT)

        med_cupy = _median(t_cupy)
        med_cccl = _median(t_cccl)
        med_nvrtc = _median(t_nvrtc)

        results = {
            "CuPy": med_cupy,
            "CCCL+CuPy": med_cccl,
            "CuPy+NVRTC": med_nvrtc,
        }
        fastest = min(results, key=results.get)

        print(f"{scale:>10,d}  "
              f"{med_cupy:>8.1f}us  "
              f"{med_cccl:>10.1f}us  "
              f"{med_nvrtc:>10.1f}us  "
              f"{fastest:>14s}")

    print()

    # -----------------------------------------------------------------------
    # Part 2: Breakdown of compact vs gather at each scale
    # -----------------------------------------------------------------------
    print("PART 2: Component breakdown (compact-only vs gather-only)")
    print(f"{'Scale':>10s}  "
          f"{'compact':>10s}  "
          f"{'CuPy gath':>10s}  "
          f"{'NVRTC gath':>10s}  "
          f"{'Gath winner':>12s}  "
          f"{'Kept':>8s}")
    print("-" * 68)

    for scale in SCALES:
        rng = cp.random.default_rng(42)
        d_x = rng.standard_normal(scale, dtype=cp.float64)
        d_y = rng.standard_normal(scale, dtype=cp.float64)
        d_keep = rng.random(scale) < KEEP_RATIO
        _sync()

        # Time compact only
        def _compact(k=d_keep):
            return cp.flatnonzero(k).astype(cp.int32)

        t_compact = _gpu_time_us(_compact, warmup=WARMUP, repeat=REPEAT)
        med_compact = _median(t_compact)

        # Pre-compute indices for isolated gather timing
        d_indices = cp.flatnonzero(d_keep).astype(cp.int32)
        n_kept = int(d_indices.shape[0])
        _sync()

        # CuPy gather only
        def _cupy_gath(x=d_x, y=d_y, idx=d_indices):
            return x[idx], y[idx]

        t_cupy_gath = _gpu_time_us(_cupy_gath, warmup=WARMUP, repeat=REPEAT)
        med_cupy_gath = _median(t_cupy_gath)

        # NVRTC gather only
        def _nvrtc_gath(x=d_x, y=d_y, idx=d_indices, n=n_kept):
            return nvrtc_gather(x, idx, n), nvrtc_gather(y, idx, n)

        t_nvrtc_gath = _gpu_time_us(_nvrtc_gath, warmup=WARMUP, repeat=REPEAT)
        med_nvrtc_gath = _median(t_nvrtc_gath)

        gath_winner = "NVRTC" if med_nvrtc_gath < med_cupy_gath else "CuPy"

        print(f"{scale:>10,d}  "
              f"{med_compact:>8.1f}us  "
              f"{med_cupy_gath:>8.1f}us  "
              f"{med_nvrtc_gath:>8.1f}us  "
              f"{gath_winner:>12s}  "
              f"{n_kept:>7,d}")

    print()

    # -----------------------------------------------------------------------
    # Part 3: Note on CCCL compact_indices internal sync
    # -----------------------------------------------------------------------
    print("NOTE: compact_indices(AUTO=CuPy) includes an internal")
    print("  cp.cuda.Stream.null.synchronize() call (cccl_primitives.py:229).")
    print("  This serializes the pipeline within the event-timed section,")
    print("  adding overhead vs raw cp.flatnonzero. For simplify's use case,")
    print("  calling cp.flatnonzero directly avoids this overhead.")
    print()


if __name__ == "__main__":
    run_benchmark()
