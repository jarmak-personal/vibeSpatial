#!/usr/bin/env python3
"""Time every CCCL and NVRTC compilation during precompile_all.

Usage:
    # Warm run (uses disk caches):
    uv run python scripts/time_precompile.py

    # Cold run (clears all caches first):
    uv run python scripts/time_precompile.py --cold
"""
from __future__ import annotations

import argparse
import shutil
import sys
from time import perf_counter


def _clear_caches() -> None:
    """Remove both NVRTC and CCCL disk caches so every compile is cold."""
    from vibespatial.cuda._runtime import _get_cache_dir as nvrtc_cache_dir
    from vibespatial.cuda.cccl_cubin_cache import _get_cache_dir as cccl_cache_dir

    for label, get_dir in [("NVRTC", nvrtc_cache_dir), ("CCCL", cccl_cache_dir)]:
        d = get_dir()
        if d.exists():
            n = sum(1 for _ in d.iterdir())
            shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            print(f"  Cleared {label} cache: {d}  ({n} files removed)")
        else:
            print(f"  {label} cache dir does not exist: {d}")


def _run_serial(args: argparse.Namespace) -> None:
    """Compile every CCCL spec one-by-one on the main thread."""
    from vibespatial.cuda.cccl_precompile import CCCLPrecompiler, SPEC_REGISTRY

    comp = CCCLPrecompiler.get()
    results = []
    print("Running CCCL compilations serially on main thread ...\n")
    t0_all = perf_counter()
    for name, spec in SPEC_REGISTRY.items():
        t0 = perf_counter()
        try:
            comp._compile_one(spec)
            elapsed = (perf_counter() - t0) * 1000.0
            results.append((name, elapsed, True, ""))
        except Exception as e:
            elapsed = (perf_counter() - t0) * 1000.0
            results.append((name, elapsed, False, str(e)))
    wall = (perf_counter() - t0_all) * 1000.0

    results.sort(key=lambda r: r[1], reverse=True)
    print("=" * 64)
    print("CCCL COMPILATIONS (serial, main thread)")
    print("=" * 64)
    print(f"{'Name':<40} {'Time (ms)':>10} {'OK':>4}")
    print("-" * 64)
    for name, ms, ok, err in results:
        tag = "yes" if ok else "NO"
        print(f"{name:<40} {ms:>10.1f} {tag:>4}")
        if not ok:
            print(f"  ERROR: {err}")
    total = sum(r[1] for r in results)
    print("-" * 64)
    print(f"{'Total':<40} {total:>10.1f}  ({len(results)} units)")
    print(f"{'Wall time':<40} {wall:>10.1f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Time precompile_all compilations")
    parser.add_argument(
        "--cold", action="store_true",
        help="Clear disk caches before running so every unit compiles from scratch",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Timeout in seconds for precompile_all (default: 300)",
    )
    parser.add_argument(
        "--serial", action="store_true",
        help="Compile each CCCL spec sequentially on the main thread (no threadpool)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cccl-only", action="store_true", help="Only run CCCL compilations")
    group.add_argument("--nvrtc-only", action="store_true", help="Only run NVRTC compilations")
    args = parser.parse_args()

    if args.cold:
        print("Clearing caches...")
        _clear_caches()
        print()

    if args.serial:
        _run_serial(args)
        return

    import importlib

    from vibespatial.cuda.cccl_precompile import (
        SPEC_REGISTRY,
        _NVRTC_CONSUMER_MODULES,
        ensure_pipelines_warm,
        precompile_status,
        request_warmup,
    )
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        print("Error: no GPU runtime available")
        sys.exit(1)

    run_cccl = not args.nvrtc_only
    run_nvrtc = not args.cccl_only
    label = "CCCL only" if args.cccl_only else "NVRTC only" if args.nvrtc_only else "all"
    print(f"Running precompile ({label}) ...")

    t0 = perf_counter()
    if run_cccl:
        request_warmup(list(SPEC_REGISTRY.keys()))
    if run_nvrtc:
        for mod_name in _NVRTC_CONSUMER_MODULES:
            try:
                importlib.import_module(mod_name)
            except Exception:
                pass
    cold = ensure_pipelines_warm(timeout=args.timeout)
    wall = (perf_counter() - t0) * 1000.0

    status = precompile_status()

    cccl_total = 0.0
    cccl_units = []
    nvrtc_total = 0.0
    nvrtc_units = []

    # ---- CCCL table ----
    if run_cccl:
        cccl = status.get("cccl", {})
        cccl_units = cccl.get("per_primitive", [])
        cccl_units.sort(key=lambda u: u["ms"], reverse=True)

        print("=" * 64)
        print("CCCL COMPILATIONS")
        print("=" * 64)
        print(f"{'Name':<40} {'Time (ms)':>10} {'OK':>4}")
        print("-" * 64)
        for u in cccl_units:
            ok = "yes" if u["ok"] else "NO"
            print(f"{u['name']:<40} {u['ms']:>10.1f} {ok:>4}")
        cccl_total = sum(u["ms"] for u in cccl_units)
        print("-" * 64)
        print(f"{'Total':<40} {cccl_total:>10.1f}  ({len(cccl_units)} units)")
        print()

    # ---- NVRTC table ----
    if run_nvrtc:
        nvrtc = status.get("nvrtc", {})
        nvrtc_units = nvrtc.get("per_unit", [])
        nvrtc_units.sort(key=lambda u: u["ms"], reverse=True)

        print("=" * 64)
        print("NVRTC COMPILATIONS")
        print("=" * 64)
        print(f"{'Key':<40} {'Time (ms)':>10} {'OK':>4}")
        print("-" * 64)
        for u in nvrtc_units:
            ok = "yes" if u["ok"] else "NO"
            key = u["key"]
            if len(key) > 39:
                key = key[:36] + "..."
            print(f"{key:<40} {u['ms']:>10.1f} {ok:>4}")
        nvrtc_total = sum(u["ms"] for u in nvrtc_units)
        print("-" * 64)
        print(f"{'Total':<40} {nvrtc_total:>10.1f}  ({len(nvrtc_units)} units)")
        print()

    # ---- Summary ----
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  Wall time:       {wall:>10.1f} ms")
    if run_cccl:
        print(f"  CCCL units:      {len(cccl_units):>10}")
        print(f"  CCCL sum:        {cccl_total:>10.1f} ms")
        cccl_st = status.get("cccl", {})
        print(f"  CCCL deferred:   {cccl_st.get('deferred', 0):>10}  (disk cache hits)")
    if run_nvrtc:
        print(f"  NVRTC units:     {len(nvrtc_units):>10}")
        print(f"  NVRTC sum:       {nvrtc_total:>10.1f} ms")
        nvrtc_st = status.get("nvrtc", {})
        print(f"  NVRTC deferred:  {nvrtc_st.get('deferred', 0):>10}  (disk cache hits)")
    cold_cccl = cold.get("cccl_cold", [])
    cold_nvrtc = cold.get("nvrtc_cold", [])
    if cold_cccl:
        print(f"  CCCL cold/failed: {cold_cccl}")
    if cold_nvrtc:
        print(f"  NVRTC cold/failed: {cold_nvrtc}")
    print()


if __name__ == "__main__":
    main()
