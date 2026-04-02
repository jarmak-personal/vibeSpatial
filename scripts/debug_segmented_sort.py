#!/usr/bin/env python3
"""Reproduce segmented sort compilation failure with full traceback.

Usage:
    uv run python scripts/debug_segmented_sort.py
"""
from vibespatial.cuda.cccl_precompile import SPEC_REGISTRY, CCCLPrecompiler

for name in ["segmented_sort_asc_f64", "segmented_sort_asc_i32"]:
    print(f"Compiling {name} ...")
    spec = SPEC_REGISTRY[name]
    CCCLPrecompiler.get()._compile_one(spec)
    print(f"  OK\n")
