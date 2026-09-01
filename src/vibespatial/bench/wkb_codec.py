"""Benchmark adapter for the device WKB codec.

Public benchmark scripts call this adapter so kernel selection remains inside
the benchmark package instead of forcing a private runtime path at the CLI.
"""

from __future__ import annotations

from vibespatial.kernels.core.wkb_decode import (
    decode_wkb_device_pipeline,
    scan_wkb_device_structural_plan,
)


def scan_device_wkb(payload, offsets, row_count: int):
    """Run the structural phase for a codec microbenchmark."""
    return scan_wkb_device_structural_plan(payload, offsets, row_count)


def decode_device_wkb(payload, offsets, row_count: int):
    """Run the complete device codec for a codec microbenchmark."""
    return decode_wkb_device_pipeline(payload, offsets, row_count)
