"""GPU text-parsing primitives for structured format readers.

This package provides composable, format-agnostic building blocks for
GPU-accelerated parsing of structured text formats (GeoJSON, WKT, CSV,
KML, etc.).  Each primitive maps to one or more NVRTC kernels that
operate on device-resident byte arrays.

Modules
-------
structural
    Quote-state and bracket-depth computation.
numeric
    Number boundary detection and ASCII-to-number conversion.
pattern
    Byte-pattern matching and span detection.

Typical pipeline
----------------
A GPU text parser composes these primitives in sequence::

    d_bytes = read_file_to_device(path)

    # Stage 1: structural analysis
    d_qp = quote_parity(d_bytes)
    d_depth = bracket_depth(d_bytes, d_qp)

    # Stage 2: locate structural markers
    d_hits = pattern_match(d_bytes, b'"coordinates":', d_qp)
    d_positions = cp.flatnonzero(d_hits).astype(cp.int64)

    # Stage 3: define value spans
    d_span_ends = span_boundaries(d_depth, d_positions, len(d_bytes),
                                  skip_bytes=14)
    d_mask = mark_spans(d_positions + 14, d_span_ends, len(d_bytes))

    # Stage 4: extract numbers within spans
    d_is_start, d_is_end = number_boundaries(d_bytes, d_qp)
    d_starts, d_ends = extract_number_positions(d_is_start, d_is_end,
                                                 d_mask=d_mask)
    d_values = parse_ascii_floats(d_bytes, d_starts, d_ends)

All operations run on the GPU with zero host materialization until
the caller explicitly requests results via ``.get()`` or
``cp.asnumpy()``.
"""
from __future__ import annotations

from vibespatial.io.gpu_parse.numeric import (
    extract_number_positions,
    number_boundaries,
    parse_ascii_floats,
    parse_ascii_ints,
)
from vibespatial.io.gpu_parse.pattern import (
    mark_spans,
    pattern_match,
    span_boundaries,
)
from vibespatial.io.gpu_parse.structural import (
    bracket_depth,
    quote_parity,
)

# Lazy import to avoid GPU init at package import time for indexing
# (it registers NVRTC warmup at module scope)

__all__ = [
    # structural
    "quote_parity",
    "bracket_depth",
    # numeric
    "number_boundaries",
    "parse_ascii_floats",
    "parse_ascii_ints",
    "extract_number_positions",
    # pattern
    "pattern_match",
    "span_boundaries",
    "mark_spans",
    # indexing
    "build_spatial_index",
    "build_index_from_reader",
    "GpuSpatialIndex",
]


def __getattr__(name: str):
    if name in ("build_spatial_index", "build_index_from_reader", "GpuSpatialIndex"):
        from vibespatial.io.gpu_parse.indexing import (
            GpuSpatialIndex,
            build_index_from_reader,
            build_spatial_index,
        )
        _exports = {
            "build_spatial_index": build_spatial_index,
            "build_index_from_reader": build_index_from_reader,
            "GpuSpatialIndex": GpuSpatialIndex,
        }
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
