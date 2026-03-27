"""GPU pattern matching and span detection primitives.

Provides byte-pattern search with optional quote-state filtering,
depth-based span boundary detection, and region mask generation.
These primitives enable GPU parsers to locate structural markers
(JSON keys, WKT keywords, XML tags) and define the byte ranges
they govern.

Typical pipeline:

1. ``pattern_match`` — find all occurrences of a byte pattern
2. ``span_boundaries`` — for each match, scan depth to find the end
3. ``mark_spans`` — create a per-byte region mask from start/end pairs

All functions operate on device-resident arrays with zero host
materialization.
"""
from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
    make_kernel_cache_key,
)

if TYPE_CHECKING:
    import cupy as cp

# ctypes for int64 kernel params (matches geojson_gpu.py convention)
KERNEL_PARAM_I64 = ctypes.c_longlong

# ---------------------------------------------------------------------------
# pattern_match — runtime-generated kernel, cached by pattern bytes
# ---------------------------------------------------------------------------

# Module-level cache: pattern bytes -> compiled kernel dict
_pattern_kernel_cache: dict[bytes, dict] = {}


def _generate_pattern_match_source(
    pattern: bytes, check_quote: bool, quote_check_offset: int
) -> str:
    """Generate NVRTC source for a specific byte pattern.

    The pattern bytes are embedded as a compile-time constant array
    in the generated CUDA source, enabling the compiler to optimize
    the comparison loop (unrolling, constant propagation).
    """
    pat_len = len(pattern)
    # Build the compile-time constant array initializer
    pat_init = ", ".join(str(b) for b in pattern)

    # Build the quote-parity check clause
    if check_quote:
        quote_param = (
            "    const unsigned char* __restrict__ quote_parity,\n"
        )
        quote_check = (
            f"    if (match && quote_parity[idx + {quote_check_offset}] != 0) {{\n"
            f"        match = 0;\n"
            f"    }}\n"
        )
    else:
        quote_param = ""
        quote_check = ""

    source = (
        f'extern "C" __global__ void pattern_match_kernel(\n'
        f"    const unsigned char* __restrict__ input,\n"
        f"{quote_param}"
        f"    unsigned char* __restrict__ hits,\n"
        f"    long long n\n"
        f") {{\n"
        f"    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
        f"    if (idx > n - {pat_len}) {{\n"
        f"        if (idx < n) hits[idx] = 0;\n"
        f"        return;\n"
        f"    }}\n"
        f"\n"
        f"    const unsigned char pat[{pat_len}] = {{{pat_init}}};\n"
        f"\n"
        f"    unsigned char match = 1;\n"
        f"    for (int i = 0; i < {pat_len}; ++i) {{\n"
        f"        if (input[idx + i] != pat[i]) {{ match = 0; break; }}\n"
        f"    }}\n"
        f"{quote_check}"
        f"    hits[idx] = match;\n"
        f"}}\n"
    )
    return source


_PATTERN_MATCH_NAMES = ("pattern_match_kernel",)


def _compile_pattern_match(
    pattern: bytes, check_quote: bool, quote_check_offset: int
) -> dict:
    """Compile (or retrieve from cache) a pattern-specific kernel."""
    # Cache key incorporates the full pattern bytes and quote-check config
    cache_key_tuple = (pattern, check_quote, quote_check_offset)
    cached = _pattern_kernel_cache.get(cache_key_tuple)
    if cached is not None:
        return cached

    source = _generate_pattern_match_source(
        pattern, check_quote, quote_check_offset
    )
    # Use pattern hex in the NVRTC cache key for uniqueness
    pat_hex = pattern.hex()
    qc_suffix = f"-qc{quote_check_offset}" if check_quote else "-noqc"
    prefix = f"pattern-match-{pat_hex}{qc_suffix}"
    runtime = get_cuda_runtime()
    nvrtc_cache_key = make_kernel_cache_key(prefix, source)
    kernels = runtime.compile_kernels(
        cache_key=nvrtc_cache_key,
        source=source,
        kernel_names=_PATTERN_MATCH_NAMES,
    )
    _pattern_kernel_cache[cache_key_tuple] = kernels
    return kernels


# ---------------------------------------------------------------------------
# span_boundaries — static kernel source
# ---------------------------------------------------------------------------

_SPAN_BOUNDARIES_SOURCE = r"""
extern "C" __global__ void span_boundaries_kernel(
    const int* __restrict__ depth,
    const long long* __restrict__ starts,
    long long* __restrict__ ends,
    int n_spans,
    long long n_bytes,
    int skip_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_spans) return;

    // Start scanning after skip_bytes past the start position
    long long pos = starts[idx] + (long long)skip_bytes;
    // Skip whitespace: advance while depth does not change
    while (pos < n_bytes && depth[pos] == depth[pos - 1]) {
        pos++;
    }
    if (pos >= n_bytes) {
        ends[idx] = n_bytes;
        return;
    }
    int start_depth = depth[pos];
    // Scan forward until depth drops below start_depth
    pos++;
    while (pos < n_bytes && depth[pos] >= start_depth) {
        pos++;
    }
    ends[idx] = pos;
}
"""

_SPAN_BOUNDARIES_NAMES = ("span_boundaries_kernel",)


# ---------------------------------------------------------------------------
# mark_spans — static kernel source
# ---------------------------------------------------------------------------

_MARK_SPANS_SOURCE = r"""
extern "C" __global__ void mark_spans_kernel(
    const long long* __restrict__ starts,
    const long long* __restrict__ ends,
    unsigned char* __restrict__ mask,
    int n_spans
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_spans) return;

    long long start = starts[idx];
    long long end = ends[idx];
    for (long long i = start; i < end; i++) {
        mask[i] = 1;
    }
}
"""

_MARK_SPANS_NAMES = ("mark_spans_kernel",)


# ---------------------------------------------------------------------------
# NVRTC warmup (ADR-0034 Level 2)
# ---------------------------------------------------------------------------
# Only the static kernels can be warmed up at module scope.
# pattern_match kernels are generated per-pattern and compiled on demand.

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("parse-span-boundaries", _SPAN_BOUNDARIES_SOURCE, _SPAN_BOUNDARIES_NAMES),
    ("parse-mark-spans", _MARK_SPANS_SOURCE, _MARK_SPANS_NAMES),
])


# ---------------------------------------------------------------------------
# Kernel compilation helpers (static kernels)
# ---------------------------------------------------------------------------

def _span_boundaries_kernels():
    return compile_kernel_group(
        "parse-span-boundaries", _SPAN_BOUNDARIES_SOURCE, _SPAN_BOUNDARIES_NAMES
    )


def _mark_spans_kernels():
    return compile_kernel_group(
        "parse-mark-spans", _MARK_SPANS_SOURCE, _MARK_SPANS_NAMES
    )


# ---------------------------------------------------------------------------
# Launch helper
# ---------------------------------------------------------------------------

def _launch_kernel(runtime, kernel, n, params):
    """Launch a kernel with occupancy-based grid/block sizing."""
    grid, block = runtime.launch_config(kernel, int(n))
    runtime.launch(kernel, grid=grid, block=block, params=params)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pattern_match(
    d_bytes: cp.ndarray,
    pattern: bytes,
    d_quote_parity: cp.ndarray | None = None,
    *,
    quote_check_offset: int = -1,
) -> cp.ndarray:
    """Find all occurrences of a byte pattern in the input.

    A per-byte GPU kernel tests whether the substring starting at
    each position matches the given pattern.  Optionally validates
    that the match is outside a quoted string by checking the
    quote parity at a specific offset within the pattern.

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 array of raw file bytes, shape ``(n,)``.
    pattern : bytes
        The byte pattern to search for.  Must be non-empty.  Maximum
        length 256 bytes.  The pattern is compiled into the NVRTC
        kernel as a constant array for optimal access.
    d_quote_parity : cp.ndarray or None, default None
        Device-resident uint8 parity mask from
        ``structural.quote_parity()``, shape ``(n,)``.  When provided,
        matches inside quoted strings are suppressed based on the
        ``quote_check_offset`` parameter.  When ``None``, no quote
        filtering is applied.
    quote_check_offset : int, default -1
        Byte offset within the pattern at which to check quote parity.
        A match is suppressed if ``d_quote_parity[pos + quote_check_offset] != 0``.
        A value of ``-1`` means: use the last byte of the pattern
        (``len(pattern) - 1``).

        For JSON key patterns like ``"coordinates":``, the check offset
        should point to the colon (last byte), because inside a real
        key the opening and closing quotes cancel to parity 0.  Inside
        a string value, parity would be 1 (odd), suppressing the match.

    Returns
    -------
    cp.ndarray
        Device-resident uint8 array, shape ``(n,)``.  Element ``i`` is
        1 if the pattern matches starting at byte offset ``i``, else 0.
        Positions where the pattern would extend past the end of the
        input are always 0.

    Notes
    -----
    This is a generalization of the ``find_coord_key`` and
    ``find_type_key`` kernels from the GeoJSON parser.  Those search
    for specific 14-byte and 7-byte patterns respectively.  This
    function parameterizes the pattern and supports arbitrary lengths.

    The kernel is generated at runtime via NVRTC with the pattern
    bytes embedded as a compile-time constant.  A kernel cache keyed
    on the pattern bytes avoids redundant compilations.

    For multi-criteria matching (e.g., pattern match AND depth check),
    combine the output with depth-based filtering after the call:

    .. code-block:: python

        hits = pattern_match(d_bytes, b'"type":', d_qp)
        # Further filter by depth
        hits = hits * (d_depth == 4).view(cp.uint8)

    Examples
    --------
    >>> # Input: {"coordinates": [1,2], "coord": 3}
    >>> # Pattern: b'"coordinates":'
    >>> # Result: 1 at position 1, 0 elsewhere
    """
    import cupy as cp

    if not pattern:
        raise ValueError("pattern must be non-empty")
    if len(pattern) > 256:
        raise ValueError(f"pattern length {len(pattern)} exceeds maximum of 256 bytes")

    n = d_bytes.shape[0]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    # Resolve default quote_check_offset
    check_quote = d_quote_parity is not None
    if quote_check_offset == -1:
        quote_check_offset = len(pattern) - 1

    # Compile (or retrieve cached) pattern-specific kernel
    kernels = _compile_pattern_match(pattern, check_quote, quote_check_offset)
    kernel = kernels["pattern_match_kernel"]

    # Allocate output (zero-filled for tail positions)
    d_hits = cp.zeros(n, dtype=cp.uint8)

    n_i64 = np.int64(n)

    if check_quote:
        params = (
            (ptr(d_bytes), ptr(d_quote_parity), ptr(d_hits), n_i64),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I64),
        )
    else:
        params = (
            (ptr(d_bytes), ptr(d_hits), n_i64),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I64),
        )

    _launch_kernel(runtime, kernel, n, params)
    return d_hits


def span_boundaries(
    d_depth: cp.ndarray,
    d_starts: cp.ndarray,
    n_bytes: int,
    *,
    skip_bytes: int = 0,
) -> cp.ndarray:
    """Find span end positions by scanning bracket depth.

    For each start position, scans forward through the depth array to
    find the byte position where the nesting depth drops below the
    depth at the scan start.  This identifies the end of a
    bracket-delimited span (e.g., the closing ``]`` of a JSON
    ``"coordinates"`` array, the closing ``)`` of a WKT geometry).

    Parameters
    ----------
    d_depth : cp.ndarray
        Device-resident int32 depth array from
        ``structural.bracket_depth()``, shape ``(n_bytes,)``.
    d_starts : cp.ndarray
        Device-resident int64 array of span start positions, shape
        ``(n_spans,)``.  Each position is the byte offset of the
        structural marker that begins the span (e.g., the first byte
        of ``"coordinates":``).
    n_bytes : int
        Total number of bytes in the input.  Used as the upper bound
        for forward scanning.
    skip_bytes : int, default 0
        Number of bytes to skip past each start position before
        beginning the depth scan.  For example, when starting from
        the position of ``"coordinates":``, ``skip_bytes=14`` skips
        past the key pattern to reach the opening bracket of the
        value.

        The scan then advances through any whitespace to find the
        first bracket, records its depth, and continues until depth
        drops below that level.

    Returns
    -------
    cp.ndarray
        Device-resident int64 array, shape ``(n_spans,)``.  Element
        ``i`` is the byte offset one past the closing bracket of the
        ``i``-th span.  If the closing bracket is not found before
        ``n_bytes``, the value is ``n_bytes``.

    Notes
    -----
    This is a generalization of the ``coord_span_end`` kernel from
    the GeoJSON parser.  That kernel hard-codes ``skip_bytes=14``
    for the ``"coordinates":`` pattern length.

    The scan algorithm:

    1. Start at ``d_starts[i] + skip_bytes``
    2. Skip forward while depth does not change (whitespace between
       key and opening bracket)
    3. Record ``start_depth = d_depth[pos]`` at the opening bracket
    4. Scan forward while ``d_depth[pos] >= start_depth``
    5. Return ``pos`` (one past the closing bracket)

    Examples
    --------
    >>> # Input: "coordinates": [[1,2],[3,4]]
    >>> #        ^pos=0          ^depth=5      ^end
    >>> # d_starts = [0], skip_bytes = 14
    >>> # Result: [end_position]
    """
    import cupy as cp

    n_spans = d_starts.shape[0]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    kernels = _span_boundaries_kernels()
    kernel = kernels["span_boundaries_kernel"]

    d_ends = cp.empty(n_spans, dtype=cp.int64)

    params = (
        (
            ptr(d_depth),
            ptr(d_starts),
            ptr(d_ends),
            np.int32(n_spans),
            np.int64(n_bytes),
            np.int32(skip_bytes),
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I64,
            KERNEL_PARAM_I32,
        ),
    )

    _launch_kernel(runtime, kernel, n_spans, params)
    return d_ends


def mark_spans(
    d_starts: cp.ndarray,
    d_ends: cp.ndarray,
    n_bytes: int,
) -> cp.ndarray:
    """Create a per-byte region mask from start/end position pairs.

    For each ``(d_starts[i], d_ends[i])`` pair, sets all bytes in the
    half-open range ``[d_starts[i], d_ends[i])`` to 1 in the output
    mask.  All other positions are 0.

    This is used to create coordinate-span masks that filter number
    detection to only relevant regions of the file.

    Parameters
    ----------
    d_starts : cp.ndarray
        Device-resident int64 array of span start positions, shape
        ``(n_spans,)``.  Each element is an inclusive byte offset.
    d_ends : cp.ndarray
        Device-resident int64 array of span end positions, shape
        ``(n_spans,)``.  Each element is an exclusive byte offset.
    n_bytes : int
        Total number of bytes in the input.  The output mask has
        this length.

    Returns
    -------
    cp.ndarray
        Device-resident uint8 array, shape ``(n_bytes,)``.  Element
        ``i`` is 1 if byte ``i`` falls within any span, else 0.
        Overlapping spans are handled correctly (union semantics).

    Notes
    -----
    This is a generalization of the ``mark_coord_spans`` kernel from
    the GeoJSON parser.  That kernel reads start positions from
    ``coord_positions`` and offsets them by 14 bytes (the length of
    ``"coordinates":``).  This function takes pre-computed start/end
    arrays directly.

    The kernel launches one thread per span (not per byte).  Each
    thread writes 1 to all bytes in its span via a serial loop.
    For large numbers of short spans, this is efficient because the
    write pattern is coalesced within each span.  For very large
    spans (>1M bytes each), a per-byte kernel with binary search
    over sorted starts would be more efficient, but in practice
    coordinate spans are small relative to file size.

    Examples
    --------
    >>> # d_starts = [10, 50], d_ends = [25, 60], n_bytes = 100
    >>> # Result: 0s except positions [10..24] and [50..59] are 1
    """
    import cupy as cp

    n_spans = d_starts.shape[0]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    kernels = _mark_spans_kernels()
    kernel = kernels["mark_spans_kernel"]

    # Zero-filled: positions outside all spans stay 0
    d_mask = cp.zeros(n_bytes, dtype=cp.uint8)

    if n_spans == 0:
        return d_mask

    params = (
        (
            ptr(d_starts),
            ptr(d_ends),
            ptr(d_mask),
            np.int32(n_spans),
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )

    _launch_kernel(runtime, kernel, n_spans, params)
    return d_mask
