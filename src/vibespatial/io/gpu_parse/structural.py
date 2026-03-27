"""GPU structural scanning primitives for text parsing.

Provides quote-state detection and bracket-depth computation via
per-byte delta kernels and CuPy prefix sums.  These are the
foundational building blocks for any structured-text GPU parser:
the quote-parity mask identifies which bytes are inside string
literals, and the bracket-depth array encodes the hierarchical
nesting structure of the document.

Both functions operate on device-resident byte arrays and return
device-resident results with zero host materialization.
"""
from __future__ import annotations

import ctypes

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

# ctypes for int64 kernel params (files > 2 GB)
KERNEL_PARAM_I64 = ctypes.c_longlong

# ---------------------------------------------------------------------------
# Kernel sources (Tier 1 NVRTC) -- integer-only byte classification,
# no floating-point computation, so no PrecisionPlan needed.
# ---------------------------------------------------------------------------

_QUOTE_TOGGLE_SOURCE = r"""
extern "C" __global__ void quote_toggle(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    unsigned char b = input[idx];
    if (b != '"') {
        output[idx] = 0;
        return;
    }
    // Check for backslash escape: count consecutive backslashes before this quote
    int backslash_count = 0;
    long long j = idx - 1;
    while (j >= 0 && input[j] == '\\') {
        backslash_count++;
        j--;
    }
    // Quote is escaped if preceded by odd number of backslashes
    output[idx] = (backslash_count % 2 == 0) ? 1 : 0;
}
"""

_QUOTE_TOGGLE_NAMES: tuple[str, ...] = ("quote_toggle",)

# ---------------------------------------------------------------------------
# Bracket depth: parameterizable kernel source template.
# Open/close characters are baked into the NVRTC source at compile time
# to avoid passing variable-length data across the kernel interface.
# The compiled kernel is cached by (open_chars, close_chars) tuple.
# ---------------------------------------------------------------------------

_DEPTH_DELTAS_TEMPLATE = r"""
extern "C" __global__ void compute_depth_deltas(
    const unsigned char* __restrict__ input,
    const unsigned char* __restrict__ quote_parity,
    signed char* __restrict__ deltas,
    long long n
) {{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    signed char d = 0;
    if (quote_parity[idx] == 0) {{
        unsigned char b = input[idx];
        {open_checks}
        {close_checks}
    }}
    deltas[idx] = d;
}}
"""

_DEPTH_DELTAS_NAMES: tuple[str, ...] = ("compute_depth_deltas",)

# Cache of compiled depth-delta kernels keyed by (open_chars, close_chars).
_depth_kernel_cache: dict[tuple[str, str], dict[str, object]] = {}


def _build_depth_source(open_chars: str, close_chars: str) -> str:
    """Generate NVRTC source for the given bracket characters.

    Each character becomes a simple equality check.  For the common
    JSON case ``{[`` / ``}]`` the generated code is equivalent to
    the original hard-coded kernel in geojson_gpu.py.
    """
    # Build open checks: if (b == '{' || b == '[') d = 1;
    open_conds = " || ".join(f"b == '{c}'" for c in open_chars)
    open_checks = f"if ({open_conds}) d = 1;"

    # Build close checks: else if (b == '}' || b == ']') d = -1;
    close_conds = " || ".join(f"b == '{c}'" for c in close_chars)
    close_checks = f"else if ({close_conds}) d = -1;"

    return _DEPTH_DELTAS_TEMPLATE.format(
        open_checks=open_checks,
        close_checks=close_checks,
    )


def _get_depth_kernels(open_chars: str, close_chars: str) -> dict[str, object]:
    """Compile (or retrieve cached) depth-delta kernel for given brackets."""
    key = (open_chars, close_chars)
    cached = _depth_kernel_cache.get(key)
    if cached is not None:
        return cached
    source = _build_depth_source(open_chars, close_chars)
    # Cache key prefix encodes the bracket chars so different
    # parameterizations compile and cache separately.
    prefix = f"structural-depth-{open_chars}-{close_chars}"
    kernels = compile_kernel_group(prefix, source, _DEPTH_DELTAS_NAMES)
    _depth_kernel_cache[key] = kernels
    return kernels


# ---------------------------------------------------------------------------
# NVRTC warmup registration (ADR-0034 Level 2)
# ---------------------------------------------------------------------------
# Register the quote-toggle kernel and the default JSON bracket kernel
# for background precompilation at import time.

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

# Default JSON depth source for warmup (most common case)
_DEFAULT_DEPTH_SOURCE = _build_depth_source("{[", "}]")

request_nvrtc_warmup([
    ("structural-quote-toggle", _QUOTE_TOGGLE_SOURCE, _QUOTE_TOGGLE_NAMES),
    ("structural-depth-{[-}]", _DEFAULT_DEPTH_SOURCE, _DEPTH_DELTAS_NAMES),
])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def quote_parity(d_bytes: cp.ndarray) -> cp.ndarray:
    """Compute per-byte quote-parity mask via toggle + cumulative sum.

    Marks each byte position as inside (1) or outside (0) a quoted
    string literal.  The algorithm is:

    1. A per-byte kernel emits 1 at unescaped quote characters (``"``),
       0 elsewhere.  Escaped quotes (preceded by an odd number of
       backslashes) emit 0.
    2. A ``uint8`` cumulative sum over the toggle array yields a
       monotonically increasing counter.  A bitwise AND with 1 extracts
       the low bit, producing the parity: 0 = outside string, 1 = inside
       string.

    Using ``uint8`` cumsum instead of ``int32`` saves 4x memory
    (2.16 GB vs 8.64 GB for a 2 GB input file).  Parity remains
    correct after uint8 overflow because 256 is even.

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 array of raw file bytes, shape ``(n,)``.

    Returns
    -------
    cp.ndarray
        Device-resident uint8 array of shape ``(n,)``.  Each element is
        0 (outside quoted string) or 1 (inside quoted string).

    Notes
    -----
    The quote character is always ASCII ``"`` (0x22).  This primitive
    does not support single-quoted strings.  Backslash-escaped quotes
    (``\\"``) are handled correctly by counting consecutive preceding
    backslashes: a quote preceded by an odd number of backslashes is
    escaped and does not toggle parity.

    This is the first stage in any GPU text-parsing pipeline.  The
    resulting parity mask is consumed by ``bracket_depth``,
    ``number_boundaries``, and ``pattern_match`` to filter out bytes
    that appear inside string literals.

    Examples
    --------
    >>> # Input: {"key": "val"}
    >>> # Bytes:  { " k e y " :   " v a l " }
    >>> # Parity: 0 0 1 1 1 0 0 0 0 1 1 1 0 0
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    n = d_bytes.shape[0]
    n_i64 = np.int64(n)

    # Compile (cached via SHA1 of source)
    kernels = compile_kernel_group(
        "structural-quote-toggle", _QUOTE_TOGGLE_SOURCE, _QUOTE_TOGGLE_NAMES,
    )

    # Allocate toggle output on device
    d_toggle = cp.empty(n, dtype=cp.uint8)

    # Launch with occupancy-based config
    grid, block = runtime.launch_config(kernels["quote_toggle"], n)
    runtime.launch(
        kernels["quote_toggle"],
        grid=grid,
        block=block,
        params=(
            (ptr(d_bytes), ptr(d_toggle), n_i64),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I64),
        ),
    )

    # uint8 cumsum + bitwise AND for parity (no sync needed -- same stream)
    d_parity = cp.cumsum(d_toggle, dtype=cp.uint8) & np.uint8(1)
    del d_toggle
    return d_parity


def bracket_depth(
    d_bytes: cp.ndarray,
    d_quote_parity: cp.ndarray,
    *,
    open_chars: str = "{[",
    close_chars: str = "}]",
) -> cp.ndarray:
    """Compute per-byte nesting depth via delta kernel + prefix sum.

    Produces an int32 array where each position holds the cumulative
    bracket depth at that byte offset.  The algorithm is:

    1. A per-byte kernel emits ``+1`` for open-bracket characters,
       ``-1`` for close-bracket characters, and ``0`` for all other
       bytes.  Brackets inside quoted strings (where ``d_quote_parity``
       is 1) are treated as ``0``.
    2. An ``int32`` cumulative sum over the delta array yields the
       running depth at each byte position.

    The open/close character sets are parameterizable so that the same
    primitive works across formats:

    - JSON: ``open_chars="{[", close_chars="}]"``
    - WKT:  ``open_chars="(", close_chars=")"``
    - XML:  ``open_chars="<", close_chars=">"``

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 array of raw file bytes, shape ``(n,)``.
    d_quote_parity : cp.ndarray
        Device-resident uint8 parity mask from ``quote_parity()``,
        shape ``(n,)``.  Positions with parity 1 (inside string) are
        excluded from depth computation.
    open_chars : str, default ``"{["``
        Characters that increment depth.  Each character is treated
        independently.  Maximum 8 characters.
    close_chars : str, default ``"}]"``
        Characters that decrement depth.  Must have the same length as
        ``open_chars``.  Each character is treated independently.

    Returns
    -------
    cp.ndarray
        Device-resident int32 array of shape ``(n,)``.  Element ``i``
        holds the cumulative nesting depth at byte offset ``i``.  The
        depth is inclusive: at an opening bracket, the depth already
        includes the ``+1`` delta from that bracket.  At a closing
        bracket, the depth includes the ``-1`` delta.

    Raises
    ------
    ValueError
        If ``len(open_chars) != len(close_chars)`` or either exceeds
        8 characters.

    Notes
    -----
    The intermediate delta array uses ``int8`` dtype (1 byte per
    position) to minimize memory before the cumsum materializes
    the ``int32`` depth array.

    For JSON documents, the depth structure follows:

    - Depth 0: outside the root object
    - Depth 1: inside ``FeatureCollection { }``
    - Depth 2: inside ``"features": [ ]``
    - Depth 3: inside each ``Feature { }``
    - Depth 4+: nested geometry objects and coordinate arrays

    Examples
    --------
    >>> # Input: {"a": [1, 2]}
    >>> # Depth:  1 1 1 1 1 2 2 2 2 1 0
    """
    if len(open_chars) != len(close_chars):
        raise ValueError(
            f"open_chars and close_chars must have the same length, "
            f"got {len(open_chars)} and {len(close_chars)}"
        )
    if len(open_chars) > 8:
        raise ValueError(
            f"Maximum 8 bracket characters supported, got {len(open_chars)}"
        )

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    n = d_bytes.shape[0]
    n_i64 = np.int64(n)

    # Compile (cached by bracket chars via _depth_kernel_cache + SHA1)
    kernels = _get_depth_kernels(open_chars, close_chars)

    # Allocate int8 delta output on device
    d_deltas = cp.empty(n, dtype=cp.int8)

    # Launch with occupancy-based config
    grid, block = runtime.launch_config(
        kernels["compute_depth_deltas"], n,
    )
    runtime.launch(
        kernels["compute_depth_deltas"],
        grid=grid,
        block=block,
        params=(
            (ptr(d_bytes), ptr(d_quote_parity), ptr(d_deltas), n_i64),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I64),
        ),
    )

    # int32 cumsum over int8 deltas (no sync needed -- same stream)
    d_depth = cp.cumsum(d_deltas, dtype=cp.int32)
    del d_deltas
    return d_depth
