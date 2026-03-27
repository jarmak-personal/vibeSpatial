"""GPU numeric parsing primitives for text formats.

Provides boundary detection and ASCII-to-number conversion for
numeric values embedded in structured text.  The pipeline is:

1. ``number_boundaries`` — per-byte kernel classifies number start/end
2. ``extract_number_positions`` — compact boundary masks to position arrays
3. ``parse_ascii_floats`` / ``parse_ascii_ints`` — per-number parallel parse

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
)

if TYPE_CHECKING:
    import cupy as cp

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

# ctypes for int64 kernel params (not in _runtime.py which only has i32)
KERNEL_PARAM_I64 = ctypes.c_longlong

# ---------------------------------------------------------------------------
# Kernel sources (all Tier 1 NVRTC)
# ---------------------------------------------------------------------------

_NUM_BOUNDS_SOURCE = r"""
extern "C" __global__ void find_number_boundaries(
    const unsigned char* __restrict__ input,
    const unsigned char* __restrict__ quote_parity,
    unsigned char* __restrict__ is_start,
    unsigned char* __restrict__ is_end,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Skip bytes inside string values (odd parity)
    if (quote_parity[idx] != 0) {
        is_start[idx] = 0;
        is_end[idx] = 0;
        return;
    }

    unsigned char c = input[idx];
    unsigned char prev = (idx > 0) ? input[idx - 1] : '[';
    unsigned char next = (idx < n - 1) ? input[idx + 1] : ']';

    // Number starts: first char of a number, preceded by separator
    // Space included because json.dumps uses ", " as separator
    unsigned char is_first_digit = (c >= '0' && c <= '9') || c == '-' || c == '+';
    unsigned char is_sep_before = (prev == ',' || prev == '[' || prev == ' '
                                   || prev == '\n' || prev == '\r' || prev == '\t');
    is_start[idx] = is_first_digit && is_sep_before;

    // Number ends: last numeric char followed by separator
    // Space/newline included because GDAL/OGR writes "0.0 ]" with
    // whitespace between the last coordinate value and closing bracket.
    unsigned char is_numeric = (c >= '0' && c <= '9') || c == '.' ||
                               c == 'e' || c == 'E' || c == '-' || c == '+';
    unsigned char is_sep_after = (next == ',' || next == ']' || next == ' '
                                  || next == '\n' || next == '\r' || next == '\t');
    is_end[idx] = is_numeric && is_sep_after;
}
"""

_PARSE_FLOAT_SOURCE = r"""
extern "C" __global__ void parse_ascii_floats(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ coord_starts,
    const long long* __restrict__ coord_ends,
    double* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    long long start = coord_starts[idx];
    long long end = coord_ends[idx];

    double result = 0.0;
    double frac_mult = 0.0;
    int negative = 0;
    int in_exponent = 0;
    int exp_val = 0;
    int exp_negative = 0;

    for (long long i = start; i < end; ++i) {
        unsigned char c = input[i];
        if (c == '-') {
            if (in_exponent) exp_negative = 1;
            else negative = 1;
        } else if (c == '+') {
            // skip
        } else if (c == '.') {
            frac_mult = 0.1;
        } else if (c == 'e' || c == 'E') {
            in_exponent = 1;
        } else if (c >= '0' && c <= '9') {
            int d = c - '0';
            if (in_exponent) {
                exp_val = exp_val * 10 + d;
            } else if (frac_mult > 0.0) {
                result += d * frac_mult;
                frac_mult *= 0.1;
            } else {
                result = result * 10.0 + d;
            }
        }
    }

    if (negative) result = -result;

    if (in_exponent) {
        double exp_mult = 1.0;
        for (int e = 0; e < exp_val; ++e) exp_mult *= 10.0;
        if (exp_negative) result /= exp_mult;
        else result *= exp_mult;
    }

    output[idx] = result;
}
"""

_PARSE_INT_SOURCE = r"""
extern "C" __global__ void parse_ascii_ints(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ token_starts,
    const long long* __restrict__ token_ends,
    long long* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    long long start = token_starts[idx];
    long long end = token_ends[idx];

    long long result = 0;
    int negative = 0;

    for (long long i = start; i < end; ++i) {
        unsigned char c = input[i];
        if (c == '-') {
            negative = 1;
        } else if (c == '+') {
            // skip
        } else if (c >= '0' && c <= '9') {
            result = result * 10 + (c - '0');
        } else {
            // Non-digit, non-sign: stop accumulating
            break;
        }
    }

    if (negative) result = -result;

    output[idx] = result;
}
"""

# Kernel name tuples
_NUM_BOUNDS_NAMES = ("find_number_boundaries",)
_PARSE_FLOAT_NAMES = ("parse_ascii_floats",)
_PARSE_INT_NAMES = ("parse_ascii_ints",)

# ---------------------------------------------------------------------------
# NVRTC warmup (ADR-0034 Level 2)
# ---------------------------------------------------------------------------
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("gpu-parse-num-bounds", _NUM_BOUNDS_SOURCE, _NUM_BOUNDS_NAMES),
    ("gpu-parse-float", _PARSE_FLOAT_SOURCE, _PARSE_FLOAT_NAMES),
    ("gpu-parse-int", _PARSE_INT_SOURCE, _PARSE_INT_NAMES),
])


# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------

def _num_bounds_kernels():
    return compile_kernel_group(
        "gpu-parse-num-bounds", _NUM_BOUNDS_SOURCE, _NUM_BOUNDS_NAMES,
    )


def _parse_float_kernels():
    return compile_kernel_group(
        "gpu-parse-float", _PARSE_FLOAT_SOURCE, _PARSE_FLOAT_NAMES,
    )


def _parse_int_kernels():
    return compile_kernel_group(
        "gpu-parse-int", _PARSE_INT_SOURCE, _PARSE_INT_NAMES,
    )


# ---------------------------------------------------------------------------
# Kernel launch helper
# ---------------------------------------------------------------------------

def _launch_kernel(runtime, kernel, n, params):
    grid, block = runtime.launch_config(kernel, int(n))
    runtime.launch(kernel, grid=grid, block=block, params=params)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def number_boundaries(
    d_bytes: cp.ndarray,
    d_quote_parity: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Identify start and end positions of numeric tokens.

    A per-byte GPU kernel classifies each byte position as either the
    start of a numeric token, the end of a numeric token, or neither.
    Bytes inside quoted strings (``d_quote_parity == 1``) are always
    classified as neither.

    Start detection
        A byte is a number start if it is a numeric-initial character
        (``0-9``, ``-``, ``+``) AND the preceding byte is a separator
        (``,``, ``[``, space, tab, newline, carriage return).

    End detection
        A byte is a number end if it is a numeric character (``0-9``,
        ``.``, ``e``, ``E``, ``-``, ``+``) AND the following byte is a
        separator (``,``, ``]``, space, tab, newline, carriage return).

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 array of raw file bytes, shape ``(n,)``.
    d_quote_parity : cp.ndarray
        Device-resident uint8 parity mask from
        ``structural.quote_parity()``, shape ``(n,)``.

    Returns
    -------
    d_is_start : cp.ndarray
        Device-resident uint8 array, shape ``(n,)``.  Element ``i`` is
        1 if byte ``i`` is the first byte of a numeric token, else 0.
    d_is_end : cp.ndarray
        Device-resident uint8 array, shape ``(n,)``.  Element ``i`` is
        1 if byte ``i`` is the last byte of a numeric token, else 0.

    Notes
    -----
    The returned arrays are *byte-level masks*, not position arrays.
    Use ``extract_number_positions`` to convert them to compact
    int64 position arrays suitable for ``parse_ascii_floats``.

    The boundary heuristic is designed for JSON/CSV numeric formats.
    It handles:

    - Integers: ``123``, ``-42``
    - Decimals: ``3.14``, ``-0.001``
    - Scientific notation: ``1.5e10``, ``-2.3E-4``
    - Leading sign: ``+1.0``, ``-1.0``

    Examples
    --------
    >>> # Input: [1.5, -2.3]
    >>> #         ^ ^   ^ ^   (start, end pairs)
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    n = len(d_bytes)
    n_i64 = np.int64(n)

    kernels = _num_bounds_kernels()

    d_is_start = cp.zeros(n, dtype=cp.uint8)
    d_is_end = cp.zeros(n, dtype=cp.uint8)

    _launch_kernel(runtime, kernels["find_number_boundaries"], n, (
        (ptr(d_bytes), ptr(d_quote_parity), ptr(d_is_start), ptr(d_is_end), n_i64),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I64),
    ))

    return d_is_start, d_is_end


def parse_ascii_floats(
    d_bytes: cp.ndarray,
    d_starts: cp.ndarray,
    d_ends: cp.ndarray,
) -> cp.ndarray:
    """Parse ASCII numeric tokens to float64 values on GPU.

    Each CUDA thread processes one token defined by the half-open byte
    range ``[d_starts[i], d_ends[i])``.  The kernel implements a
    character-by-character state machine supporting:

    - Optional leading sign (``+`` or ``-``)
    - Integer part (digits before decimal point)
    - Optional fractional part (``.`` followed by digits)
    - Optional scientific notation exponent (``e``/``E``, optional
      sign, digits)

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 array of raw file bytes, shape ``(n_bytes,)``.
    d_starts : cp.ndarray
        Device-resident int64 array, shape ``(n_numbers,)``.  Element
        ``i`` is the byte offset of the first character of the ``i``-th
        numeric token (inclusive).
    d_ends : cp.ndarray
        Device-resident int64 array, shape ``(n_numbers,)``.  Element
        ``i`` is the byte offset one past the last character of the
        ``i``-th numeric token (exclusive).

    Returns
    -------
    cp.ndarray
        Device-resident float64 array, shape ``(n_numbers,)``.  Each
        element is the parsed floating-point value.  Invalid tokens
        produce ``0.0`` (not NaN) --- callers should validate input
        boundaries.

    Notes
    -----
    The start/end convention is half-open: ``[start, end)``.  This
    matches the output of ``extract_number_positions``, where ends
    are already incremented by 1 from the ``d_is_end`` mask positions.

    The kernel does not handle ``NaN``, ``Infinity``, or hexadecimal
    float literals.  These are not valid in JSON or standard CSV.

    Examples
    --------
    >>> # d_bytes contains b"[1.5, -2.3e4]"
    >>> # d_starts = [1, 6],  d_ends = [4, 12]
    >>> # result = [1.5, -23000.0]
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    n_nums = len(d_starts)

    d_output = cp.empty(n_nums, dtype=cp.float64)

    if n_nums > 0:
        kernels = _parse_float_kernels()
        _launch_kernel(runtime, kernels["parse_ascii_floats"], n_nums, (
            (ptr(d_bytes), ptr(d_starts), ptr(d_ends), ptr(d_output), np.int32(n_nums)),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ))

    return d_output


def parse_ascii_ints(
    d_bytes: cp.ndarray,
    d_starts: cp.ndarray,
    d_ends: cp.ndarray,
) -> cp.ndarray:
    """Parse ASCII integer tokens to int64 values on GPU.

    Each CUDA thread processes one token defined by the half-open byte
    range ``[d_starts[i], d_ends[i])``.  The kernel implements a simple
    character-by-character accumulator supporting:

    - Optional leading sign (``+`` or ``-``)
    - Decimal digits (``0-9``)

    Fractional parts and exponent notation are not supported.  If a
    non-digit character (other than a leading sign) is encountered,
    accumulation stops at that position.

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 array of raw file bytes, shape ``(n_bytes,)``.
    d_starts : cp.ndarray
        Device-resident int64 array, shape ``(n_numbers,)``.  Element
        ``i`` is the byte offset of the first character of the ``i``-th
        integer token (inclusive).
    d_ends : cp.ndarray
        Device-resident int64 array, shape ``(n_numbers,)``.  Element
        ``i`` is the byte offset one past the last character of the
        ``i``-th integer token (exclusive).

    Returns
    -------
    cp.ndarray
        Device-resident int64 array, shape ``(n_numbers,)``.  Each
        element is the parsed integer value.  Tokens that contain no
        valid digits produce ``0``.  Overflow wraps silently (int64
        range: ``-2^63`` to ``2^63 - 1``).

    Notes
    -----
    This function does NOT exist in the current geojson_gpu.py pipeline.
    It is a new primitive for formats that contain integer fields
    (e.g., feature IDs in GeoJSON, integer attributes in CSV, SRID
    values in WKT).

    The start/end convention is half-open: ``[start, end)``, consistent
    with ``parse_ascii_floats``.

    Examples
    --------
    >>> # d_bytes contains b"SRID=4326;POINT(1 2)"
    >>> # d_starts = [5],  d_ends = [9]
    >>> # result = [4326]
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    n_nums = len(d_starts)

    d_output = cp.empty(n_nums, dtype=cp.int64)

    if n_nums > 0:
        kernels = _parse_int_kernels()
        _launch_kernel(runtime, kernels["parse_ascii_ints"], n_nums, (
            (ptr(d_bytes), ptr(d_starts), ptr(d_ends), ptr(d_output), np.int32(n_nums)),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ))

    return d_output


def extract_number_positions(
    d_is_start: cp.ndarray,
    d_is_end: cp.ndarray,
    d_mask: cp.ndarray | None = None,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Convert boundary masks to compact position arrays.

    Takes the per-byte start/end masks from ``number_boundaries`` and
    produces compact int64 position arrays suitable for
    ``parse_ascii_floats`` or ``parse_ascii_ints``.

    Optionally filters by a region mask so that only numbers within
    specific spans (e.g., coordinate spans in GeoJSON, value columns
    in CSV) are included.

    Parameters
    ----------
    d_is_start : cp.ndarray
        Device-resident uint8 array, shape ``(n_bytes,)``.  Per-byte
        number-start indicators from ``number_boundaries``.
    d_is_end : cp.ndarray
        Device-resident uint8 array, shape ``(n_bytes,)``.  Per-byte
        number-end indicators from ``number_boundaries``.
    d_mask : cp.ndarray or None, default None
        Optional device-resident uint8 region mask, shape ``(n_bytes,)``.
        If provided, only number boundaries where ``d_mask[i] == 1``
        are included.  When ``None``, all detected boundaries are
        returned.

    Returns
    -------
    d_starts : cp.ndarray
        Device-resident int64 array, shape ``(n_numbers,)``.  Byte
        offsets of the first character of each detected number
        (inclusive).
    d_ends : cp.ndarray
        Device-resident int64 array, shape ``(n_numbers,)``.  Byte
        offsets one past the last character of each detected number
        (exclusive).  This is computed as ``flatnonzero(d_is_end) + 1``
        so that the range ``[start, end)`` spans the full token.

    Notes
    -----
    When ``d_mask`` is provided, the function computes element-wise
    multiplication of both boundary masks with the region mask before
    extracting positions.  This avoids materializing filtered
    intermediate arrays.

    The returned arrays are always contiguous int64 arrays suitable
    for direct kernel parameter passing.

    Examples
    --------
    >>> # d_is_start marks positions [3, 7, 15]
    >>> # d_is_end marks positions [5, 10, 18]
    >>> # d_mask is 1 only in [0..12]
    >>> # Result: d_starts=[3, 7], d_ends=[6, 11]
    """
    if d_mask is not None:
        d_is_start = d_is_start * d_mask
        d_is_end = d_is_end * d_mask

    d_starts = cp.flatnonzero(d_is_start).astype(cp.int64)
    d_ends = cp.flatnonzero(d_is_end).astype(cp.int64) + 1

    return d_starts, d_ends
