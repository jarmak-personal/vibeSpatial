"""GPU WKT reader -- structural analysis and geometry type detection.

First stage of a GPU-accelerated WKT parser.  Given a device-resident
byte array containing one or more WKT geometries (one per line), this
module performs:

1. **Line splitting** -- detect newline boundaries to delimit individual
   geometry strings.
2. **Parenthesis depth** -- reuse ``gpu_parse.bracket_depth`` with
   ``open_chars="("``, ``close_chars=")"``.
3. **Geometry type classification** -- a custom NVRTC kernel scans the
   start of each geometry string and emits a family tag (POINT=0,
   LINESTRING=1, POLYGON=2, MULTIPOINT=3, MULTILINESTRING=4,
   MULTIPOLYGON=5) plus an EMPTY flag.  Handles case-insensitive
   matching and EWKT ``SRID=NNNN;`` prefixes.

All operations run on the GPU with zero host materialization until the
caller explicitly requests results.

Tier classification (ADR-0033):
    - Line splitting: Tier 2 (CuPy element-wise + flatnonzero)
    - Parenthesis depth: delegates to gpu_parse.bracket_depth (Tier 1)
    - Type classification: Tier 1 (custom NVRTC -- text-specific prefix matching)

Precision (ADR-0002):
    All kernels are integer-only byte classification.  No floating-point
    coordinate computation occurs, so no PrecisionPlan is needed (same
    rationale as gpu_parse/structural.py).
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.io.gpu_parse.structural import bracket_depth

if TYPE_CHECKING:
    import cupy as cp

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

_WKT_CLASSIFY_SOURCE = r"""
// WKT geometry type classification kernel.
//
// Each thread examines one geometry start position and classifies it
// by checking byte prefixes.  Case-insensitive: each byte comparison
// tests both upper and lower case via bitwise OR with 0x20 (which
// maps ASCII A-Z to a-z while leaving a-z unchanged).
//
// EWKT support: if the geometry starts with 'S' or 's' (for "SRID="),
// the kernel scans forward past the semicolon before classifying.
//
// Family tags match GeometryFamily enum order:
//   POINT=0, LINESTRING=1, POLYGON=2,
//   MULTIPOINT=3, MULTILINESTRING=4, MULTIPOLYGON=5,
//   unknown/unsupported=-2
//
// Also detects the EMPTY keyword after the type name and any
// optional dimension suffix (Z, M, ZM).

extern "C" __global__ void __launch_bounds__(256, 4)
wkt_classify_geometry_type(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ geom_starts,
    signed char* __restrict__ family_tags,
    unsigned char* __restrict__ empty_flags,
    const int n_geoms,
    const long long n_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_geoms) return;

    long long pos = geom_starts[idx];

    // Skip leading whitespace
    while (pos < n_bytes) {
        unsigned char c = input[pos];
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
        pos++;
    }

    if (pos >= n_bytes) {
        family_tags[idx] = -2;
        empty_flags[idx] = 0;
        return;
    }

    // EWKT: handle SRID=NNNN; prefix
    // Check for 'S'/'s' followed by 'R'/'r' (start of "SRID=")
    unsigned char first = input[pos] | 0x20;  // to lowercase
    if (first == 's' && pos + 1 < n_bytes && (input[pos + 1] | 0x20) == 'r') {
        // Scan forward past the semicolon
        while (pos < n_bytes && input[pos] != ';') pos++;
        if (pos < n_bytes) pos++;  // skip the semicolon
        // Skip whitespace after semicolon
        while (pos < n_bytes) {
            unsigned char c = input[pos];
            if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
            pos++;
        }
        if (pos >= n_bytes) {
            family_tags[idx] = -2;
            empty_flags[idx] = 0;
            return;
        }
    }

    // Now pos points to the geometry type keyword.
    // Helper macro: case-insensitive byte comparison
    #define LC(p) ((p) | 0x20)

    // Read first character (lowercased) to branch into type groups
    unsigned char c0 = LC(input[pos]);
    signed char tag = -2;
    long long type_end = pos;  // will track where the type name ends

    if (c0 == 'p') {
        // Could be POINT or POLYGON
        if (pos + 4 < n_bytes
            && LC(input[pos + 1]) == 'o'
            && LC(input[pos + 2]) == 'i'
            && LC(input[pos + 3]) == 'n'
            && LC(input[pos + 4]) == 't') {
            tag = 0;  // POINT
            type_end = pos + 5;
        } else if (pos + 6 < n_bytes
            && LC(input[pos + 1]) == 'o'
            && LC(input[pos + 2]) == 'l'
            && LC(input[pos + 3]) == 'y'
            && LC(input[pos + 4]) == 'g'
            && LC(input[pos + 5]) == 'o'
            && LC(input[pos + 6]) == 'n') {
            tag = 2;  // POLYGON
            type_end = pos + 7;
        }
    } else if (c0 == 'l') {
        // LINESTRING
        if (pos + 9 < n_bytes
            && LC(input[pos + 1]) == 'i'
            && LC(input[pos + 2]) == 'n'
            && LC(input[pos + 3]) == 'e'
            && LC(input[pos + 4]) == 's'
            && LC(input[pos + 5]) == 't'
            && LC(input[pos + 6]) == 'r'
            && LC(input[pos + 7]) == 'i'
            && LC(input[pos + 8]) == 'n'
            && LC(input[pos + 9]) == 'g') {
            tag = 1;  // LINESTRING
            type_end = pos + 10;
        }
    } else if (c0 == 'm') {
        // MULTI* types: check "MULTI" prefix first
        if (pos + 4 < n_bytes
            && LC(input[pos + 1]) == 'u'
            && LC(input[pos + 2]) == 'l'
            && LC(input[pos + 3]) == 't'
            && LC(input[pos + 4]) == 'i') {
            long long mpos = pos + 5;
            if (mpos < n_bytes) {
                unsigned char mc = LC(input[mpos]);
                if (mc == 'p') {
                    // MULTIPOINT or MULTIPOLYGON
                    if (mpos + 4 < n_bytes
                        && LC(input[mpos + 1]) == 'o'
                        && LC(input[mpos + 2]) == 'i'
                        && LC(input[mpos + 3]) == 'n'
                        && LC(input[mpos + 4]) == 't') {
                        tag = 3;  // MULTIPOINT
                        type_end = mpos + 5;
                    } else if (mpos + 6 < n_bytes
                        && LC(input[mpos + 1]) == 'o'
                        && LC(input[mpos + 2]) == 'l'
                        && LC(input[mpos + 3]) == 'y'
                        && LC(input[mpos + 4]) == 'g'
                        && LC(input[mpos + 5]) == 'o'
                        && LC(input[mpos + 6]) == 'n') {
                        tag = 5;  // MULTIPOLYGON
                        type_end = mpos + 7;
                    }
                } else if (mc == 'l') {
                    // MULTILINESTRING
                    if (mpos + 9 < n_bytes
                        && LC(input[mpos + 1]) == 'i'
                        && LC(input[mpos + 2]) == 'n'
                        && LC(input[mpos + 3]) == 'e'
                        && LC(input[mpos + 4]) == 's'
                        && LC(input[mpos + 5]) == 't'
                        && LC(input[mpos + 6]) == 'r'
                        && LC(input[mpos + 7]) == 'i'
                        && LC(input[mpos + 8]) == 'n'
                        && LC(input[mpos + 9]) == 'g') {
                        tag = 4;  // MULTILINESTRING
                        type_end = mpos + 10;
                    }
                }
            }
        }
    } else if (c0 == 'g') {
        // GEOMETRYCOLLECTION
        if (pos + 17 < n_bytes
            && LC(input[pos + 1]) == 'e'
            && LC(input[pos + 2]) == 'o'
            && LC(input[pos + 3]) == 'm'
            && LC(input[pos + 4]) == 'e'
            && LC(input[pos + 5]) == 't'
            && LC(input[pos + 6]) == 'r'
            && LC(input[pos + 7]) == 'y'
            && LC(input[pos + 8]) == 'c'
            && LC(input[pos + 9]) == 'o'
            && LC(input[pos + 10]) == 'l'
            && LC(input[pos + 11]) == 'l'
            && LC(input[pos + 12]) == 'e'
            && LC(input[pos + 13]) == 'c'
            && LC(input[pos + 14]) == 't'
            && LC(input[pos + 15]) == 'i'
            && LC(input[pos + 16]) == 'o'
            && LC(input[pos + 17]) == 'n') {
            tag = -2;  // unsupported for now
            type_end = pos + 18;
        }
    }

    family_tags[idx] = tag;

    // Detect EMPTY keyword after type name.
    // WKT allows optional dimension suffix (Z, M, ZM) between the
    // type name and EMPTY/opening paren, with or without space:
    //   POINT EMPTY, POINT Z EMPTY, POINTZ EMPTY, POINT ZM EMPTY
    long long ep = type_end;
    // Skip optional dimension suffix: Z, M, ZM (with or without space)
    if (ep < n_bytes) {
        unsigned char dc = LC(input[ep]);
        // Space before dimension suffix
        if (dc == ' ' || dc == '\t') {
            long long sp = ep;
            while (sp < n_bytes && (input[sp] == ' ' || input[sp] == '\t')) sp++;
            if (sp < n_bytes) {
                unsigned char sc = LC(input[sp]);
                if (sc == 'z' || sc == 'm') {
                    ep = sp;
                    dc = sc;
                }
            }
        }
        if (dc == 'z') {
            ep++;
            if (ep < n_bytes && LC(input[ep]) == 'm') ep++;  // ZM
        } else if (dc == 'm') {
            ep++;
        }
    }

    // Skip whitespace before EMPTY or (
    while (ep < n_bytes && (input[ep] == ' ' || input[ep] == '\t')) ep++;

    // Check for EMPTY keyword (case-insensitive)
    unsigned char is_empty = 0;
    if (ep + 4 < n_bytes
        && LC(input[ep]) == 'e'
        && LC(input[ep + 1]) == 'm'
        && LC(input[ep + 2]) == 'p'
        && LC(input[ep + 3]) == 't'
        && LC(input[ep + 4]) == 'y') {
        is_empty = 1;
    }
    empty_flags[idx] = is_empty;

    #undef LC
}
"""

_WKT_CLASSIFY_NAMES: tuple[str, ...] = ("wkt_classify_geometry_type",)


# ---------------------------------------------------------------------------
# NVRTC warmup registration (ADR-0034 Level 2)
# ---------------------------------------------------------------------------
# Register the WKT classify kernel for background precompilation.
# bracket_depth warmup for WKT parentheses is handled lazily by
# structural.py's _get_depth_kernels on first use (cached by char pair).

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("wkt-classify-type", _WKT_CLASSIFY_SOURCE, _WKT_CLASSIFY_NAMES),
])


# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------

def _classify_type_kernels() -> dict:
    """Compile (or retrieve cached) WKT type classification kernel."""
    return compile_kernel_group(
        "wkt-classify-type", _WKT_CLASSIFY_SOURCE, _WKT_CLASSIFY_NAMES,
    )


def _launch_kernel(runtime, kernel, n, params):
    """Launch a kernel with occupancy-based grid/block sizing."""
    grid, block = runtime.launch_config(kernel, int(n))
    runtime.launch(kernel, grid=grid, block=block, params=params)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class WktStructuralResult:
    """Result of WKT structural analysis.

    All arrays are device-resident CuPy arrays except ``n_geometries``
    which is a Python int.

    Attributes
    ----------
    d_depth : cp.ndarray
        Per-byte parenthesis depth, int32, shape ``(n_bytes,)``.
    d_geom_starts : cp.ndarray
        Start byte offset of each geometry, int64, shape ``(n_geometries,)``.
    d_family_tags : cp.ndarray
        Geometry family tag per geometry, int8, shape ``(n_geometries,)``.
        Values: 0=POINT, 1=LINESTRING, 2=POLYGON, 3=MULTIPOINT,
        4=MULTILINESTRING, 5=MULTIPOLYGON, -2=unknown/unsupported.
    d_empty_flags : cp.ndarray
        Per-geometry EMPTY flag, uint8, shape ``(n_geometries,)``.
        1 if the geometry uses the EMPTY keyword, 0 otherwise.
    n_geometries : int
        Number of geometries detected.
    """

    d_depth: cp.ndarray
    d_geom_starts: cp.ndarray
    d_family_tags: cp.ndarray
    d_empty_flags: cp.ndarray
    n_geometries: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def wkt_structural_analysis(d_bytes: cp.ndarray) -> WktStructuralResult:
    """Perform structural analysis and geometry type detection on WKT input.

    Given a device-resident byte array containing one or more WKT
    geometry strings separated by newlines, this function:

    1. Detects line boundaries (newline positions) to delimit geometries.
    2. Computes per-byte parenthesis depth using ``bracket_depth``.
    3. Classifies each geometry by type keyword and detects EMPTY.

    The input may contain:

    - Standard WKT: ``POINT(1 2)``
    - EWKT with SRID prefix: ``SRID=4326;POINT(1 2)``
    - Mixed case: ``Point(1 2)``, ``LINESTRING(...)``
    - 3D/M suffixes: ``POINT Z(1 2 3)``, ``POINTZ(1 2 3)``
    - Empty geometries: ``POINT EMPTY``

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 array of WKT text bytes, shape ``(n,)``.
        Multiple geometries are separated by newline characters
        (``\\n``, 0x0A).  Trailing newlines are handled gracefully.

    Returns
    -------
    WktStructuralResult
        Dataclass containing all structural analysis outputs on device.

    Notes
    -----
    WKT has no string quoting, so ``bracket_depth`` receives an
    all-zeros quote-parity array.  This causes the depth kernel to
    treat every parenthesis as structural.

    The parenthesis depth array uses the same convention as the
    GeoJSON bracket depth:

    - Depth 0: outside all geometry parentheses
    - Depth 1: inside the outermost ``(...)``
    - Depth 2+: nested rings, coordinate groups, etc.

    Examples
    --------
    >>> import cupy as cp
    >>> wkt = b"POINT(1 2)\\nLINESTRING(0 0, 1 1)\\nPOLYGON EMPTY"
    >>> d_bytes = cp.frombuffer(wkt, dtype=cp.uint8)
    >>> result = wkt_structural_analysis(d_bytes)
    >>> result.n_geometries
    3
    >>> result.d_family_tags.get()  # array([0, 1, 2], dtype=int8)
    >>> result.d_empty_flags.get()  # array([0, 0, 1], dtype=uint8)
    """
    import cupy as cp

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    n = d_bytes.shape[0]
    n_i64 = np.int64(n)

    # ------------------------------------------------------------------
    # Stage 1: Parenthesis depth
    # ------------------------------------------------------------------
    # WKT has no string quoting, so we pass an all-zeros quote parity.
    # This makes bracket_depth treat every '(' and ')' as structural.
    d_quote_parity = cp.zeros(n, dtype=cp.uint8)
    d_depth = bracket_depth(
        d_bytes, d_quote_parity, open_chars="(", close_chars=")",
    )
    del d_quote_parity  # free immediately

    # ------------------------------------------------------------------
    # Stage 2: Detect line boundaries (geometry starts)
    # ------------------------------------------------------------------
    # Each geometry is one line.  Find newline positions, then derive
    # geometry start offsets.  Geometry 0 always starts at byte 0;
    # subsequent geometries start at newline_pos + 1.
    #
    # Tier 2 (CuPy): element-wise comparison + flatnonzero.
    d_is_newline = (d_bytes == ord('\n'))
    d_newline_positions = cp.flatnonzero(d_is_newline).astype(cp.int64)
    del d_is_newline

    n_newlines = d_newline_positions.shape[0]

    # Build geometry start positions: [0, nl[0]+1, nl[1]+1, ...]
    # But only include starts that point to non-empty lines.
    if n_newlines == 0:
        # Single geometry (no newlines) or empty input
        if n == 0:
            return WktStructuralResult(
                d_depth=d_depth,
                d_geom_starts=cp.empty(0, dtype=cp.int64),
                d_family_tags=cp.empty(0, dtype=cp.int8),
                d_empty_flags=cp.empty(0, dtype=cp.uint8),
                n_geometries=0,
            )
        d_geom_starts = cp.zeros(1, dtype=cp.int64)
    else:
        # Starts = [0] + [nl_pos + 1 for each newline that is not the last byte]
        d_after_newlines = d_newline_positions + 1
        # Include byte 0 as the first geometry start
        d_zero = cp.zeros(1, dtype=cp.int64)
        # Filter out starts that are >= n (trailing newline at end of file)
        d_candidate_starts = cp.concatenate([d_zero, d_after_newlines])
        d_valid_mask = d_candidate_starts < n
        d_geom_starts = d_candidate_starts[d_valid_mask]
        del d_after_newlines, d_zero, d_candidate_starts, d_valid_mask

    del d_newline_positions

    # Filter out empty lines (blank or whitespace-only).  A start whose
    # byte is a newline or carriage return indicates a blank line.  We
    # filter these on device to avoid the classify kernel skipping
    # whitespace across line boundaries into the next geometry.
    if d_geom_starts.shape[0] > 0:
        d_start_bytes = d_bytes[d_geom_starts]
        d_non_empty = (d_start_bytes != ord('\n')) & (d_start_bytes != ord('\r'))
        d_geom_starts = d_geom_starts[d_non_empty]
        del d_start_bytes, d_non_empty

    n_geoms = d_geom_starts.shape[0]

    if n_geoms == 0:
        return WktStructuralResult(
            d_depth=d_depth,
            d_geom_starts=d_geom_starts,
            d_family_tags=cp.empty(0, dtype=cp.int8),
            d_empty_flags=cp.empty(0, dtype=cp.uint8),
            n_geometries=0,
        )

    # ------------------------------------------------------------------
    # Stage 3: Classify geometry types
    # ------------------------------------------------------------------
    kernels = _classify_type_kernels()
    d_family_tags = cp.empty(n_geoms, dtype=cp.int8)
    d_empty_flags = cp.empty(n_geoms, dtype=cp.uint8)

    _launch_kernel(
        runtime,
        kernels["wkt_classify_geometry_type"],
        n_geoms,
        (
            (
                ptr(d_bytes),
                ptr(d_geom_starts),
                ptr(d_family_tags),
                ptr(d_empty_flags),
                np.int32(n_geoms),
                n_i64,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I64,
            ),
        ),
    )

    # No sync needed before returning -- all outputs are device arrays
    # and the caller will sync when materializing to host.

    return WktStructuralResult(
        d_depth=d_depth,
        d_geom_starts=d_geom_starts,
        d_family_tags=d_family_tags,
        d_empty_flags=d_empty_flags,
        n_geometries=int(n_geoms),
    )
