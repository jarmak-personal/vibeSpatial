"""GPU CSV reader -- structural analysis stage.

GPU-accelerated CSV structural analysis.  Given a device-resident byte
array containing a CSV file, this module performs:

1. **Quote parity** -- a CSV-specific quote toggle kernel that emits 1 at
   each ``"`` character without backslash-escape checking (CSV uses ``""``
   doubled-quote escaping, not backslash escaping).  Doubled quotes
   naturally cancel in the cumulative-sum parity computation.

2. **Row boundary detection** -- mark positions of ``\\n`` characters where
   quote parity is 0 (outside quoted fields).  Handles both ``\\n`` and
   ``\\r\\n`` line endings.

3. **Column boundary detection** -- mark positions of the delimiter
   character where quote parity is 0.  The delimiter is configurable
   (comma, tab, pipe) and passed as a kernel parameter.

4. **Column count verification** -- pure CuPy helper that counts
   delimiters per row and verifies a consistent column count.

5. **Header parsing** -- CPU-side helper (small data, one-time D->H copy)
   that splits the first row by delimiter to extract column names and
   identify spatial columns by name heuristics.

All structural kernels are integer-only byte classification (no
floating-point computation), so no PrecisionPlan is needed per ADR-0002
-- same rationale as ``gpu_parse/structural.py``.

Tier classification (ADR-0033):
    - Quote toggle: Tier 1 (custom NVRTC -- byte classification)
    - Row end detection: Tier 1 (custom NVRTC -- byte + parity check)
    - Delimiter detection: Tier 1 (custom NVRTC -- byte + parity check)
    - Parity cumsum: Tier 2 (CuPy cumsum)
    - Position extraction: Tier 2 (CuPy flatnonzero)
    - Column count verification: Tier 2 (CuPy element-wise)
    - Header parsing: CPU (small data, one-time)
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

_CSV_QUOTE_TOGGLE_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(256, 4)
csv_quote_toggle(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // CSV uses "" (doubled quotes) for escaping, NOT backslash escaping.
    // Simply emit 1 at every " character.  Two consecutive quotes produce
    // two toggles that cancel each other in the cumulative-sum parity.
    output[idx] = (input[idx] == '"') ? (unsigned char)1 : (unsigned char)0;
}
"""

_CSV_QUOTE_TOGGLE_NAMES: tuple[str, ...] = ("csv_quote_toggle",)

_CSV_FIND_ROW_ENDS_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(256, 4)
csv_find_row_ends(
    const unsigned char* __restrict__ input,
    const unsigned char* __restrict__ quote_parity,
    unsigned char* __restrict__ is_row_end,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // A row ends at \n that is outside quotes (parity == 0).
    // For \r\n sequences, only the \n is marked as the row end;
    // the \r is ignored (it will be stripped during field extraction).
    unsigned char b = input[idx];
    is_row_end[idx] = (b == '\n' && quote_parity[idx] == 0)
                      ? (unsigned char)1
                      : (unsigned char)0;
}
"""

_CSV_FIND_ROW_ENDS_NAMES: tuple[str, ...] = ("csv_find_row_ends",)

_CSV_FIND_DELIMITERS_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(256, 4)
csv_find_delimiters(
    const unsigned char* __restrict__ input,
    const unsigned char* __restrict__ quote_parity,
    unsigned char* __restrict__ is_delimiter,
    long long n,
    int delimiter
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Mark delimiter characters that are outside quoted fields.
    unsigned char b = input[idx];
    is_delimiter[idx] = (b == (unsigned char)delimiter && quote_parity[idx] == 0)
                        ? (unsigned char)1
                        : (unsigned char)0;
}
"""

_CSV_FIND_DELIMITERS_NAMES: tuple[str, ...] = ("csv_find_delimiters",)

# ---------------------------------------------------------------------------
# NVRTC warmup registration (ADR-0034 Level 2)
# ---------------------------------------------------------------------------

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("csv-quote-toggle", _CSV_QUOTE_TOGGLE_SOURCE, _CSV_QUOTE_TOGGLE_NAMES),
    ("csv-find-row-ends", _CSV_FIND_ROW_ENDS_SOURCE, _CSV_FIND_ROW_ENDS_NAMES),
    ("csv-find-delimiters", _CSV_FIND_DELIMITERS_SOURCE, _CSV_FIND_DELIMITERS_NAMES),
])

# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------


def _quote_toggle_kernels() -> dict[str, object]:
    return compile_kernel_group(
        "csv-quote-toggle", _CSV_QUOTE_TOGGLE_SOURCE, _CSV_QUOTE_TOGGLE_NAMES,
    )


def _row_end_kernels() -> dict[str, object]:
    return compile_kernel_group(
        "csv-find-row-ends", _CSV_FIND_ROW_ENDS_SOURCE, _CSV_FIND_ROW_ENDS_NAMES,
    )


def _delimiter_kernels() -> dict[str, object]:
    return compile_kernel_group(
        "csv-find-delimiters", _CSV_FIND_DELIMITERS_SOURCE, _CSV_FIND_DELIMITERS_NAMES,
    )


# ---------------------------------------------------------------------------
# Kernel launch helper (matches gpu_parse/numeric.py pattern)
# ---------------------------------------------------------------------------


def _launch_kernel(runtime, kernel, n: int, params) -> None:
    grid, block = runtime.launch_config(kernel, n)
    runtime.launch(kernel, grid=grid, block=block, params=params)


# ---------------------------------------------------------------------------
# Spatial column detection heuristics
# ---------------------------------------------------------------------------

# Canonical name sets for spatial column identification.
# All comparisons are done against lower-cased, stripped column names.
_LATITUDE_NAMES: frozenset[str] = frozenset({
    "latitude", "lat", "y", "lat_y", "point_y",
})
_LONGITUDE_NAMES: frozenset[str] = frozenset({
    "longitude", "lon", "lng", "x", "long", "lon_x", "point_x",
})
_GEOMETRY_NAMES: frozenset[str] = frozenset({
    "geometry", "geom", "wkt", "the_geom", "shape",
})


def _detect_spatial_columns(column_names: list[str]) -> dict[str, int]:
    """Identify spatial columns by name heuristics.

    Returns a dict with keys like ``"lat"``, ``"lon"``, ``"geom"``
    mapped to their 0-based column indices.  Returns an empty dict
    if no spatial columns are detected.

    Detection priority:
    1. Geometry/WKT column (takes precedence if both geom and lat/lon exist)
    2. Latitude + longitude pair
    """
    result: dict[str, int] = {}

    for idx, name in enumerate(column_names):
        lower = name.strip().lower()
        if lower in _LATITUDE_NAMES:
            result["lat"] = idx
        elif lower in _LONGITUDE_NAMES:
            result["lon"] = idx
        elif lower in _GEOMETRY_NAMES:
            result["geom"] = idx

    # If we have a geometry column, prefer it (drop partial lat/lon)
    if "geom" in result:
        result.pop("lat", None)
        result.pop("lon", None)
    # If we have only one of lat/lon, drop it (incomplete pair)
    elif ("lat" in result) != ("lon" in result):
        result.pop("lat", None)
        result.pop("lon", None)

    return result


def _parse_header_row(
    d_bytes: cp.ndarray,
    row_end_pos: int,
    delimiter: str,
) -> list[str]:
    """Extract column names from the first row (CPU-side, small data).

    This is the ONLY acceptable D->H copy in the CSV structural analysis
    pipeline.  The first row is typically a few hundred bytes.

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 byte array of the full CSV file.
    row_end_pos : int
        Byte offset of the first ``\\n`` (end of header row).
    delimiter : str
        Single-character field delimiter.
    """
    # Copy only the header row bytes D->H (small, one-time)
    header_bytes = cp.asnumpy(d_bytes[:row_end_pos])
    header_str = header_bytes.tobytes().decode("utf-8", errors="replace")

    # Strip trailing \r if present (Windows line endings)
    header_str = header_str.rstrip("\r")

    # Split by delimiter.  RFC 4180: fields may be quoted.
    # For the header row, strip surrounding quotes from each field name.
    raw_fields = header_str.split(delimiter)
    column_names: list[str] = []
    for field in raw_fields:
        field = field.strip()
        # Remove surrounding double quotes if present
        if len(field) >= 2 and field[0] == '"' and field[-1] == '"':
            # Unescape doubled quotes inside
            field = field[1:-1].replace('""', '"')
        column_names.append(field)

    return column_names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CsvStructuralResult:
    """Result of CSV structural analysis.

    All device arrays remain on GPU.  Only ``column_names`` and
    ``spatial_columns`` are host-side Python objects (derived from
    the small header row).

    Attributes
    ----------
    d_row_ends : cp.ndarray
        int64 positions of row-ending ``\\n`` characters, shape ``(n_rows_total,)``.
        Includes the header row end at index 0 if ``has_header`` was True.
    d_delimiters : cp.ndarray
        int64 positions of all delimiter characters outside quotes,
        shape ``(n_delimiters_total,)``.
    d_quote_parity : cp.ndarray
        uint8 per-byte quote state, shape ``(n_bytes,)``.
        0 = outside quoted field, 1 = inside.
    n_rows : int
        Number of data rows (excluding header if present).
    n_columns : int
        Number of columns (fields per row).
    column_names : list[str]
        Column names from the header row.  Empty list if no header.
    spatial_columns : dict[str, int]
        Detected spatial columns.  Keys are ``"lat"``, ``"lon"``,
        or ``"geom"`` mapped to 0-based column indices.
    """

    d_row_ends: cp.ndarray
    d_delimiters: cp.ndarray
    d_quote_parity: cp.ndarray
    n_rows: int
    n_columns: int
    column_names: list[str]
    spatial_columns: dict[str, int]


def csv_structural_analysis(
    d_bytes: cp.ndarray,
    delimiter: str = ",",
    has_header: bool = True,
) -> CsvStructuralResult:
    """Perform GPU-accelerated structural analysis of a CSV file.

    Identifies row boundaries, column boundaries, and header metadata
    for a device-resident CSV byte stream.  All structural scanning
    runs on the GPU; only the small header row is copied to host for
    column name extraction.

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 array of raw CSV file bytes, shape ``(n,)``.
    delimiter : str, default ``","``
        Single-character field delimiter.  Common values: ``","``
        (comma), ``"\\t"`` (tab), ``"|"`` (pipe).
    has_header : bool, default True
        If True, the first row is treated as a header containing
        column names.  If False, columns are named ``col_0``, ``col_1``, etc.

    Returns
    -------
    CsvStructuralResult
        Frozen dataclass with device-resident boundary arrays and
        host-side column metadata.

    Raises
    ------
    ValueError
        If the delimiter is not a single ASCII character, or if column
        counts are inconsistent across rows.

    Notes
    -----
    The CSV quoting rules follow RFC 4180:

    - Fields containing the delimiter, newline, or double-quote are
      enclosed in double quotes.
    - Literal double quotes inside quoted fields are escaped as ``""``
      (two consecutive double-quote characters).
    - Newline characters inside quoted fields are NOT row boundaries.

    The quote parity algorithm exploits the fact that ``""`` escaping
    naturally cancels in a cumulative-sum toggle: each ``"`` flips
    parity, so two consecutive ``"`` characters flip it twice, returning
    to the original state.  This is simpler than backslash-escape
    detection (used by JSON) because CSV has no backslash escaping.

    Examples
    --------
    >>> import cupy as cp
    >>> csv_bytes = b'name,lat,lon\\nAlice,40.7,-74.0\\nBob,34.0,-118.2\\n'
    >>> d_bytes = cp.frombuffer(csv_bytes, dtype=cp.uint8)
    >>> result = csv_structural_analysis(d_bytes)
    >>> result.n_rows
    2
    >>> result.n_columns
    3
    >>> result.column_names
    ['name', 'lat', 'lon']
    >>> result.spatial_columns
    {'lat': 1, 'lon': 2}
    """
    if len(delimiter) != 1:
        raise ValueError(
            f"Delimiter must be a single character, got {delimiter!r}"
        )
    delimiter_byte = ord(delimiter)
    if delimiter_byte > 127:
        raise ValueError(
            f"Delimiter must be ASCII, got {delimiter!r} (ord={delimiter_byte})"
        )

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    n = d_bytes.shape[0]
    n_i64 = np.int64(n)

    if n == 0:
        return CsvStructuralResult(
            d_row_ends=cp.empty(0, dtype=cp.int64),
            d_delimiters=cp.empty(0, dtype=cp.int64),
            d_quote_parity=cp.empty(0, dtype=cp.uint8),
            n_rows=0,
            n_columns=0,
            column_names=[],
            spatial_columns={},
        )

    # ------------------------------------------------------------------
    # Step 1: CSV quote parity
    # ------------------------------------------------------------------
    # CSV-specific: emit 1 at every " without backslash-escape checking.
    # Doubled quotes ("") naturally cancel in the cumsum parity.
    kernels_qt = _quote_toggle_kernels()
    d_toggle = cp.empty(n, dtype=cp.uint8)
    _launch_kernel(runtime, kernels_qt["csv_quote_toggle"], n, (
        (ptr(d_bytes), ptr(d_toggle), n_i64),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I64),
    ))
    # uint8 cumsum + bitwise AND for parity (no sync needed -- same stream).
    # uint8 overflow is safe because 256 is even, so parity bit is preserved.
    d_quote_parity = cp.cumsum(d_toggle, dtype=cp.uint8) & np.uint8(1)
    del d_toggle

    # ------------------------------------------------------------------
    # Step 2: Row boundary detection
    # ------------------------------------------------------------------
    kernels_re = _row_end_kernels()
    d_is_row_end = cp.empty(n, dtype=cp.uint8)
    _launch_kernel(runtime, kernels_re["csv_find_row_ends"], n, (
        (ptr(d_bytes), ptr(d_quote_parity), ptr(d_is_row_end), n_i64),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I64),
    ))
    d_row_ends = cp.flatnonzero(d_is_row_end).astype(cp.int64)
    del d_is_row_end

    # Handle file not ending with newline: treat EOF as implicit row end.
    # Check last byte on device without D->H transfer.
    last_byte_is_newline = bool(d_bytes[-1] == ord("\n"))
    if not last_byte_is_newline and n > 0:
        # Append (n-1) as a virtual row-end position at EOF.
        # This is the index of the last byte, marking the end of the last row.
        eof_pos = cp.array([n - 1], dtype=cp.int64)
        d_row_ends = cp.concatenate([d_row_ends, eof_pos])

    n_rows_total = d_row_ends.shape[0]
    if n_rows_total == 0:
        # Single row, no newline -- the entire file is one row
        d_row_ends = cp.array([n - 1], dtype=cp.int64)
        n_rows_total = 1

    # ------------------------------------------------------------------
    # Step 3: Column boundary detection (delimiters)
    # ------------------------------------------------------------------
    kernels_dl = _delimiter_kernels()
    d_is_delimiter = cp.empty(n, dtype=cp.uint8)
    _launch_kernel(runtime, kernels_dl["csv_find_delimiters"], n, (
        (ptr(d_bytes), ptr(d_quote_parity), ptr(d_is_delimiter), n_i64,
         np.int32(delimiter_byte)),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I64,
         KERNEL_PARAM_I32),
    ))
    d_delimiters = cp.flatnonzero(d_is_delimiter).astype(cp.int64)
    del d_is_delimiter

    # ------------------------------------------------------------------
    # Step 4: Column count verification
    # ------------------------------------------------------------------
    n_delimiters_total = d_delimiters.shape[0]

    # Count delimiters per row using device-side operations.
    # Strategy: for each delimiter, binary-search into d_row_ends to find
    # which row it belongs to, then count per row.
    if n_rows_total > 0 and n_delimiters_total > 0:
        # Each delimiter's row index = searchsorted(d_row_ends, d_delimiters, side='right')
        # This gives the row index (0-based) for each delimiter.
        d_delim_row = cp.searchsorted(d_row_ends, d_delimiters, side="right")
        # Count delimiters per row via bincount (Tier 2 CuPy)
        d_delims_per_row = cp.bincount(d_delim_row, minlength=n_rows_total)
        del d_delim_row

        # Verify consistent column count: all rows should have the same
        # number of delimiters.
        delims_min = int(d_delims_per_row.min())
        delims_max = int(d_delims_per_row.max())
        del d_delims_per_row

        if delims_min != delims_max:
            raise ValueError(
                f"Inconsistent column count: rows have between "
                f"{delims_min + 1} and {delims_max + 1} columns "
                f"(expected uniform {delims_max + 1})"
            )
        n_columns = delims_max + 1
    elif n_rows_total > 0:
        # No delimiters at all -- single-column file
        n_columns = 1
    else:
        n_columns = 0

    # ------------------------------------------------------------------
    # Step 5: Header parsing (CPU-side, small data, one-time D->H)
    # ------------------------------------------------------------------
    if has_header and n_rows_total > 0:
        # The first row end position marks the end of the header
        first_row_end = int(d_row_ends[0])
        column_names = _parse_header_row(d_bytes, first_row_end, delimiter)

        # Sanity check: header column count should match structural detection
        if len(column_names) != n_columns:
            raise ValueError(
                f"Header has {len(column_names)} fields but structural "
                f"analysis detected {n_columns} columns per row"
            )

        spatial_columns = _detect_spatial_columns(column_names)
        n_data_rows = n_rows_total - 1  # exclude header
    elif has_header and n_rows_total == 0:
        column_names = []
        spatial_columns = {}
        n_data_rows = 0
    else:
        # No header: generate synthetic column names
        column_names = [f"col_{i}" for i in range(n_columns)]
        spatial_columns = _detect_spatial_columns(column_names)
        n_data_rows = n_rows_total

    return CsvStructuralResult(
        d_row_ends=d_row_ends,
        d_delimiters=d_delimiters,
        d_quote_parity=d_quote_parity,
        n_rows=n_data_rows,
        n_columns=n_columns,
        column_names=column_names,
        spatial_columns=spatial_columns,
    )
