"""GPU CSV reader -- structural analysis and spatial geometry extraction.

GPU-accelerated CSV reader with two stages:

**Stage 1: Structural Analysis** (``csv_structural_analysis``)

Given a device-resident byte array containing a CSV file, identifies:

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

**Stage 2: Spatial Column Extraction** (``read_csv_gpu``)

Given the structural analysis result, extracts spatial geometry:

- **Lat/lon mode**: extracts numeric lat/lon columns, parses with
  ``parse_ascii_floats``, assembles as Point OwnedGeometryArray.
- **WKT mode**: extracts WKT column, concatenates with newline separators,
  delegates to ``read_wkt_gpu``.
- **WKB mode**: extracts hex-encoded WKB column (auto-detected), decodes
  hex to binary on CPU, delegates to ``decode_wkb_arrow_array_owned``
  which has a GPU fast path via pylibcudf.

All structural kernels are integer-only byte classification (no
floating-point computation), so no PrecisionPlan is needed per ADR-0002
-- same rationale as ``gpu_parse/structural.py``.  Coordinate parsing
delegates to ``parse_ascii_floats`` which always produces fp64 -- storage
precision is always fp64 per ADR-0002.

Tier classification (ADR-0033):
    - Quote toggle: Tier 1 (custom NVRTC -- byte classification)
    - Row end detection: Tier 1 (custom NVRTC -- byte + parity check)
    - Delimiter detection: Tier 1 (custom NVRTC -- byte + parity check)
    - Parity cumsum: Tier 2 (CuPy cumsum)
    - Position extraction: Tier 2 (CuPy flatnonzero)
    - Column count verification: Tier 2 (CuPy element-wise)
    - Header parsing: CPU (small data, one-time)
    - Field span extraction: Tier 2 (CuPy index arithmetic)
    - Quote stripping: Tier 2 (CuPy element-wise byte comparison)
    - Numeric parsing: delegates to gpu_parse.parse_ascii_floats (Tier 1)
    - WKT field concatenation: Tier 2 (CuPy scatter/copy)
    - WKT parsing: delegates to wkt_gpu.read_wkt_gpu
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
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.io.csv_gpu_kernels import (
    _CSV_FIND_DELIMITERS_NAMES,
    _CSV_FIND_DELIMITERS_SOURCE,
    _CSV_FIND_ROW_ENDS_NAMES,
    _CSV_FIND_ROW_ENDS_SOURCE,
    _CSV_HEX_WKB_NAMES,
    _CSV_HEX_WKB_SOURCE,
    _CSV_QUOTE_TOGGLE_NAMES,
    _CSV_QUOTE_TOGGLE_SOURCE,
)
from vibespatial.io.gpu_parse.numeric import parse_ascii_floats

if TYPE_CHECKING:
    import cupy as cp

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

# ctypes for int64 kernel params (files > 2 GB)
KERNEL_PARAM_I64 = ctypes.c_longlong

# ---------------------------------------------------------------------------
# NVRTC warmup registration (ADR-0034 Level 2)
# ---------------------------------------------------------------------------

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("csv-quote-toggle", _CSV_QUOTE_TOGGLE_SOURCE, _CSV_QUOTE_TOGGLE_NAMES),
    ("csv-find-row-ends", _CSV_FIND_ROW_ENDS_SOURCE, _CSV_FIND_ROW_ENDS_NAMES),
    ("csv-find-delimiters", _CSV_FIND_DELIMITERS_SOURCE, _CSV_FIND_DELIMITERS_NAMES),
    ("csv-hex-wkb", _CSV_HEX_WKB_SOURCE, _CSV_HEX_WKB_NAMES),
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


def _hex_wkb_kernels() -> dict[str, object]:
    return compile_kernel_group(
        "csv-hex-wkb", _CSV_HEX_WKB_SOURCE, _CSV_HEX_WKB_NAMES,
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
    "geometry", "geom", "wkt", "the_geom", "shape", "wkb",
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


# ---------------------------------------------------------------------------
# Stage 2: Spatial column extraction and geometry assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CsvGpuResult:
    """Result of GPU CSV spatial reading.

    Attributes
    ----------
    geometry : OwnedGeometryArray
        Device-resident geometry array.  For lat/lon mode, contains
        Point geometries.  For WKT mode, contains whatever types
        were in the WKT column.
    n_rows : int
        Number of data rows read.
    attributes : dict[str, list[str]] or None
        Non-spatial columns extracted as host-resident string lists.
        Keys are column names, values are per-row string values.
        None when there are no non-spatial columns.
    """

    geometry: OwnedGeometryArray
    n_rows: int
    attributes: dict[str, list[str]] | None = None


def _extract_field_spans(
    d_bytes: cp.ndarray,
    structural: CsvStructuralResult,
    col_idx: int,
    has_header: bool,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Extract per-row byte spans for a specific column.

    Given the structural analysis result, computes (field_starts,
    field_ends) byte offset arrays for column ``col_idx`` across all
    data rows.  All computation is device-resident (Tier 2 CuPy).

    The returned spans are half-open: ``[field_starts[i], field_ends[i])``.

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 byte array of the full CSV file.
    structural : CsvStructuralResult
        Result from ``csv_structural_analysis``.
    col_idx : int
        0-based column index to extract.
    has_header : bool
        Whether the CSV has a header row (affects row/delimiter indexing).

    Returns
    -------
    d_field_starts : cp.ndarray
        int64 byte offsets of field start (inclusive), shape ``(n_rows,)``.
    d_field_ends : cp.ndarray
        int64 byte offsets of field end (exclusive), shape ``(n_rows,)``.
    """
    n_rows = structural.n_rows
    n_columns = structural.n_columns
    d_row_ends = structural.d_row_ends
    d_delimiters = structural.d_delimiters
    delims_per_row = n_columns - 1

    if n_rows == 0:
        return cp.empty(0, dtype=cp.int64), cp.empty(0, dtype=cp.int64)

    # Row indices into d_row_ends for data rows.
    # When has_header=True, data rows are d_row_ends[1..n_rows_total-1],
    # so data row i corresponds to d_row_ends index (i + 1).
    # When has_header=False, data row i corresponds to d_row_ends index i.
    row_offset = 1 if has_header else 0

    # Build data row indices: [0, 1, ..., n_rows - 1]
    d_data_idx = cp.arange(n_rows, dtype=cp.int64)

    # Row start byte positions (byte after previous row's \n)
    if has_header:
        # Data row i: row starts at d_row_ends[i] + 1  (prev row is i in d_row_ends)
        d_row_starts = d_row_ends[d_data_idx] + 1
    else:
        # Data row 0: starts at byte 0
        # Data row i > 0: starts at d_row_ends[i - 1] + 1
        d_row_starts = cp.empty(n_rows, dtype=cp.int64)
        if n_rows > 0:
            d_row_starts[0] = 0
        if n_rows > 1:
            d_row_starts[1:] = d_row_ends[d_data_idx[1:] - 1] + 1

    # Row end byte positions -- used as exclusive field end for the last column.
    # d_row_ends stores the position of \n for normal rows, but for the
    # virtual EOF row end (file not ending with \n), it stores the index
    # of the last content byte.  We need exclusive ends, so: if the byte
    # at the row end position is \n, use that position as the exclusive end
    # (field content stops before the \n).  If it is NOT \n (EOF case),
    # use position + 1 (the byte IS part of the field content).
    d_raw_row_ends = d_row_ends[d_data_idx + row_offset]
    d_byte_at_end = d_bytes[d_raw_row_ends]
    d_is_newline = d_byte_at_end == np.uint8(ord("\n"))
    d_row_end_positions = cp.where(d_is_newline, d_raw_row_ends, d_raw_row_ends + 1)

    if delims_per_row == 0:
        # Single-column file: field = entire row (up to \n or EOF+1)
        return d_row_starts, d_row_end_positions

    # Delimiter base index for each data row.
    # Delimiter array is flat across ALL rows (including header).
    # Row r (0-based in d_row_ends) has delimiters at indices
    # [r * delims_per_row, (r + 1) * delims_per_row).
    # For data row i, row r = row_offset + i.
    d_delim_base = (d_data_idx + np.int64(row_offset)) * np.int64(delims_per_row)

    # Field start
    if col_idx == 0:
        d_field_starts = d_row_starts
    else:
        # Start = delimiter[delim_base + col_idx - 1] + 1
        d_field_starts = d_delimiters[d_delim_base + np.int64(col_idx - 1)] + 1

    # Field end (exclusive)
    if col_idx == n_columns - 1:
        d_field_ends = d_row_end_positions
    else:
        # End = delimiter[delim_base + col_idx]
        d_field_ends = d_delimiters[d_delim_base + np.int64(col_idx)]

    return d_field_starts, d_field_ends


def _strip_quotes_from_spans(
    d_bytes: cp.ndarray,
    d_starts: cp.ndarray,
    d_ends: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Strip surrounding double-quote characters from field spans.

    If a field starts with ``"`` and ends with ``"`` (the byte before the
    exclusive end), the span is narrowed by one byte on each side.
    Fields that are not quoted or are too short to be quoted (< 2 bytes)
    are left unchanged.  All computation is device-resident (Tier 2 CuPy).

    Also strips ``\\r`` from the end of fields (Windows line endings).

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 byte array of the full CSV file.
    d_starts : cp.ndarray
        int64 field start positions (inclusive), shape ``(n,)``.
    d_ends : cp.ndarray
        int64 field end positions (exclusive), shape ``(n,)``.

    Returns
    -------
    d_starts_out : cp.ndarray
        Adjusted int64 start positions, shape ``(n,)``.
    d_ends_out : cp.ndarray
        Adjusted int64 end positions, shape ``(n,)``.
    """
    if d_starts.size == 0:
        return d_starts, d_ends

    # Strip trailing \r (Windows line endings: field end may point at \r)
    # Check byte at d_ends - 1 (last byte of field)
    d_last_valid = d_ends - 1
    # Clamp to valid range
    d_last_valid_clamped = cp.maximum(d_last_valid, d_starts)
    d_is_cr = d_bytes[d_last_valid_clamped] == ord("\r")
    # Only strip if field is non-empty after stripping
    d_can_strip_cr = d_is_cr & (d_ends - 1 > d_starts)
    d_ends_stripped = cp.where(d_can_strip_cr, d_ends - 1, d_ends)

    # Now strip surrounding quotes.
    # Check if first byte is " and last byte (in stripped range) is "
    d_field_len = d_ends_stripped - d_starts
    d_long_enough = d_field_len >= 2

    # Get first and last bytes (safe: only access when long_enough)
    d_first_byte = d_bytes[d_starts]
    d_last_byte_pos = cp.maximum(d_ends_stripped - 1, d_starts)
    d_last_byte = d_bytes[d_last_byte_pos]

    d_is_quoted = (
        d_long_enough
        & (d_first_byte == ord('"'))
        & (d_last_byte == ord('"'))
    )

    d_starts_out = cp.where(d_is_quoted, d_starts + 1, d_starts)
    d_ends_out = cp.where(d_is_quoted, d_ends_stripped - 1, d_ends_stripped)

    return d_starts_out, d_ends_out


_HEX_CHARS: frozenset[int] = frozenset(
    b"0123456789abcdefABCDEF"
)


def _detect_geom_format(
    d_bytes: cp.ndarray,
    d_starts: cp.ndarray,
    d_ends: cp.ndarray,
) -> str:
    """Peek at the first non-empty geometry value to detect WKT vs hex WKB.

    Transfers only a few bytes D->H (the first non-empty field) to classify
    the geometry encoding.  This is a one-time, small transfer.

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 byte array of the full CSV file.
    d_starts : cp.ndarray
        int64 field start positions (inclusive), shape ``(n_rows,)``.
    d_ends : cp.ndarray
        int64 field end positions (exclusive), shape ``(n_rows,)``.

    Returns
    -------
    str
        ``"wkt"`` or ``"wkb"``.
    """
    n_rows = d_starts.shape[0]
    if n_rows == 0:
        return "wkt"

    # Find the first non-empty field on device.
    d_field_lens = (d_ends - d_starts).astype(cp.int64)
    d_nonempty_mask = d_field_lens > 0
    d_nonempty_idx = cp.flatnonzero(d_nonempty_mask)

    if d_nonempty_idx.shape[0] == 0:
        return "wkt"

    first_idx = int(d_nonempty_idx[0])
    start = int(d_starts[first_idx])
    end = int(d_ends[first_idx])
    # Transfer at most 64 bytes -- enough to classify the field.
    peek_end = min(end, start + 64)
    sample_bytes: bytes = cp.asnumpy(d_bytes[start:peek_end]).tobytes()

    # Strip leading whitespace.
    stripped = sample_bytes.lstrip()
    if not stripped:
        return "wkt"

    # WKB detection: hex-encoded WKB always starts with byte-order marker
    # "00" (big-endian) or "01" (little-endian), has even length (two hex
    # chars per byte), and the rest must be hex digits.  This is much more
    # specific than "starts with a hex digit" — it rules out city names,
    # IDs, or other hex-ish strings in a geometry column.
    if (
        len(stripped) >= 10  # minimum WKB point is 42 hex chars; 10 is generous floor
        and stripped[0:2] in (b"00", b"01")
        and len(stripped) % 2 == 0
        and all(b in _HEX_CHARS for b in stripped)
    ):
        return "wkb"

    # Otherwise assume WKT (backwards-compatible default).
    # This handles: POINT(...), LINESTRING(...), POLYGON(...), etc.
    # as well as any unrecognized format (will fail at parse time with
    # a clear error from the WKT parser).
    return "wkt"


def _extract_wkb_and_parse(
    d_bytes: cp.ndarray,
    d_starts: cp.ndarray,
    d_ends: cp.ndarray,
) -> OwnedGeometryArray:
    """Extract hex-encoded WKB fields and decode via the GPU WKB pipeline.

    Fully device-resident: hex-to-binary decode runs on GPU via an NVRTC
    kernel, then the resulting binary WKB column is fed directly into the
    pylibcudf-based GPU WKB decode pipeline -- no D->H transfers.

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 byte array of the full CSV file.
    d_starts : cp.ndarray
        int64 field start positions (inclusive), shape ``(n_rows,)``.
    d_ends : cp.ndarray
        int64 field end positions (exclusive), shape ``(n_rows,)``.

    Returns
    -------
    OwnedGeometryArray
        Geometry array decoded from hex WKB.
    """
    import pylibcudf as plc

    from vibespatial.cuda.cccl_primitives import exclusive_sum
    from vibespatial.io.pylibcudf import _decode_pylibcudf_wkb_general_column_to_owned

    n_rows = d_starts.shape[0]

    if n_rows == 0:
        import pyarrow as pa

        from vibespatial.io.wkb import decode_wkb_arrow_array_owned

        empty_arrow = pa.array([], type=pa.binary())
        return decode_wkb_arrow_array_owned(empty_arrow)

    runtime = get_cuda_runtime()
    kernels = _hex_wkb_kernels()

    # --- Pass 1: count binary bytes per row + validity ---
    d_counts = cp.zeros(n_rows, dtype=cp.int32)
    d_valid = cp.zeros(n_rows, dtype=cp.uint8)

    ptr = runtime.pointer
    count_params = (
        (ptr(d_bytes), ptr(d_starts), ptr(d_ends),
         ptr(d_counts), ptr(d_valid), n_rows),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
    )
    _launch_kernel(runtime, kernels["csv_hex_wkb_count"], n_rows, count_params)

    # --- Prefix sum for output offsets ---
    d_offsets_body = exclusive_sum(d_counts, synchronize=False)

    # Build full offsets array (n_rows + 1) for pylibcudf column.
    d_offsets = cp.empty(n_rows + 1, dtype=cp.int32)
    d_offsets[:n_rows] = d_offsets_body
    d_offsets[n_rows] = d_offsets_body[-1] + d_counts[-1]
    # Read total on host (single scalar sync).
    total_bytes = int(d_offsets[n_rows])

    if total_bytes == 0:
        # All rows invalid -- return empty/all-null geometry.
        import pyarrow as pa

        from vibespatial.io.wkb import decode_wkb_arrow_array_owned

        empty_arrow = pa.array([None] * n_rows, type=pa.binary())
        return decode_wkb_arrow_array_owned(empty_arrow)

    # --- Pass 2: decode hex to binary ---
    d_payload = cp.empty(total_bytes, dtype=cp.uint8)
    decode_params = (
        (ptr(d_bytes), ptr(d_starts), ptr(d_ends),
         ptr(d_offsets), ptr(d_valid), ptr(d_payload), n_rows),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_I32),
    )
    _launch_kernel(runtime, kernels["csv_hex_wkb_decode"], n_rows, decode_params)
    runtime.synchronize()

    # --- Build pylibcudf column on device ---
    offsets_column = plc.Column.from_cuda_array_interface(d_offsets)
    column = plc.Column(
        plc.types.DataType(plc.types.TypeId.STRING),
        n_rows,
        plc.gpumemoryview(d_payload),
        None,
        0,
        0,
        [offsets_column],
    )

    # Attach null mask for invalid rows.
    null_count = int(n_rows - int(d_valid.sum()))
    if null_count:
        validity_bool = d_valid.astype(cp.uint8)
        validity_bytes = cp.packbits(validity_bool, bitorder="little")  # zcopy:ok terminal
        column = column.with_mask(plc.gpumemoryview(validity_bytes), null_count)

    return _decode_pylibcudf_wkb_general_column_to_owned(column)


def _assemble_point_geometry(
    d_lon: cp.ndarray,
    d_lat: cp.ndarray,
    n_rows: int,
) -> OwnedGeometryArray:
    """Assemble lat/lon arrays into a Point OwnedGeometryArray.

    GIS convention: x = longitude, y = latitude.

    Parameters
    ----------
    d_lon : cp.ndarray
        Device-resident float64 longitude values, shape ``(n_rows,)``.
    d_lat : cp.ndarray
        Device-resident float64 latitude values, shape ``(n_rows,)``.
    n_rows : int
        Number of points.

    Returns
    -------
    OwnedGeometryArray
        Device-resident Point geometry array.
    """
    from vibespatial.io.pylibcudf import _build_device_single_family_owned

    # All rows are valid points (no null handling in lat/lon mode)
    d_validity = cp.ones(n_rows, dtype=cp.bool_)
    d_empty_mask = cp.zeros(n_rows, dtype=cp.bool_)

    # Point geometry_offsets: identity [0, 1, 2, ..., n_rows]
    d_geom_offsets = cp.arange(n_rows + 1, dtype=cp.int32)

    return _build_device_single_family_owned(
        family=GeometryFamily.POINT,
        validity_device=d_validity,
        x_device=d_lon,
        y_device=d_lat,
        geometry_offsets_device=d_geom_offsets,
        empty_mask_device=d_empty_mask,
        detail="GPU CSV parse (lat/lon -> Point)",
    )


def _extract_wkt_and_parse(
    d_bytes: cp.ndarray,
    d_starts: cp.ndarray,
    d_ends: cp.ndarray,
) -> OwnedGeometryArray:
    """Extract WKT field bytes and delegate to read_wkt_gpu.

    Concatenates per-row WKT fields with newline separators into a
    single device byte array, then delegates to the WKT GPU parser.
    All intermediate work is device-resident (Tier 2 CuPy scatter/copy).

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 byte array of the full CSV file.
    d_starts : cp.ndarray
        int64 field start positions (inclusive), shape ``(n_rows,)``.
    d_ends : cp.ndarray
        int64 field end positions (exclusive), shape ``(n_rows,)``.

    Returns
    -------
    OwnedGeometryArray
        Device-resident geometry array from WKT parsing.
    """
    from vibespatial.io.wkt_gpu import read_wkt_gpu

    n_rows = d_starts.shape[0]

    if n_rows == 0:
        # Empty input: create empty WKT bytes and delegate
        return read_wkt_gpu(cp.empty(0, dtype=cp.uint8))

    # Compute per-field lengths
    d_field_lens = (d_ends - d_starts).astype(cp.int64)

    # Total output size: sum of field lengths + n_rows newlines (one per row)
    # Using int64 to handle large files safely.
    total_field_bytes = int(d_field_lens.sum())
    total_output_bytes = total_field_bytes + n_rows

    # Build output offsets: where each field's bytes start in the output.
    # Each entry in the output is: field_bytes + \n
    # So entry i starts at cumsum of (field_lens[0..i-1] + 1 each).
    d_entry_lens = d_field_lens + 1  # +1 for the \n separator
    d_output_offsets = cp.zeros(n_rows + 1, dtype=cp.int64)
    cp.cumsum(d_entry_lens, out=d_output_offsets[1:])

    # Allocate output buffer
    d_output = cp.empty(total_output_bytes, dtype=cp.uint8)

    # Copy each field's bytes to the output using a vectorized approach.
    # For each byte position in the output, we need to determine which
    # field it belongs to and what source byte to copy.
    #
    # Strategy: build a flat index map.  For each field i, bytes go to
    # output[d_output_offsets[i]:d_output_offsets[i] + d_field_lens[i]]
    # from source d_bytes[d_starts[i]:d_ends[i]], then a \n at the end.
    #
    # We generate flat source indices and scatter them.
    # This avoids a Python loop over rows.

    # For each field i, we need d_field_lens[i] source byte copies + 1 newline.
    # Total operations = total_output_bytes (one per output byte).

    # Build per-output-byte metadata using searchsorted to map output
    # positions back to field indices.
    d_out_positions = cp.arange(total_output_bytes, dtype=cp.int64)
    # Which field does each output byte belong to?
    d_field_idx = cp.searchsorted(d_output_offsets[1:], d_out_positions, side="right")
    # d_field_idx[pos] is the index of the field that output byte `pos` belongs to.

    # Position within the field's output chunk:
    # local_pos = pos - d_output_offsets[d_field_idx]
    d_local_pos = d_out_positions - d_output_offsets[d_field_idx]

    # A byte is a newline separator if local_pos == d_field_lens[d_field_idx]
    # (i.e., it's the last byte of the entry, after the field content).
    d_is_newline = d_local_pos >= d_field_lens[d_field_idx]

    # Source byte position: d_starts[d_field_idx] + local_pos
    d_src_pos = d_starts[d_field_idx] + d_local_pos

    # Clamp source positions to valid range for the newline bytes
    # (they won't be used because we overwrite with \n, but we need
    # valid indices for the gather operation).
    d_src_pos_safe = cp.minimum(d_src_pos, np.int64(d_bytes.shape[0] - 1))

    # Gather source bytes
    d_output[:] = d_bytes[d_src_pos_safe]

    # Overwrite newline positions
    d_output[d_is_newline] = ord("\n")

    return read_wkt_gpu(d_output)


def _extract_nonspatial_columns(
    d_bytes: cp.ndarray,
    structural: CsvStructuralResult,
    spatial_column_indices: set[int],
) -> dict[str, list[str]] | None:
    """Extract all non-spatial columns as host-resident string lists.

    For each column whose index is NOT in ``spatial_column_indices``,
    extracts field byte spans on the GPU, transfers to host, and decodes
    as UTF-8 strings.  This is the correct approach for attribute data
    (names, IDs, categories) which is inherently text and host-resident.

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 byte array of the full CSV file.
    structural : CsvStructuralResult
        Result from ``csv_structural_analysis``.
    spatial_column_indices : set[int]
        Set of 0-based column indices that are spatial (lat, lon, geom)
        and should be excluded from the result.

    Returns
    -------
    dict[str, list[str]] or None
        Column name -> list of string values.  None if there are no
        non-spatial columns.
    """
    n_rows = structural.n_rows
    n_columns = structural.n_columns
    column_names = structural.column_names

    nonspatial_indices = [
        i for i in range(n_columns) if i not in spatial_column_indices
    ]
    if not nonspatial_indices or n_rows == 0:
        return None

    # ---- Phase 1: compute all field spans on device ----
    # Keeps all GPU work batched before any D->H transfer.
    device_spans: list[tuple[int, cp.ndarray, cp.ndarray]] = []
    for col_idx in nonspatial_indices:
        d_starts, d_ends = _extract_field_spans(
            d_bytes, structural, col_idx, has_header=True,
        )
        d_starts, d_ends = _strip_quotes_from_spans(d_bytes, d_starts, d_ends)
        device_spans.append((col_idx, d_starts, d_ends))

    # ---- Phase 2: bulk D->H transfer ----
    # Single transfer for the byte payload; batch span arrays together.
    h_bytes: bytes = cp.asnumpy(d_bytes).tobytes()
    host_spans: list[tuple[int, np.ndarray, np.ndarray]] = [
        (col_idx, cp.asnumpy(d_s), cp.asnumpy(d_e))
        for col_idx, d_s, d_e in device_spans
    ]

    # ---- Phase 3: host-side decode (pure CPU, no device calls) ----
    attributes: dict[str, list[str]] = {}
    for col_idx, h_starts, h_ends in host_spans:
        col_name = column_names[col_idx]
        values: list[str] = []
        for i in range(n_rows):
            s = int(h_starts[i])
            e = int(h_ends[i])
            if s >= e:
                values.append("")
            else:
                field_bytes = h_bytes[s:e]
                # Strip \r from Windows line endings and decode
                value = field_bytes.decode("utf-8", errors="replace").rstrip("\r")
                # Unescape doubled quotes inside quoted CSV fields
                if '""' in value:
                    value = value.replace('""', '"')
                values.append(value)
        attributes[col_name] = values

    return attributes if attributes else None


def read_csv_gpu(
    d_bytes: cp.ndarray,
    *,
    delimiter: str = ",",
    lat_col: str | None = None,
    lon_col: str | None = None,
    geom_col: str | None = None,
) -> CsvGpuResult:
    """Read a CSV file on GPU and extract spatial geometry.

    Performs structural analysis followed by spatial column extraction
    and geometry assembly.  Supports two modes:

    1. **Lat/lon mode**: When ``lat_col`` and ``lon_col`` are specified
       (or auto-detected), extracts numeric latitude and longitude
       columns and assembles them as Point geometries.

    2. **WKT/WKB mode**: When ``geom_col`` is specified (or auto-detected),
       extracts the geometry column.  Hex-encoded WKB is auto-detected
       and decoded via the GPU WKB pipeline.  WKT text is parsed via the
       GPU WKT parser.

    All computation is device-resident.  The only D->H transfers are
    the small header row (for column name extraction) and the structural
    metadata in the OwnedGeometryArray (offsets, validity -- KB-scale).

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 array of raw CSV file bytes, shape ``(n,)``.
    delimiter : str, default ``","``
        Single-character field delimiter.
    lat_col : str or None, default None
        Name of the latitude column.  If None, auto-detected from
        header names.
    lon_col : str or None, default None
        Name of the longitude column.  If None, auto-detected from
        header names.
    geom_col : str or None, default None
        Name of the geometry column (WKT or hex-encoded WKB).  If None,
        auto-detected from header names.  The format (WKT vs hex WKB)
        is auto-detected from field content.

    Returns
    -------
    CsvGpuResult
        Frozen dataclass with ``geometry`` (OwnedGeometryArray) and
        ``n_rows`` (int).

    Raises
    ------
    ValueError
        If no spatial columns can be identified (neither lat/lon pair
        nor WKT geometry column), or if specified column names are not
        found in the header.

    Notes
    -----
    GIS convention: x = longitude, y = latitude.  The Point geometry
    array stores longitude in the x coordinate and latitude in y.

    Examples
    --------
    >>> import cupy as cp
    >>> csv_bytes = b'name,lat,lon\\nAlice,40.7,-74.0\\nBob,34.0,-118.2\\n'
    >>> d_bytes = cp.frombuffer(csv_bytes, dtype=cp.uint8)
    >>> result = read_csv_gpu(d_bytes)
    >>> result.n_rows
    2
    >>> result.geometry.row_count
    2
    """
    # ------------------------------------------------------------------
    # Stage 1: Structural analysis
    # ------------------------------------------------------------------
    structural = csv_structural_analysis(d_bytes, delimiter=delimiter, has_header=True)

    if structural.n_rows == 0:
        # No data rows -- return empty geometry
        from vibespatial.io.pylibcudf import _build_device_single_family_owned

        d_validity = cp.empty(0, dtype=cp.bool_)
        d_x = cp.empty(0, dtype=cp.float64)
        d_y = cp.empty(0, dtype=cp.float64)
        d_geom_offsets = cp.zeros(1, dtype=cp.int32)
        d_empty_mask = cp.empty(0, dtype=cp.bool_)
        geometry = _build_device_single_family_owned(
            family=GeometryFamily.POINT,
            validity_device=d_validity,
            x_device=d_x,
            y_device=d_y,
            geometry_offsets_device=d_geom_offsets,
            empty_mask_device=d_empty_mask,
            detail="GPU CSV parse (empty)",
        )
        return CsvGpuResult(geometry=geometry, n_rows=0)

    # ------------------------------------------------------------------
    # Stage 2: Resolve spatial columns
    # ------------------------------------------------------------------
    spatial = dict(structural.spatial_columns)  # mutable copy
    column_names = structural.column_names

    # User overrides: look up column indices by name
    if geom_col is not None:
        if geom_col not in column_names:
            raise ValueError(
                f"Geometry column {geom_col!r} not found in header. "
                f"Available columns: {column_names}"
            )
        spatial = {"geom": column_names.index(geom_col)}
    elif lat_col is not None or lon_col is not None:
        if lat_col is None or lon_col is None:
            raise ValueError(
                "Both lat_col and lon_col must be specified together, "
                f"got lat_col={lat_col!r}, lon_col={lon_col!r}"
            )
        if lat_col not in column_names:
            raise ValueError(
                f"Latitude column {lat_col!r} not found in header. "
                f"Available columns: {column_names}"
            )
        if lon_col not in column_names:
            raise ValueError(
                f"Longitude column {lon_col!r} not found in header. "
                f"Available columns: {column_names}"
            )
        spatial = {
            "lat": column_names.index(lat_col),
            "lon": column_names.index(lon_col),
        }

    if not spatial:
        raise ValueError(
            "No spatial columns detected or specified.  Provide "
            "lat_col/lon_col or geom_col, or use column names that "
            "match spatial heuristics (e.g., 'lat', 'lon', 'geometry')."
        )

    # ------------------------------------------------------------------
    # Stage 3: Extract geometry
    # ------------------------------------------------------------------
    if "geom" in spatial:
        # Geometry column mode: auto-detect WKT vs hex WKB
        geom_idx = spatial["geom"]
        spatial_indices = {geom_idx}
        d_starts, d_ends = _extract_field_spans(d_bytes, structural, geom_idx, has_header=True)
        d_starts, d_ends = _strip_quotes_from_spans(d_bytes, d_starts, d_ends)
        fmt = _detect_geom_format(d_bytes, d_starts, d_ends)
        if fmt == "wkb":
            geometry = _extract_wkb_and_parse(d_bytes, d_starts, d_ends)
        else:
            geometry = _extract_wkt_and_parse(d_bytes, d_starts, d_ends)
    else:
        # Lat/lon mode
        lat_idx = spatial["lat"]
        lon_idx = spatial["lon"]
        spatial_indices = {lat_idx, lon_idx}

        # Extract lat field spans
        d_lat_starts, d_lat_ends = _extract_field_spans(d_bytes, structural, lat_idx, has_header=True)
        d_lat_starts, d_lat_ends = _strip_quotes_from_spans(d_bytes, d_lat_starts, d_lat_ends)

        # Extract lon field spans
        d_lon_starts, d_lon_ends = _extract_field_spans(d_bytes, structural, lon_idx, has_header=True)
        d_lon_starts, d_lon_ends = _strip_quotes_from_spans(d_bytes, d_lon_starts, d_lon_ends)

        # Parse numeric values (delegates to gpu_parse NVRTC kernel)
        d_lat = parse_ascii_floats(d_bytes, d_lat_starts, d_lat_ends)
        d_lon = parse_ascii_floats(d_bytes, d_lon_starts, d_lon_ends)

        # Assemble Point geometry: x = longitude, y = latitude (GIS convention)
        geometry = _assemble_point_geometry(d_lon, d_lat, structural.n_rows)

    # ------------------------------------------------------------------
    # Stage 4: Extract non-spatial attribute columns
    # ------------------------------------------------------------------
    attributes = _extract_nonspatial_columns(
        d_bytes, structural, spatial_column_indices=spatial_indices,
    )

    return CsvGpuResult(
        geometry=geometry, n_rows=structural.n_rows, attributes=attributes,
    )
