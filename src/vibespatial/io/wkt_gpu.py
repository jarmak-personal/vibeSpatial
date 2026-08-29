"""GPU WKT reader -- structural analysis, coordinate extraction, and assembly.

GPU-accelerated WKT parser.  Given a device-resident byte array
containing one or more WKT geometries (one per line), this module
performs:

1. **Fixed-capacity row discovery** -- detect newline-delimited geometry
   starts without a data-dependent selection.
2. **Geometry type classification** -- a custom NVRTC kernel scans the
   start of each geometry string and emits a family tag (POINT=0,
   LINESTRING=1, POLYGON=2, MULTIPOINT=3, MULTILINESTRING=4,
   MULTIPOLYGON=5) plus an EMPTY flag.  Handles case-insensitive
   matching and EWKT ``SRID=NNNN;`` prefixes.
3. **Capacity count/validation** -- one row-parallel kernel validates numeric
   grammar and topology while counting family output sizes.
4. **Exact numeric conversion** -- fixed-capacity token spans feed an NVRTC
   binary64 parser with an exact common path and midpoint refinement.
5. **Planning boundary** -- one compact D2H packet preserves synchronous error
   semantics and provides exact family allocation totals.
6. **Direct family scatter** -- a second kernel writes parsed coordinates into
   exact owned buffers without data-dependent CuPy selection.

Geometry text and coordinates remain device-resident. The only successful
parse transfer is the explicit semantic-validation/allocation packet.

Tier classification (ADR-0033):
    - Row discovery: Tier 2 (fixed-shape CuPy scan + Tier 1 scatter)
    - Type classification: Tier 1 (custom NVRTC -- text-specific prefix matching)
    - Count/validation: Tier 1 (custom NVRTC -- row-capacity parse)
    - Numeric conversion: Tier 1 exact parse and rounding refinement
    - Coordinate/topology scatter: Tier 1 (custom NVRTC -- family-specialized)
    - Offset building: Tier 2 (CuPy cumsum) / CCCL exclusive_sum
    - Assembly: follows geojson_gpu.py patterns

Precision (ADR-0002):
    Structural and counting kernels are integer-only byte classification.
    No floating-point coordinate computation occurs in those kernels, so
    no PrecisionPlan is needed (same rationale as gpu_parse/structural.py).
    Coordinate parsing always produces fp64 storage per ADR-0002.
"""

from __future__ import annotations

import ctypes
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.io.gpu_parse.structural import bracket_depth

if TYPE_CHECKING:
    import cupy as cp

from vibespatial.io.wkt_gpu_kernels import (
    _WKT_CAPACITY_NAMES,
    _WKT_CAPACITY_SOURCE,
    _WKT_CLASSIFY_NAMES,
    _WKT_CLASSIFY_SOURCE,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

# ctypes for int64 kernel params (files > 2 GB)
KERNEL_PARAM_I64 = ctypes.c_longlong
_WKT_TOKEN_POSITION_DTYPE = np.int64


class _GpuWktOnInvalidError(ValueError):
    """A native WKT semantic failure that must reach the public caller."""


class _GpuWktCompatibilityDecline(_GpuWktOnInvalidError):
    """A WKT grammar/family failure requiring public compatibility parsing."""


def _wkt_device_to_host(device_array: object, *, reason: str) -> np.ndarray:
    return get_cuda_runtime().copy_device_to_host(device_array, reason=reason)


# ---------------------------------------------------------------------------
# NVRTC warmup registration (ADR-0034 Level 2)
# ---------------------------------------------------------------------------
# Register all WKT kernels for background precompilation.
# bracket_depth warmup for WKT parentheses is handled lazily by
# structural.py's _get_depth_kernels on first use (cached by char pair).

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup(
    [
        ("wkt-capacity-parse", _WKT_CAPACITY_SOURCE, _WKT_CAPACITY_NAMES),
        ("wkt-classify-type", _WKT_CLASSIFY_SOURCE, _WKT_CLASSIFY_NAMES),
    ]
)

# CCCL warmup for exclusive_sum used in offset building
from vibespatial.cuda.cccl_precompile import request_warmup  # noqa: E402

request_warmup(["exclusive_scan_i32"])


# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------


def _classify_type_kernels() -> dict:
    """Compile (or retrieve cached) WKT type classification kernel."""
    return compile_kernel_group(
        "wkt-classify-type",
        _WKT_CLASSIFY_SOURCE,
        _WKT_CLASSIFY_NAMES,
    )


def _wkt_capacity_kernels() -> dict:
    """Compile the fixed-capacity WKT count/validate/scatter pipeline."""
    return compile_kernel_group(
        "wkt-capacity-parse",
        _WKT_CAPACITY_SOURCE,
        _WKT_CAPACITY_NAMES,
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


_WKT_TAG_TO_FAMILY = {
    0: GeometryFamily.POINT,
    1: GeometryFamily.LINESTRING,
    2: GeometryFamily.POLYGON,
    3: GeometryFamily.MULTIPOINT,
    4: GeometryFamily.MULTILINESTRING,
    5: GeometryFamily.MULTIPOLYGON,
}


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
    - 2D geometry only; Z/M/ZM suffixes are classified as unsupported
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
        d_bytes,
        d_quote_parity,
        open_chars="(",
        close_chars=")",
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
    d_is_newline = d_bytes == ord("\n")
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
        d_non_empty = (d_start_bytes != ord("\n")) & (d_start_bytes != ord("\r"))
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


# ---------------------------------------------------------------------------
# Coordinate extraction pipeline
# ---------------------------------------------------------------------------


def _device_compact_offsets(d_counts: cp.ndarray) -> cp.ndarray:
    """Build (n+1) offset array from per-element counts via exclusive sum.

    Returns int32 array of shape (n+1,) where offsets[0]=0 and
    offsets[i+1] = offsets[i] + counts[i].
    """
    n = d_counts.shape[0]
    d_offsets = cp.empty(n + 1, dtype=cp.int32)
    d_offsets[0] = 0
    if n > 0:
        d_excl = exclusive_sum(d_counts, synchronize=False)
        d_offsets[1:] = d_excl.astype(cp.int32) + d_counts.astype(cp.int32)
    return d_offsets


def _capacity_prefix(d_counts: cp.ndarray) -> cp.ndarray:
    """Fixed-shape int32 prefix; its terminal stays on device until planning."""
    return _device_compact_offsets(cp.asarray(d_counts, dtype=cp.int32))


def _read_wkt_gpu_capacity(
    d_bytes: cp.ndarray,
    *,
    row_count_hint: int | None,
    on_invalid: Literal["raise", "warn", "ignore"] = "raise",
    input_validity: cp.ndarray | None = None,
) -> OwnedGeometryArray:
    """Parse WKT with one explicit semantic/allocation planning packet.

    The packet is required by the synchronous ``ValueError`` contract: the
    host must know whether device validation succeeded before returning an
    owned geometry.  All data-dependent cardinalities share that one packet;
    fixed-capacity counting and direct scatter avoid implicit CuPy compaction
    fences before and after it.
    """
    from vibespatial.io.pylibcudf import (
        _build_device_mixed_owned,
        _build_device_single_family_owned,
    )

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    n_bytes = int(d_bytes.size)
    if n_bytes == 0:
        return _build_empty_owned()
    if row_count_hint is not None and int(row_count_hint) < 0:
        raise ValueError("row_count_hint must be nonnegative")

    # Without ingress metadata, byte count is a conservative row capacity.
    # The single packet below reports the actual line count together with all
    # semantic and allocation totals. Public constructors pass the exact hint.
    row_capacity = (
        max(int(row_count_hint), 1)
        if row_count_hint is not None
        else max(n_bytes, 1)
    )
    kernels = _wkt_capacity_kernels()
    d_is_geom_start = cp.zeros(n_bytes, dtype=cp.uint8)
    d_is_geom_start[0] = (d_bytes[0] != ord("\n")) & (d_bytes[0] != ord("\r"))
    if n_bytes > 1:
        d_is_geom_start[1:] = (
            (d_bytes[:-1] == ord("\n"))
            & (d_bytes[1:] != ord("\n"))
            & (d_bytes[1:] != ord("\r"))
        )
    d_start_prefix = cp.cumsum(d_is_geom_start, dtype=cp.int32)
    d_geom_starts = cp.full(row_capacity, n_bytes, dtype=cp.int64)
    _launch_kernel(
        runtime,
        kernels["wkt_capacity_scatter_geom_starts"],
        n_bytes,
        (
            (
                ptr(d_is_geom_start),
                ptr(d_start_prefix),
                ptr(d_geom_starts),
                np.int32(row_capacity),
                np.int64(n_bytes),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I64,
            ),
        ),
    )
    d_actual_rows = d_start_prefix[-1].astype(cp.int64)

    d_family_tags = cp.empty(row_capacity, dtype=cp.int8)
    d_empty_flags = cp.empty(row_capacity, dtype=cp.uint8)
    classify = _classify_type_kernels()["wkt_classify_geometry_type"]
    _launch_kernel(
        runtime,
        classify,
        row_capacity,
        (
            (
                ptr(d_bytes),
                ptr(d_geom_starts),
                ptr(d_family_tags),
                ptr(d_empty_flags),
                np.int32(row_capacity),
                np.int64(n_bytes),
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

    d_status = cp.empty(row_capacity, dtype=cp.int8)
    d_pair_counts = cp.empty(row_capacity, dtype=cp.int32)
    d_token_counts = cp.empty(row_capacity, dtype=cp.int32)
    d_polygon_rings = cp.empty(row_capacity, dtype=cp.int32)
    d_multiline_parts = cp.empty(row_capacity, dtype=cp.int32)
    d_multipolygon_parts = cp.empty(row_capacity, dtype=cp.int32)
    d_multipolygon_rings = cp.empty(row_capacity, dtype=cp.int32)
    _launch_kernel(
        runtime,
        kernels["wkt_capacity_count_validate"],
        row_capacity,
        (
            (
                ptr(d_bytes),
                ptr(d_geom_starts),
                ptr(d_family_tags),
                ptr(d_empty_flags),
                ptr(d_status),
                ptr(d_pair_counts),
                ptr(d_token_counts),
                ptr(d_polygon_rings),
                ptr(d_multiline_parts),
                ptr(d_multipolygon_parts),
                ptr(d_multipolygon_rings),
                np.int32(row_capacity),
                np.int64(n_bytes),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I64,
            ),
        ),
    )

    # Materialize every numeric token span at fixed capacity. A valid token
    # occupies at least one byte and adjacent tokens
    # require a separator, so ceil(n_bytes / 2) is a conservative upper bound
    # independent of device-discovered token cardinality.
    d_token_offsets_by_row = _capacity_prefix(d_token_counts)
    token_capacity = max((n_bytes + 1) // 2, 1)
    # Byte positions share the input buffer's 64-bit address domain. Keeping
    # these absolute offsets at int64 preserves tokens beyond INT32_MAX for
    # WKT payloads larger than 2 GiB.
    d_token_starts = cp.zeros(token_capacity, dtype=_WKT_TOKEN_POSITION_DTYPE)
    d_token_ends = cp.zeros(token_capacity, dtype=_WKT_TOKEN_POSITION_DTYPE)
    _launch_kernel(
        runtime,
        kernels["wkt_capacity_scatter_numeric_tokens"],
        row_capacity,
        (
            (
                ptr(d_bytes),
                ptr(d_geom_starts),
                ptr(d_token_offsets_by_row),
                ptr(d_token_starts),
                ptr(d_token_ends),
                np.int32(row_capacity),
                np.int64(n_bytes),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I64,
            ),
        ),
    )
    d_parsed_values = cp.empty(token_capacity, dtype=cp.float64)

    _launch_kernel(
        runtime,
        kernels["wkt_capacity_refine_fp64"],
        token_capacity,
        (
            (
                ptr(d_bytes),
                ptr(d_token_starts),
                ptr(d_token_ends),
                ptr(d_parsed_values),
                np.int32(token_capacity),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
    )

    _launch_kernel(
        runtime,
        kernels["wkt_capacity_validate_ring_closure"],
        row_capacity,
        (
            (
                ptr(d_bytes),
                ptr(d_geom_starts),
                ptr(d_family_tags),
                ptr(d_empty_flags),
                ptr(d_token_offsets_by_row),
                ptr(d_parsed_values),
                ptr(d_status),
                np.int32(row_capacity),
                np.int64(n_bytes),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I64,
            ),
        ),
    )

    d_rows = cp.arange(row_capacity, dtype=cp.int64)
    d_active = d_rows < d_actual_rows
    if input_validity is None:
        d_input_validity = cp.ones(row_capacity, dtype=cp.bool_)
    else:
        if int(input_validity.size) != row_capacity:
            raise ValueError("WKT input_validity must match the row capacity")
        d_input_validity = cp.asarray(input_validity, dtype=cp.bool_)
    row_offsets: dict[int, cp.ndarray] = {}
    coordinate_offsets: dict[int, cp.ndarray] = {}
    first_offsets: dict[int, cp.ndarray | None] = {}
    second_offsets: dict[int, cp.ndarray | None] = {}
    packet_scalars: list[cp.ndarray] = [d_actual_rows]
    for code in (-2, -3, -4, -5, -6, -7):
        packet_scalars.append(
            cp.any(d_active & (d_status == code)).astype(cp.int64, copy=False)
        )

    for tag in _WKT_TAG_TO_FAMILY:
        d_family = (
            d_active
            & d_input_validity
            & (d_status == 0)
            & (d_family_tags == tag)
        )
        d_rows_for_family = d_family.astype(cp.int32, copy=False)
        d_coords_for_family = cp.where(d_family, d_pair_counts, 0).astype(
            cp.int32,
            copy=False,
        )
        row_offsets[tag] = _capacity_prefix(d_rows_for_family)
        coordinate_offsets[tag] = _capacity_prefix(d_coords_for_family)
        if tag == 2:
            first_offsets[tag] = _capacity_prefix(
                cp.where(d_family, d_polygon_rings, 0)
            )
        elif tag == 4:
            first_offsets[tag] = _capacity_prefix(
                cp.where(d_family, d_multiline_parts, 0)
            )
        elif tag == 5:
            first_offsets[tag] = _capacity_prefix(
                cp.where(d_family, d_multipolygon_parts, 0)
            )
            second_offsets[tag] = _capacity_prefix(
                cp.where(d_family, d_multipolygon_rings, 0)
            )
        else:
            first_offsets[tag] = None
        second_offsets.setdefault(tag, None)
        packet_scalars.extend(
            (
                row_offsets[tag][-1].astype(cp.int64),
                coordinate_offsets[tag][-1].astype(cp.int64),
                (
                    cp.asarray(0, dtype=cp.int64)
                    if first_offsets[tag] is None
                    else first_offsets[tag][-1].astype(cp.int64)
                ),
                (
                    cp.asarray(0, dtype=cp.int64)
                    if second_offsets[tag] is None
                    else second_offsets[tag][-1].astype(cp.int64)
                ),
            )
        )

    d_packet = cp.stack(packet_scalars)
    h_packet = np.asarray(
        _wkt_device_to_host(
            d_packet,
            reason="WKT semantic validation and exact allocation packet",
        ),
        dtype=np.int64,
    )
    n_geoms = int(h_packet[0])
    if row_count_hint is not None and n_geoms != int(row_count_hint):
        raise ValueError(
            "WKT row_count_hint does not match newline-delimited device input"
        )
    invalid = h_packet[1:7]
    invalid_message = None
    if invalid[5]:
        invalid_message = "point array must contain 0 or >1 elements"
    elif invalid[1]:
        invalid_message = (
            "2D WKT coordinate stream contains an odd number of values in a geometry"
        )
    elif invalid[2]:
        invalid_message = (
            "WKT geometry has unbalanced parentheses or unexpected trailing structure"
        )
    elif invalid[3]:
        invalid_message = "WKT geometry violates family coordinate cardinality or nesting"
    elif invalid[4]:
        invalid_message = "WKT polygon ring is not closed"
    elif invalid[0]:
        invalid_message = (
            "GPU WKT parsing supports 2D Point, LineString, Polygon, and Multi* families"
        )
    if invalid_message is not None:
        if on_invalid == "raise":
            error_type = (
                _GpuWktOnInvalidError
                if invalid[5]
                else _GpuWktCompatibilityDecline
            )
            raise error_type(invalid_message)
        if on_invalid == "warn":
            warnings.warn(invalid_message, UserWarning, stacklevel=3)

    totals = h_packet[7:].reshape(len(_WKT_TAG_TO_FAMILY), 4)
    d_source_rows = cp.arange(n_geoms, dtype=cp.int64)
    d_validity = (
        (d_status[:n_geoms] == 0) & d_input_validity[:n_geoms]
    ).astype(cp.bool_, copy=False)
    d_tags = cp.where(d_validity, d_family_tags[:n_geoms], np.int8(-1)).astype(
        cp.int8,
        copy=False,
    )
    d_family_row_offsets = cp.full(n_geoms, -1, dtype=cp.int32)
    family_devices: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}
    dummy_i32 = cp.zeros(1, dtype=cp.int32)

    for tag, family in _WKT_TAG_TO_FAMILY.items():
        family_rows, coordinate_count, first_count, second_count = (
            int(value) for value in totals[tag]
        )
        if family_rows == 0:
            continue
        d_row_offsets = row_offsets[tag]
        d_coordinate_offsets = coordinate_offsets[tag]
        d_first_offsets = first_offsets[tag]
        d_second_offsets = second_offsets[tag]
        d_x = cp.empty(coordinate_count, dtype=cp.float64)
        d_y = cp.empty(coordinate_count, dtype=cp.float64)
        d_empty = cp.empty(family_rows, dtype=cp.uint8)
        d_geometry = cp.zeros(family_rows + 1, dtype=cp.int32)
        d_first = cp.zeros(first_count + 1, dtype=cp.int32)
        d_second = cp.zeros(second_count + 1, dtype=cp.int32)
        _launch_kernel(
            runtime,
            kernels["wkt_capacity_scatter_family"],
            n_geoms,
            (
                (
                    ptr(d_bytes),
                    ptr(d_geom_starts),
                    ptr(d_tags),
                    ptr(d_empty_flags),
                    ptr(d_row_offsets),
                    ptr(d_coordinate_offsets),
                    ptr(d_first_offsets if d_first_offsets is not None else dummy_i32),
                    ptr(d_second_offsets if d_second_offsets is not None else dummy_i32),
                    ptr(d_token_offsets_by_row),
                    ptr(d_parsed_values),
                    ptr(d_x),
                    ptr(d_y),
                    ptr(d_empty),
                    ptr(d_geometry),
                    ptr(d_first),
                    ptr(d_second),
                    np.int32(tag),
                    np.int32(n_geoms),
                    np.int64(n_bytes),
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I64,
                ),
            ),
        )
        d_family_row_offsets = cp.where(
            d_tags == tag,
            d_row_offsets[d_source_rows].astype(cp.int32, copy=False),
            d_family_row_offsets,
        )
        if family in {
            GeometryFamily.POINT,
            GeometryFamily.LINESTRING,
            GeometryFamily.MULTIPOINT,
        }:
            d_geometry_offsets = d_geometry
            d_part_offsets = None
            d_ring_offsets = None
        elif family is GeometryFamily.POLYGON:
            d_geometry_offsets = d_geometry
            d_part_offsets = None
            d_ring_offsets = d_first
        elif family is GeometryFamily.MULTILINESTRING:
            d_geometry_offsets = d_geometry
            d_part_offsets = d_first
            d_ring_offsets = None
        else:
            d_geometry_offsets = d_geometry
            d_part_offsets = d_first
            d_ring_offsets = d_second
        family_devices[family] = DeviceFamilyGeometryBuffer(
            family=family,
            x=d_x,
            y=d_y,
            geometry_offsets=cp.asarray(d_geometry_offsets, dtype=cp.int32),
            empty_mask=d_empty.astype(cp.bool_, copy=False),
            part_offsets=d_part_offsets,
            ring_offsets=d_ring_offsets,
        )

    all_valid = invalid_message is None and input_validity is None
    if not family_devices:
        return _build_device_single_family_owned(
            family=GeometryFamily.POINT,
            validity_device=d_validity,
            x_device=cp.empty(0, dtype=cp.float64),
            y_device=cp.empty(0, dtype=cp.float64),
            geometry_offsets_device=cp.zeros(1, dtype=cp.int32),
            empty_mask_device=cp.empty(0, dtype=cp.bool_),
            detail="GPU WKT parse (all rows invalid, capacity-backed)",
            valid_count=0,
        )
    if len(family_devices) == 1:
        family, buffer = next(iter(family_devices.items()))
        return _build_device_single_family_owned(
            family=family,
            validity_device=d_validity,
            x_device=buffer.x,
            y_device=buffer.y,
            geometry_offsets_device=buffer.geometry_offsets,
            empty_mask_device=buffer.empty_mask,
            part_offsets_device=buffer.part_offsets,
            ring_offsets_device=buffer.ring_offsets,
            detail=f"GPU WKT parse ({family.value}, capacity-backed)",
            all_valid=all_valid,
        )

    return _build_device_mixed_owned(
        validity_device=d_validity,
        tags_device=d_tags.astype(cp.int8, copy=False),
        family_row_offsets_device=d_family_row_offsets,
        family_devices=family_devices,
        detail="GPU WKT parse (mixed, capacity-backed)",
        all_valid=all_valid,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_wkt_gpu(
    d_bytes: cp.ndarray,
    *,
    row_count_hint: int | None = None,
    on_invalid: Literal["raise", "warn", "ignore"] = "raise",
    input_validity: cp.ndarray | None = None,
) -> OwnedGeometryArray:
    """Parse WKT bytes on GPU and return device-resident geometry.

    Given a device-resident byte array containing one or more WKT
    geometry strings separated by newlines, this function performs
    full GPU-accelerated parsing: structural analysis, coordinate
    extraction, and assembly into an ``OwnedGeometryArray``.

    Supported geometry types:

    - ``POINT``, ``LINESTRING``, ``POLYGON`` (full support)
    - ``MULTIPOINT``, ``MULTILINESTRING``, ``MULTIPOLYGON``
    - ``EMPTY`` variants of all types

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 array of WKT text bytes, shape ``(n,)``.
        Multiple geometries are separated by newline characters
        (``\\n``, 0x0A).
    row_count_hint : int, optional
        Host-known number of input rows. Public ingress should provide this
        structural metadata so row scratch uses exact capacity. Raw device
        byte callers may omit it; byte count then provides a safe capacity.

    Returns
    -------
    OwnedGeometryArray
        Device-resident geometry array.  Coordinates are always fp64.
        Structural metadata (offsets, validity) is materialized on both
        host and device per the standard ``_build_device_*_owned``
        pattern.

    Raises
    ------
    ValueError
        If the input contains unsupported geometry types or dimensional
        coordinates (Z, M, or ZM), or cannot be parsed.

    Notes
    -----
    Precision (ADR-0002):
        All coordinates are parsed and stored as fp64.  The structural
        analysis and counting kernels are integer-only byte
        classification -- no PrecisionPlan is needed for those stages.

    Tier classification (ADR-0033):
        Uses Tier 1 (custom NVRTC) for geometry-specific scanning and
        exact binary64 refinement, and Tier 2 (CuPy) for fixed-shape
        element-wise operations.

    Examples
    --------
    >>> import cupy as cp
    >>> wkt = b"POINT(1 2)\\nLINESTRING(0 0, 1 1, 2 0)"
    >>> d_bytes = cp.frombuffer(wkt, dtype=cp.uint8)
    >>> owned = read_wkt_gpu(d_bytes)
    >>> owned.row_count
    2
    """
    return _read_wkt_gpu_capacity(
        d_bytes,
        row_count_hint=row_count_hint,
        on_invalid=on_invalid,
        input_validity=input_validity,
    )


def _build_empty_owned() -> OwnedGeometryArray:
    """Build an empty OwnedGeometryArray with zero rows."""
    from vibespatial.io.pylibcudf import _build_device_single_family_owned

    d_validity = cp.empty(0, dtype=cp.bool_)
    d_x = cp.empty(0, dtype=cp.float64)
    d_y = cp.empty(0, dtype=cp.float64)
    d_geom_offsets = cp.zeros(1, dtype=cp.int32)
    d_empty_mask = cp.empty(0, dtype=cp.bool_)

    return _build_device_single_family_owned(
        family=GeometryFamily.POINT,
        validity_device=d_validity,
        x_device=d_x,
        y_device=d_y,
        geometry_offsets_device=d_geom_offsets,
        empty_mask_device=d_empty_mask,
        detail="GPU WKT parse (empty)",
    )
