"""GPU KML reader -- structural analysis, coordinate extraction, and assembly.

GPU-accelerated KML parser.  Given a device-resident byte array
containing a KML document, this module performs:

1. **XML comment masking** -- detect ``<!-- ... -->`` comment boundaries
   and produce a per-byte mask that suppresses tag matches inside comments.
2. **Tag boundary detection** -- use ``gpu_parse.pattern_match()`` to find
   specific KML structural tags (``<coordinates>``, ``<Placemark>``,
   ``<Point>``, ``<LineString>``, ``<Polygon>``, ``<MultiGeometry>``,
   ``<outerBoundaryIs>``, ``<innerBoundaryIs>``).
3. **Coordinate region detection** -- pair ``<coordinates>``/``</coordinates>``
   tags to identify byte ranges containing coordinate content.
4. **Placemark boundary detection** -- pair ``<Placemark>``/``</Placemark>``
   tags to group geometry by feature.
5. **Geometry type detection** -- within each Placemark, detect the geometry
   type tag to classify as Point, LineString, Polygon, or MultiGeometry.
6. **Dimensionality detection** -- count commas vs spaces in coordinate
   regions to determine 2D (lon,lat) or 3D (lon,lat,alt) format.
7. **Coordinate extraction** -- extract numeric values via gpu_parse
   primitives and de-interleave into x (lon) and y (lat) arrays.
8. **OwnedGeometryArray assembly** -- build device-resident geometry with
   proper offset arrays for Point, LineString, and Polygon types.

All operations run on the GPU with zero host materialization until the
caller explicitly requests results.

KML coordinate format: ``lon,lat[,alt] lon,lat[,alt] ...``
- Components within a tuple: COMMA separated
- Tuples: SPACE or NEWLINE separated
- Longitude is FIRST (KML convention, unlike WKT which is also lon-first)

Tier classification (ADR-0033):
    - Comment masking: Tier 1 (custom NVRTC -- XML-specific comment detection)
    - Tag matching: delegates to gpu_parse.pattern_match (Tier 1)
    - Tag pairing + filtering: Tier 2 (CuPy element-wise + flatnonzero)
    - Geometry type assignment: Tier 1 (custom NVRTC -- per-Placemark tag scan)
    - Comma/space counting: Tier 1 (custom NVRTC -- per-region counting)
    - Region-to-Placemark assignment: Tier 1 (custom NVRTC -- binary search)
    - Number extraction: delegates to gpu_parse primitives (Tier 1/2)
    - Offset building: Tier 2 (CuPy) / CCCL exclusive_sum
    - Assembly: follows wkt_gpu.py / geojson_gpu.py patterns

Precision (ADR-0002):
    Structural and counting kernels are integer-only byte classification.
    No floating-point coordinate computation occurs in those kernels, so
    no PrecisionPlan is needed (same rationale as gpu_parse/structural.py).
    Coordinate parsing delegates to gpu_parse.parse_ascii_floats which
    always produces fp64 -- storage precision is always fp64 per ADR-0002.
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
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.io.gpu_parse.numeric import (
    extract_number_positions,
    number_boundaries,
    parse_ascii_floats,
)
from vibespatial.io.gpu_parse.pattern import mark_spans, pattern_match

if TYPE_CHECKING:
    import cupy as cp

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

# ctypes for int64 kernel params (files > 2 GB)
KERNEL_PARAM_I64 = ctypes.c_longlong


# ---------------------------------------------------------------------------
# Geometry family constants (match GeometryFamily enum order)
# ---------------------------------------------------------------------------
KML_FAMILY_POINT: int = 0
KML_FAMILY_LINESTRING: int = 1
KML_FAMILY_POLYGON: int = 2
KML_FAMILY_MULTI: int = 6  # MultiGeometry -- not a standard GeometryFamily value
KML_FAMILY_UNKNOWN: int = -2


# ---------------------------------------------------------------------------
# Kernel sources (Tier 1 NVRTC) -- integer-only byte classification,
# no floating-point computation, so no PrecisionPlan needed.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Kernel: XML comment masking
#
# Detect <!-- ... --> comment boundaries and produce a per-byte mask
# where bytes inside comments are set to 1.  This mask is used to
# suppress tag matches that fall inside XML comments.
#
# Two-pass approach:
#   Pass 1 (this kernel): each thread checks if its byte position starts
#   the 4-byte sequence "<!--".  Writes 1 to d_comment_starts for matches.
#   A second pattern_match finds "-->".
#   Pass 2: a pairing kernel (or CuPy sort + pair) matches each start
#   to its nearest end, then mark_spans fills the mask.
#
# We reuse pattern_match for both "<!--" and "-->" detection, then
# pair them with CuPy operations.  No custom kernel needed for this.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Kernel: assign geometry family tag per Placemark
#
# For each Placemark, scan the bytes between its start and end to find
# the first geometry type tag (<Point>, <LineString>, <Polygon>,
# <MultiGeometry>).  Also handles namespace prefixes (e.g., <kml:Point>).
#
# The scan is case-sensitive (XML tag names are case-sensitive in KML).
# ---------------------------------------------------------------------------

_KML_ASSIGN_GEOM_TYPE_SOURCE = r"""
// Per-Placemark geometry type classification kernel.
//
// Each thread handles one Placemark.  It scans the byte range
// [placemark_start, placemark_end) looking for the first geometry
// type opening tag.  Tag names are case-sensitive in XML/KML.
//
// Handles namespace prefixes: after '<', if we see alphabetic chars
// followed by ':', we skip the prefix and match the local name.
//
// Family tags:
//   0 = Point
//   1 = LineString
//   2 = Polygon
//   6 = MultiGeometry
//  -2 = unknown/none

extern "C" __global__ void __launch_bounds__(256, 4)
kml_assign_geometry_type(
    const unsigned char* __restrict__ input,
    const unsigned char* __restrict__ comment_mask,
    const long long* __restrict__ placemark_starts,
    const long long* __restrict__ placemark_ends,
    signed char* __restrict__ family_tags,
    const int n_placemarks,
    const long long n_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_placemarks) return;

    long long start = placemark_starts[idx];
    long long end = placemark_ends[idx];
    if (end > n_bytes) end = n_bytes;

    family_tags[idx] = -2;  // default: unknown

    // Scan for '<' characters within this Placemark
    for (long long pos = start; pos < end - 1; pos++) {
        if (input[pos] != '<') continue;

        // Skip bytes inside XML comments
        if (comment_mask[pos]) continue;

        // Skip closing tags
        if (input[pos + 1] == '/') continue;

        // Skip XML declarations and processing instructions
        if (input[pos + 1] == '?' || input[pos + 1] == '!') continue;

        // Find the tag name start (skip '<')
        long long name_start = pos + 1;

        // Handle namespace prefix: skip "prefix:" if present
        // Scan for ':' before any whitespace or '>' or '/'
        long long scan = name_start;
        long long colon_pos = -1;
        while (scan < end && scan < name_start + 64) {
            unsigned char sc = input[scan];
            if (sc == ':') { colon_pos = scan; break; }
            if (sc == '>' || sc == '/' || sc == ' ' || sc == '\t'
                || sc == '\n' || sc == '\r') break;
            scan++;
        }
        if (colon_pos > 0) {
            name_start = colon_pos + 1;
        }

        // Now match the local tag name (case-sensitive for XML)
        // We need at least a few bytes to match
        if (name_start >= end) continue;

        // Match "Point"
        if (name_start + 5 <= end
            && input[name_start]     == 'P'
            && input[name_start + 1] == 'o'
            && input[name_start + 2] == 'i'
            && input[name_start + 3] == 'n'
            && input[name_start + 4] == 't') {
            // Verify next char is '>' or whitespace or '/' (not part of longer name)
            if (name_start + 5 >= end) { family_tags[idx] = 0; return; }
            unsigned char after = input[name_start + 5];
            if (after == '>' || after == ' ' || after == '\t'
                || after == '\n' || after == '\r' || after == '/') {
                family_tags[idx] = 0;  // Point
                return;
            }
        }

        // Match "LineString"
        if (name_start + 10 <= end
            && input[name_start]     == 'L'
            && input[name_start + 1] == 'i'
            && input[name_start + 2] == 'n'
            && input[name_start + 3] == 'e'
            && input[name_start + 4] == 'S'
            && input[name_start + 5] == 't'
            && input[name_start + 6] == 'r'
            && input[name_start + 7] == 'i'
            && input[name_start + 8] == 'n'
            && input[name_start + 9] == 'g') {
            if (name_start + 10 >= end) { family_tags[idx] = 1; return; }
            unsigned char after = input[name_start + 10];
            if (after == '>' || after == ' ' || after == '\t'
                || after == '\n' || after == '\r' || after == '/') {
                family_tags[idx] = 1;  // LineString
                return;
            }
        }

        // Match "Polygon"
        if (name_start + 7 <= end
            && input[name_start]     == 'P'
            && input[name_start + 1] == 'o'
            && input[name_start + 2] == 'l'
            && input[name_start + 3] == 'y'
            && input[name_start + 4] == 'g'
            && input[name_start + 5] == 'o'
            && input[name_start + 6] == 'n') {
            if (name_start + 7 >= end) { family_tags[idx] = 2; return; }
            unsigned char after = input[name_start + 7];
            if (after == '>' || after == ' ' || after == '\t'
                || after == '\n' || after == '\r' || after == '/') {
                family_tags[idx] = 2;  // Polygon
                return;
            }
        }

        // Match "MultiGeometry"
        if (name_start + 13 <= end
            && input[name_start]      == 'M'
            && input[name_start + 1]  == 'u'
            && input[name_start + 2]  == 'l'
            && input[name_start + 3]  == 't'
            && input[name_start + 4]  == 'i'
            && input[name_start + 5]  == 'G'
            && input[name_start + 6]  == 'e'
            && input[name_start + 7]  == 'o'
            && input[name_start + 8]  == 'm'
            && input[name_start + 9]  == 'e'
            && input[name_start + 10] == 't'
            && input[name_start + 11] == 'r'
            && input[name_start + 12] == 'y') {
            if (name_start + 13 >= end) { family_tags[idx] = 6; return; }
            unsigned char after = input[name_start + 13];
            if (after == '>' || after == ' ' || after == '\t'
                || after == '\n' || after == '\r' || after == '/') {
                family_tags[idx] = 6;  // MultiGeometry
                return;
            }
        }
    }
}
"""

_KML_ASSIGN_GEOM_TYPE_NAMES: tuple[str, ...] = ("kml_assign_geometry_type",)


# ---------------------------------------------------------------------------
# NVRTC warmup registration (ADR-0034 Level 2)
# ---------------------------------------------------------------------------
# Register kernels for background precompilation.
# pattern_match kernels for specific KML tags are compiled on-demand
# by gpu_parse/pattern.py and cached by pattern bytes.

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    (
        "kml-assign-geom-type",
        _KML_ASSIGN_GEOM_TYPE_SOURCE,
        _KML_ASSIGN_GEOM_TYPE_NAMES,
    ),
])


# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------

def _assign_geom_type_kernels() -> dict:
    """Compile (or retrieve cached) KML geometry type assignment kernel."""
    return compile_kernel_group(
        "kml-assign-geom-type",
        _KML_ASSIGN_GEOM_TYPE_SOURCE,
        _KML_ASSIGN_GEOM_TYPE_NAMES,
    )


def _launch_kernel(runtime, kernel, n, params):
    """Launch a kernel with occupancy-based grid/block sizing."""
    grid, block = runtime.launch_config(kernel, int(n))
    runtime.launch(kernel, grid=grid, block=block, params=params)


# ---------------------------------------------------------------------------
# Tag pattern definitions
#
# Each tag we care about has both a plain and a namespace-prefixed variant.
# For example, <coordinates> can appear as <kml:coordinates>.
#
# For closing tags, the namespace prefix also applies:
# </coordinates> or </kml:coordinates>.
#
# Self-closing tags (<tag/>) are handled by the geometry type assignment
# kernel -- a self-closing geometry tag like <Point/> would be a Placemark
# with no coordinates, which is valid (empty geometry).
# ---------------------------------------------------------------------------

# Coordinate region tags
_TAG_COORD_OPEN = b"<coordinates>"
_TAG_COORD_OPEN_NS = b"<kml:coordinates>"
_TAG_COORD_CLOSE = b"</coordinates>"
_TAG_COORD_CLOSE_NS = b"</kml:coordinates>"

# Placemark boundary tags
_TAG_PLACEMARK_OPEN = b"<Placemark>"
_TAG_PLACEMARK_OPEN_NS = b"<kml:Placemark>"
_TAG_PLACEMARK_OPEN_ATTR = b"<Placemark "  # with attributes
_TAG_PLACEMARK_OPEN_NS_ATTR = b"<kml:Placemark "
_TAG_PLACEMARK_CLOSE = b"</Placemark>"
_TAG_PLACEMARK_CLOSE_NS = b"</kml:Placemark>"

# XML comment markers
_TAG_COMMENT_OPEN = b"<!--"
_TAG_COMMENT_CLOSE = b"-->"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_tag_positions(
    d_bytes: cp.ndarray,
    patterns: list[bytes],
    d_comment_mask: cp.ndarray | None = None,
) -> cp.ndarray:
    """Find all positions matching any of the given tag patterns.

    Runs ``pattern_match`` for each pattern, combines hits, and returns
    sorted int64 positions.  Optionally suppresses matches inside XML
    comments using the comment mask.

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 byte array.
    patterns : list[bytes]
        One or more byte patterns to search for.
    d_comment_mask : cp.ndarray or None
        If provided, a uint8 mask where 1 = inside comment.  Matches at
        positions where the mask is 1 are suppressed.

    Returns
    -------
    cp.ndarray
        Sorted int64 array of match positions (byte offsets).
    """
    import cupy as cp

    combined_hits = None
    for pat in patterns:
        d_hits = pattern_match(d_bytes, pat, d_quote_parity=None)
        if combined_hits is None:
            combined_hits = d_hits
        else:
            combined_hits = combined_hits | d_hits

    if combined_hits is None:
        return cp.empty(0, dtype=cp.int64)

    # Suppress matches inside XML comments (in-place, no temporary for ~mask)
    if d_comment_mask is not None:
        combined_hits[d_comment_mask.view(cp.bool_)] = 0

    d_positions = cp.flatnonzero(combined_hits).astype(cp.int64)
    del combined_hits
    return d_positions


def _build_comment_mask(d_bytes: cp.ndarray) -> cp.ndarray:
    """Build a per-byte mask for XML comment regions.

    Detects ``<!--`` and ``-->`` boundaries, pairs them, and marks all
    bytes within comment regions as 1.

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 byte array, shape ``(n,)``.

    Returns
    -------
    cp.ndarray
        Device-resident uint8 array, shape ``(n,)``.  1 for bytes inside
        an XML comment, 0 otherwise.
    """
    import cupy as cp

    n = d_bytes.shape[0]

    # Find comment open and close positions
    d_open_hits = pattern_match(d_bytes, _TAG_COMMENT_OPEN, d_quote_parity=None)
    d_close_hits = pattern_match(d_bytes, _TAG_COMMENT_CLOSE, d_quote_parity=None)

    d_open_pos = cp.flatnonzero(d_open_hits).astype(cp.int64)
    d_close_pos = cp.flatnonzero(d_close_hits).astype(cp.int64)
    del d_open_hits, d_close_hits

    n_opens = d_open_pos.shape[0]
    n_closes = d_close_pos.shape[0]

    if n_opens == 0 or n_closes == 0:
        return cp.zeros(n, dtype=cp.uint8)

    # Pair each open with the nearest subsequent close.
    # Comments cannot nest in XML, so we do a simple greedy pairing:
    # walk opens and closes in order.  For each open, find the first
    # close that comes after it.
    #
    # Since both arrays are sorted, we use a device-side merge approach
    # via CuPy searchsorted: for each open position, find the first
    # close position that is >= open_pos + 4 (length of "<!--").
    d_close_idx = cp.searchsorted(d_close_pos, d_open_pos + 4, side="left")

    # Filter out opens that have no matching close
    d_valid = d_close_idx < n_closes
    d_valid_open_pos = d_open_pos[d_valid]
    d_valid_close_idx = d_close_idx[d_valid]
    del d_open_pos, d_close_idx, d_valid

    if d_valid_open_pos.shape[0] == 0:
        return cp.zeros(n, dtype=cp.uint8)

    # Get the close positions for valid pairs
    d_paired_close_pos = d_close_pos[d_valid_close_idx]
    del d_close_pos, d_valid_close_idx

    # Comment spans: [open_pos, close_pos + 3)  (3 = len("-->"))
    d_span_starts = d_valid_open_pos
    d_span_ends = d_paired_close_pos + 3  # inclusive of "-->"
    del d_valid_open_pos, d_paired_close_pos

    # Clamp ends to n
    d_span_ends = cp.minimum(d_span_ends, n)

    # Use mark_spans from gpu_parse to fill the mask
    from vibespatial.io.gpu_parse.pattern import mark_spans

    d_mask = mark_spans(d_span_starts, d_span_ends, n)
    del d_span_starts, d_span_ends
    return d_mask


def _pair_open_close_tags(
    d_open_positions: cp.ndarray,
    d_close_positions: cp.ndarray,
    close_tag_len: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Pair opening and closing tag positions.

    Uses searchsorted to find the nearest closing tag after each opening
    tag.  Returns paired start/end arrays where the start is the byte
    after the opening tag's ``>`` and the end is the byte of the closing
    tag's ``<``.

    Parameters
    ----------
    d_open_positions : cp.ndarray
        Sorted int64 positions of opening tag starts (byte of ``<``).
    d_close_positions : cp.ndarray
        Sorted int64 positions of closing tag starts (byte of ``<``).
    close_tag_len : int
        Length of the closing tag pattern (e.g. 14 for ``</coordinates>``).

    Returns
    -------
    tuple[cp.ndarray, cp.ndarray]
        ``(d_starts, d_ends)`` where ``d_starts`` are the opening tag
        positions and ``d_ends`` are the closing tag positions.  Only
        valid pairs are returned (opens with a matching close).
    """
    import cupy as cp

    n_opens = d_open_positions.shape[0]
    n_closes = d_close_positions.shape[0]

    if n_opens == 0 or n_closes == 0:
        return cp.empty(0, dtype=cp.int64), cp.empty(0, dtype=cp.int64)

    # For each open, find the first close that is >= open position
    # (close must come after the open)
    d_close_idx = cp.searchsorted(d_close_positions, d_open_positions, side="left")

    # Filter out opens that have no matching close
    d_valid = d_close_idx < n_closes
    d_valid_opens = d_open_positions[d_valid]
    d_valid_close_idx = d_close_idx[d_valid]
    del d_close_idx, d_valid

    if d_valid_opens.shape[0] == 0:
        return cp.empty(0, dtype=cp.int64), cp.empty(0, dtype=cp.int64)

    d_valid_closes = d_close_positions[d_valid_close_idx]
    del d_valid_close_idx

    return d_valid_opens, d_valid_closes


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class KmlStructuralResult:
    """Result of KML structural analysis.

    All arrays are device-resident CuPy arrays except ``n_placemarks``
    which is a Python int.

    Attributes
    ----------
    d_coord_starts : cp.ndarray
        Byte offset of the first content byte after each ``<coordinates>``
        tag, int64, shape ``(n_coord_regions,)``.  This is the position
        immediately after the ``>`` of the opening tag.
    d_coord_ends : cp.ndarray
        Byte offset of the ``<`` of each ``</coordinates>`` tag, int64,
        shape ``(n_coord_regions,)``.  Content bytes are in the half-open
        range ``[d_coord_starts[i], d_coord_ends[i])``.
    d_placemark_starts : cp.ndarray
        Byte offset of each ``<Placemark>`` tag, int64, shape
        ``(n_placemarks,)``.
    d_placemark_ends : cp.ndarray
        Byte offset one past the ``>`` of each ``</Placemark>`` tag,
        int64, shape ``(n_placemarks,)``.
    d_family_tags : cp.ndarray
        Geometry family tag per Placemark, int8, shape ``(n_placemarks,)``.
        Values: 0=Point, 1=LineString, 2=Polygon, 6=MultiGeometry,
        -2=unknown/none.
    n_placemarks : int
        Number of Placemarks detected.
    """

    d_coord_starts: cp.ndarray
    d_coord_ends: cp.ndarray
    d_placemark_starts: cp.ndarray
    d_placemark_ends: cp.ndarray
    d_family_tags: cp.ndarray
    n_placemarks: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def kml_structural_analysis(d_bytes: cp.ndarray) -> KmlStructuralResult:
    """Perform structural analysis on a KML document.

    Given a device-resident byte array containing a KML document, detects:

    1. XML comment regions (suppressed from all subsequent matching).
    2. ``<coordinates>`` / ``</coordinates>`` region boundaries.
    3. ``<Placemark>`` / ``</Placemark>`` feature boundaries.
    4. Geometry type per Placemark (Point, LineString, Polygon, MultiGeometry).

    Handles KML namespace prefixes: tags like ``<kml:coordinates>`` and
    ``<kml:Placemark>`` are matched alongside their unprefixed variants.
    Also handles opening tags with attributes (e.g., ``<Placemark id="1">``).

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 array of KML file bytes, shape ``(n,)``.

    Returns
    -------
    KmlStructuralResult
        Dataclass containing all structural analysis outputs on device.

    Notes
    -----
    This function uses ``gpu_parse.pattern_match()`` for tag detection
    rather than building a full XML parser.  This is sufficient because
    KML has a fixed, well-known tag vocabulary and we only need to locate
    a small set of specific tags.

    XML comments (``<!-- ... -->``) are detected and masked so that tags
    inside comments are not matched.  CDATA sections are not specifically
    handled since coordinate content in KML does not use CDATA.

    The coordinate region boundaries identify the raw text between
    ``<coordinates>`` and ``</coordinates>`` tags.  The actual coordinate
    parsing (splitting ``lon,lat[,alt]`` tuples) is handled downstream
    by a separate coordinate extraction step.

    Examples
    --------
    >>> import cupy as cp
    >>> kml = b'''<kml><Document>
    ...   <Placemark><Point><coordinates>-122.08,37.42,0</coordinates></Point></Placemark>
    ...   <Placemark><LineString><coordinates>-122.08,37.42 -122.09,37.43</coordinates></LineString></Placemark>
    ... </Document></kml>'''
    >>> d_bytes = cp.frombuffer(kml, dtype=cp.uint8)
    >>> result = kml_structural_analysis(d_bytes)
    >>> result.n_placemarks
    2
    """
    import cupy as cp

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    n = d_bytes.shape[0]

    if n == 0:
        return KmlStructuralResult(
            d_coord_starts=cp.empty(0, dtype=cp.int64),
            d_coord_ends=cp.empty(0, dtype=cp.int64),
            d_placemark_starts=cp.empty(0, dtype=cp.int64),
            d_placemark_ends=cp.empty(0, dtype=cp.int64),
            d_family_tags=cp.empty(0, dtype=cp.int8),
            n_placemarks=0,
        )

    # ------------------------------------------------------------------
    # Stage 1: Build XML comment mask
    # ------------------------------------------------------------------
    # Detect <!-- ... --> regions so tags inside comments are suppressed.
    d_comment_mask = _build_comment_mask(d_bytes)

    # ------------------------------------------------------------------
    # Stage 2: Find coordinate region boundaries
    # ------------------------------------------------------------------
    # Match both plain and namespace-prefixed variants.
    d_coord_open_pos = _find_tag_positions(
        d_bytes,
        [_TAG_COORD_OPEN, _TAG_COORD_OPEN_NS],
        d_comment_mask=d_comment_mask,
    )
    d_coord_close_pos = _find_tag_positions(
        d_bytes,
        [_TAG_COORD_CLOSE, _TAG_COORD_CLOSE_NS],
        d_comment_mask=d_comment_mask,
    )

    # Pair opens with closes.
    # For coordinate content boundaries, we want:
    #   start = byte after the '>' of the opening tag
    #   end = byte of the '<' of the closing tag
    #
    # The opening tag match position points to the '<' of <coordinates>.
    # We need to find the '>' that ends the opening tag to get the
    # content start.  Since <coordinates> has no attributes in KML,
    # the '>' is at a fixed offset: len("<coordinates>") - 1 from the '<'.
    # But with namespace prefix, it could be len("<kml:coordinates>") - 1.
    #
    # Strategy: for each open position, compute the content start as
    # open_pos + tag_length.  Since we matched exact patterns, we know
    # the tag length.  But we combined plain and NS matches.  We need to
    # determine which pattern matched at each position.
    #
    # Simpler: scan forward from each open position to find '>'.
    # Since coordinate tags have no attributes, '>' immediately follows
    # the tag name.  We can use a fixed offset based on which pattern
    # matched.
    #
    # Even simpler: use _pair_open_close_tags with the raw positions,
    # then compute content starts by finding '>' after each open.
    # For the standard <coordinates>, '>' is at open_pos + 12.
    # For <kml:coordinates>, '>' is at open_pos + 16.
    # We handle this by checking the byte at the expected plain offset.

    d_paired_opens, d_paired_closes = _pair_open_close_tags(
        d_coord_open_pos, d_coord_close_pos, close_tag_len=14,
    )
    del d_coord_open_pos, d_coord_close_pos

    # Compute content start: find '>' after the opening tag '<'.
    # For <coordinates>, the '>' is at offset 12 (len("<coordinates>") - 1).
    # For <kml:coordinates>, it is at offset 16 (len("<kml:coordinates>") - 1).
    # Check the byte: if d_bytes[open_pos + 1] == 'k' then it is the NS variant.
    if d_paired_opens.shape[0] > 0:
        # Check second byte to determine tag variant
        d_second_byte = d_bytes[(d_paired_opens + 1).astype(cp.intp)]
        d_is_ns = (d_second_byte == ord('k'))
        # Plain tag: content starts at open_pos + len("<coordinates>") = open_pos + 13
        # NS tag: content starts at open_pos + len("<kml:coordinates>") = open_pos + 17
        d_coord_starts = d_paired_opens + cp.where(d_is_ns, 17, 13)
        del d_second_byte, d_is_ns
    else:
        d_coord_starts = cp.empty(0, dtype=cp.int64)

    # Content end is at the close tag position (byte of '<' in </coordinates>)
    d_coord_ends = d_paired_closes
    del d_paired_opens, d_paired_closes

    # ------------------------------------------------------------------
    # Stage 3: Find Placemark boundaries
    # ------------------------------------------------------------------
    # Placemarks can have attributes: <Placemark id="foo">
    # So we match both <Placemark> and <Placemark  (with trailing space).
    # Also handle namespace prefix variants.
    d_pm_open_pos = _find_tag_positions(
        d_bytes,
        [
            _TAG_PLACEMARK_OPEN,
            _TAG_PLACEMARK_OPEN_NS,
            _TAG_PLACEMARK_OPEN_ATTR,
            _TAG_PLACEMARK_OPEN_NS_ATTR,
        ],
        d_comment_mask=d_comment_mask,
    )
    d_pm_close_pos = _find_tag_positions(
        d_bytes,
        [_TAG_PLACEMARK_CLOSE, _TAG_PLACEMARK_CLOSE_NS],
        d_comment_mask=d_comment_mask,
    )

    d_placemark_starts, d_placemark_closes = _pair_open_close_tags(
        d_pm_open_pos, d_pm_close_pos, close_tag_len=13,
    )
    del d_pm_open_pos, d_pm_close_pos

    # Placemark end = close_pos + len("</Placemark>") or len("</kml:Placemark>")
    # To compute the exact end, check the close tag variant.
    if d_placemark_closes.shape[0] > 0:
        d_close_second = d_bytes[(d_placemark_closes + 2).astype(cp.intp)]
        d_close_is_ns = (d_close_second == ord('k'))
        # </Placemark> = 13 bytes, </kml:Placemark> = 17 bytes
        d_placemark_ends = d_placemark_closes + cp.where(d_close_is_ns, 17, 13)
        del d_close_second, d_close_is_ns
    else:
        d_placemark_ends = cp.empty(0, dtype=cp.int64)

    n_placemarks = int(d_placemark_starts.shape[0])

    # ------------------------------------------------------------------
    # Stage 4: Assign geometry type per Placemark
    # ------------------------------------------------------------------
    if n_placemarks == 0:
        d_family_tags = cp.empty(0, dtype=cp.int8)
    else:
        kernels = _assign_geom_type_kernels()
        d_family_tags = cp.empty(n_placemarks, dtype=cp.int8)

        _launch_kernel(
            runtime,
            kernels["kml_assign_geometry_type"],
            n_placemarks,
            (
                (
                    ptr(d_bytes),
                    ptr(d_comment_mask),
                    ptr(d_placemark_starts),
                    ptr(d_placemark_ends),
                    ptr(d_family_tags),
                    np.int32(n_placemarks),
                    np.int64(n),
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

    del d_comment_mask

    # No sync needed before returning -- all outputs are device arrays
    # and the caller will sync when materializing to host.

    return KmlStructuralResult(
        d_coord_starts=d_coord_starts,
        d_coord_ends=d_coord_ends,
        d_placemark_starts=d_placemark_starts,
        d_placemark_ends=d_placemark_ends,
        d_family_tags=d_family_tags,
        n_placemarks=n_placemarks,
    )


# ---------------------------------------------------------------------------
# NVRTC kernel: count commas and spaces per coordinate region
#
# Used to detect dimensionality (2D vs 3D).  In a coordinate region,
# the number of commas relative to the number of whitespace-separated
# tuples indicates whether each tuple has 2 or 3 components:
#   - "lon,lat lon,lat" => 1 comma per tuple => 2D
#   - "lon,lat,alt lon,lat,alt" => 2 commas per tuple => 3D
#
# We count total commas and total spaces (tuple separators) in each
# region to compute: dim = (commas / max(spaces+1, 1)) + 1
# If dim > 2 => 3D, else 2D.
# ---------------------------------------------------------------------------

_KML_COUNT_COMMAS_SPACES_SOURCE = r"""
// Count commas and spaces per coordinate region.
//
// Each thread handles one coordinate region.  It scans the byte range
// [coord_starts[i], coord_ends[i]) and counts:
//   - commas (0x2C) => commas_out[i]
//   - tuple separators (space/newline/tab/cr) => spaces_out[i]
//     (consecutive separators are collapsed to one count)
//
// These counts let the host infer dimensionality without materializing
// the full coordinate text.

extern "C" __global__ void __launch_bounds__(256, 4)
kml_count_commas_spaces(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ coord_starts,
    const long long* __restrict__ coord_ends,
    int* __restrict__ commas_out,
    int* __restrict__ spaces_out,
    const int n_regions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_regions) return;

    long long start = coord_starts[idx];
    long long end = coord_ends[idx];

    int commas = 0;
    int sep_count = 0;
    int in_separator = 1;  // treat start as after a separator

    for (long long pos = start; pos < end; pos++) {
        unsigned char c = input[pos];
        if (c == ',') {
            commas++;
            in_separator = 0;
        } else if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
            if (!in_separator) {
                sep_count++;
                in_separator = 1;
            }
        } else {
            in_separator = 0;
        }
    }

    commas_out[idx] = commas;
    spaces_out[idx] = sep_count;
}
"""

_KML_COUNT_COMMAS_SPACES_NAMES: tuple[str, ...] = ("kml_count_commas_spaces",)

# Register for warmup (ADR-0034 Level 2)
request_nvrtc_warmup([
    (
        "kml-count-commas-spaces",
        _KML_COUNT_COMMAS_SPACES_SOURCE,
        _KML_COUNT_COMMAS_SPACES_NAMES,
    ),
])


def _count_commas_spaces_kernels() -> dict:
    """Compile (or retrieve cached) KML comma/space counter kernel."""
    return compile_kernel_group(
        "kml-count-commas-spaces",
        _KML_COUNT_COMMAS_SPACES_SOURCE,
        _KML_COUNT_COMMAS_SPACES_NAMES,
    )


# ---------------------------------------------------------------------------
# NVRTC kernel: assign coordinate regions to Placemarks
#
# For each Placemark, find which coordinate regions fall within its
# byte range.  This maps coordinate regions to Placemarks so we know
# how many coordinate values belong to each geometry.
#
# For Polygon Placemarks, each <coordinates> block within the Placemark
# corresponds to one ring (outer boundary + optional inner boundaries).
# ---------------------------------------------------------------------------

_KML_ASSIGN_COORD_REGIONS_SOURCE = r"""
// Assign coordinate regions to Placemarks.
//
// Each thread handles one coordinate region.  It binary-searches the
// Placemark boundaries to find which Placemark contains this region.
//
// Output: placemark_idx[i] = index of the Placemark containing
//         coordinate region i, or -1 if none.

extern "C" __global__ void __launch_bounds__(256, 4)
kml_assign_coord_regions(
    const long long* __restrict__ coord_starts,
    const long long* __restrict__ placemark_starts,
    const long long* __restrict__ placemark_ends,
    int* __restrict__ placemark_idx,
    const int n_coord_regions,
    const int n_placemarks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_coord_regions) return;

    long long cs = coord_starts[idx];

    // Binary search: find the last Placemark whose start <= cs
    int lo = 0;
    int hi = n_placemarks - 1;
    int best = -1;

    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (placemark_starts[mid] <= cs) {
            best = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    // Verify the coordinate region is within the Placemark's range
    if (best >= 0 && cs < placemark_ends[best]) {
        placemark_idx[idx] = best;
    } else {
        placemark_idx[idx] = -1;
    }
}
"""

_KML_ASSIGN_COORD_REGIONS_NAMES: tuple[str, ...] = ("kml_assign_coord_regions",)

request_nvrtc_warmup([
    (
        "kml-assign-coord-regions",
        _KML_ASSIGN_COORD_REGIONS_SOURCE,
        _KML_ASSIGN_COORD_REGIONS_NAMES,
    ),
])


def _assign_coord_regions_kernels() -> dict:
    """Compile (or retrieve cached) KML coordinate region assignment kernel."""
    return compile_kernel_group(
        "kml-assign-coord-regions",
        _KML_ASSIGN_COORD_REGIONS_SOURCE,
        _KML_ASSIGN_COORD_REGIONS_NAMES,
    )


# ---------------------------------------------------------------------------
# KML family tag to GeometryFamily mapping
# ---------------------------------------------------------------------------

_KML_TAG_TO_FAMILY = {
    KML_FAMILY_POINT: GeometryFamily.POINT,
    KML_FAMILY_LINESTRING: GeometryFamily.LINESTRING,
    KML_FAMILY_POLYGON: GeometryFamily.POLYGON,
}


# ---------------------------------------------------------------------------
# Coordinate extraction pipeline
# ---------------------------------------------------------------------------

def _detect_dimensionality(
    d_bytes: cp.ndarray,
    d_coord_starts: cp.ndarray,
    d_coord_ends: cp.ndarray,
) -> int:
    """Detect whether KML coordinates are 2D or 3D.

    Counts commas and tuple separators across all coordinate regions.
    If the ratio of commas to tuples is >= 2 (i.e., 2 commas per tuple),
    the data is 3D (lon,lat,alt).  Otherwise 2D (lon,lat).

    Returns 2 or 3.
    """
    n_regions = d_coord_starts.shape[0]
    if n_regions == 0:
        return 2

    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    kernels = _count_commas_spaces_kernels()

    d_commas = cp.empty(n_regions, dtype=cp.int32)
    d_spaces = cp.empty(n_regions, dtype=cp.int32)

    _launch_kernel(
        runtime,
        kernels["kml_count_commas_spaces"],
        n_regions,
        (
            (
                ptr(d_bytes),
                ptr(d_coord_starts),
                ptr(d_coord_ends),
                ptr(d_commas),
                ptr(d_spaces),
                np.int32(n_regions),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
    )

    # Compute total commas and total tuple separators (spaces+1 per region
    # gives tuple count).  A single sync to read the sums.
    total_commas = int(d_commas.sum())
    total_spaces = int(d_spaces.sum())

    # Number of tuples = spaces + n_regions (each region has at least one tuple)
    n_tuples = total_spaces + n_regions
    if n_tuples == 0:
        return 2

    # Commas per tuple: 1 => 2D (lon,lat), 2 => 3D (lon,lat,alt)
    commas_per_tuple = total_commas / n_tuples
    return 3 if commas_per_tuple >= 1.5 else 2


def _extract_kml_coordinates(
    d_bytes: cp.ndarray,
    structural: KmlStructuralResult,
) -> tuple[cp.ndarray, cp.ndarray, int, cp.ndarray]:
    """Extract all coordinates from KML coordinate regions.

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 byte array.
    structural : KmlStructuralResult
        Result of ``kml_structural_analysis()``.

    Returns
    -------
    d_x : cp.ndarray
        Device-resident float64 x (longitude) coordinates.
    d_y : cp.ndarray
        Device-resident float64 y (latitude) coordinates.
    dim : int
        Dimensionality (2 or 3).
    d_placemark_coord_counts : cp.ndarray
        Number of coordinate pairs per Placemark, int32.
    """
    n = d_bytes.shape[0]
    n_coord_regions = structural.d_coord_starts.shape[0]

    if n_coord_regions == 0:
        return (
            cp.empty(0, dtype=cp.float64),
            cp.empty(0, dtype=cp.float64),
            2,
            cp.zeros(structural.n_placemarks, dtype=cp.int32),
        )

    # Step 1: Detect dimensionality (2D or 3D)
    dim = _detect_dimensionality(
        d_bytes, structural.d_coord_starts, structural.d_coord_ends,
    )

    # Step 2: Build a span mask for all coordinate regions.
    # This marks bytes that are inside <coordinates>...</coordinates> content.
    d_coord_mask = mark_spans(
        structural.d_coord_starts, structural.d_coord_ends, n,
    )

    # Step 3: Find number boundaries using gpu_parse.
    # KML coordinate text has no quote context, so provide a zero parity mask.
    d_zero_parity = cp.zeros(n, dtype=cp.uint8)
    d_is_start, d_is_end = number_boundaries(d_bytes, d_zero_parity)
    del d_zero_parity

    # Step 3b: Fix boundary detection at coordinate region edges.
    #
    # The number_boundaries kernel (designed for JSON) expects numbers
    # to be preceded/followed by JSON separator chars ([, ], comma, space).
    # In KML, coordinate content is directly adjacent to XML tags:
    #   <coordinates>-122.08,37.42</coordinates>
    #                ^            ^
    #   preceded by '>'           followed by '<'
    #
    # These chars are not in the JSON separator list, so the first number
    # start and last number end within each region may be missed.
    # Fix: force start=1 at each region start if the byte is numeric-initial,
    # and force end=1 at each region end-1 if the byte is a numeric char.
    d_coord_starts_arr = structural.d_coord_starts
    d_coord_ends_arr = structural.d_coord_ends

    # Fix first-byte starts: at each coord_starts[i], if the byte is a
    # numeric-initial char, force is_start=1
    d_first_bytes = d_bytes[d_coord_starts_arr.astype(cp.intp)]
    d_is_numeric_initial = (
        ((d_first_bytes >= ord('0')) & (d_first_bytes <= ord('9')))
        | (d_first_bytes == ord('-'))
        | (d_first_bytes == ord('+'))
    )
    d_is_start[d_coord_starts_arr[d_is_numeric_initial].astype(cp.intp)] = 1

    # Fix last-byte ends: at each coord_ends[i]-1, if the byte is numeric,
    # force is_end=1
    d_last_pos = (d_coord_ends_arr - 1).astype(cp.intp)
    d_last_pos = d_last_pos[d_last_pos >= 0]
    if d_last_pos.shape[0] > 0:
        d_last_bytes = d_bytes[d_last_pos]
        d_is_numeric = (
            ((d_last_bytes >= ord('0')) & (d_last_bytes <= ord('9')))
            | (d_last_bytes == ord('.'))
            | (d_last_bytes == ord('e'))
            | (d_last_bytes == ord('E'))
            | (d_last_bytes == ord('-'))
            | (d_last_bytes == ord('+'))
        )
        d_is_end[d_last_pos[d_is_numeric]] = 1

    # Step 4: Extract number positions within coordinate regions only.
    d_num_starts, d_num_ends = extract_number_positions(
        d_is_start, d_is_end, d_mask=d_coord_mask,
    )
    del d_is_start, d_is_end, d_coord_mask

    # Step 5: Parse all numbers to float64.
    d_values = parse_ascii_floats(d_bytes, d_num_starts, d_num_ends)
    del d_num_starts, d_num_ends

    n_values = d_values.shape[0]
    if n_values == 0:
        return (
            cp.empty(0, dtype=cp.float64),
            cp.empty(0, dtype=cp.float64),
            dim,
            cp.zeros(structural.n_placemarks, dtype=cp.int32),
        )

    # Step 6: De-interleave flat value array into x (lon) and y (lat).
    # KML convention: longitude FIRST => x = lon, y = lat.
    if dim == 3:
        # [lon0, lat0, alt0, lon1, lat1, alt1, ...]
        d_x = d_values[0::3].copy()  # longitude
        d_y = d_values[1::3].copy()  # latitude
        n_coords = d_x.shape[0]
    else:
        # [lon0, lat0, lon1, lat1, ...] -- zero-copy strided views;
        # assembly calls cp.ascontiguousarray later when needed.
        d_x = d_values[0::2]  # longitude
        d_y = d_values[1::2]  # latitude
        n_coords = d_x.shape[0]
    del d_values

    # Step 7: Count coordinate pairs per Placemark.
    # Assign each coordinate region to its Placemark, then sum up
    # the values per Placemark and convert to pair counts.
    d_placemark_coord_counts = _count_coords_per_placemark(
        structural, n_coords, dim,
        d_bytes,
    )

    return d_x, d_y, dim, d_placemark_coord_counts


def _count_coords_per_placemark(
    structural: KmlStructuralResult,
    n_coords: int,
    dim: int,
    d_bytes: cp.ndarray,
) -> cp.ndarray:
    """Count coordinate pairs per Placemark.

    Uses the coordinate region assignment kernel to map each coordinate
    region to its parent Placemark, then sums values per region and
    converts to pair counts.

    Returns int32 array of shape ``(n_placemarks,)``.
    """
    n_placemarks = structural.n_placemarks
    n_regions = structural.d_coord_starts.shape[0]

    if n_placemarks == 0 or n_regions == 0:
        return cp.zeros(n_placemarks, dtype=cp.int32)

    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    # Assign each coordinate region to a Placemark
    kernels = _assign_coord_regions_kernels()
    d_region_pm_idx = cp.empty(n_regions, dtype=cp.int32)

    _launch_kernel(
        runtime,
        kernels["kml_assign_coord_regions"],
        n_regions,
        (
            (
                ptr(structural.d_coord_starts),
                ptr(structural.d_placemark_starts),
                ptr(structural.d_placemark_ends),
                ptr(d_region_pm_idx),
                np.int32(n_regions),
                np.int32(n_placemarks),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        ),
    )

    # Count the number of raw numeric values per region.
    # Each region spans [coord_starts[i], coord_ends[i]) in the byte stream.
    # The number of values in each region = (commas + 1) approximately,
    # but we already counted commas and spaces.  Recount more precisely:
    # we know total coords = n_coords, and we need per-region distribution.
    #
    # Simpler approach: count values per region using the coord region lengths
    # and the dimensionality.  We use commas + spaces to count tuples per
    # region, then convert tuples to coordinate pairs.
    cs_kernels = _count_commas_spaces_kernels()
    d_commas = cp.empty(n_regions, dtype=cp.int32)
    d_spaces = cp.empty(n_regions, dtype=cp.int32)

    _launch_kernel(
        runtime,
        cs_kernels["kml_count_commas_spaces"],
        n_regions,
        (
            (
                ptr(d_bytes),
                ptr(structural.d_coord_starts),
                ptr(structural.d_coord_ends),
                ptr(d_commas),
                ptr(d_spaces),
                np.int32(n_regions),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
    )

    # Each region has (spaces + 1) tuples (spaces separate tuples,
    # plus one for the first tuple).
    # But we need to handle empty regions: if commas == 0 and the region
    # is empty (no numeric content), tuples = 0.
    d_region_lengths = (structural.d_coord_ends - structural.d_coord_starts).astype(
        cp.int32,
    )
    d_region_nonempty = (d_region_lengths > 0).astype(cp.int32)
    d_tuples_per_region = (d_spaces + d_region_nonempty).astype(cp.int32)
    del d_commas, d_spaces, d_region_lengths, d_region_nonempty

    # Scatter tuple counts to Placemarks using atomic-like bincount.
    # d_region_pm_idx maps each region -> Placemark index.
    d_valid_mask = d_region_pm_idx >= 0
    d_valid_pm_idx = d_region_pm_idx[d_valid_mask]
    d_valid_tuples = d_tuples_per_region[d_valid_mask]
    del d_region_pm_idx, d_valid_mask, d_tuples_per_region

    if d_valid_pm_idx.shape[0] > 0:
        d_pm_pair_counts = cp.zeros(n_placemarks, dtype=cp.int32)
        # Use cupyx.scatter_add for GPU-side accumulation (Tier 2)
        import cupyx

        cupyx.scatter_add(d_pm_pair_counts, d_valid_pm_idx, d_valid_tuples)
    else:
        d_pm_pair_counts = cp.zeros(n_placemarks, dtype=cp.int32)
    del d_valid_pm_idx, d_valid_tuples

    return d_pm_pair_counts


def _count_rings_per_placemark(
    structural: KmlStructuralResult,
) -> cp.ndarray:
    """Count coordinate regions (rings) per Placemark.

    For Polygon Placemarks, each <coordinates> block within the Placemark
    corresponds to one ring (the first is the outer boundary, subsequent
    ones are inner boundaries/holes).

    Returns int32 array of shape ``(n_placemarks,)``.
    """
    n_placemarks = structural.n_placemarks
    n_regions = structural.d_coord_starts.shape[0]

    if n_placemarks == 0 or n_regions == 0:
        return cp.zeros(n_placemarks, dtype=cp.int32)

    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    kernels = _assign_coord_regions_kernels()
    d_region_pm_idx = cp.empty(n_regions, dtype=cp.int32)

    _launch_kernel(
        runtime,
        kernels["kml_assign_coord_regions"],
        n_regions,
        (
            (
                ptr(structural.d_coord_starts),
                ptr(structural.d_placemark_starts),
                ptr(structural.d_placemark_ends),
                ptr(d_region_pm_idx),
                np.int32(n_regions),
                np.int32(n_placemarks),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        ),
    )

    # Count regions per Placemark
    d_valid_mask = d_region_pm_idx >= 0
    d_valid_pm_idx = d_region_pm_idx[d_valid_mask]
    del d_region_pm_idx, d_valid_mask

    d_ring_counts = cp.zeros(n_placemarks, dtype=cp.int32)
    if d_valid_pm_idx.shape[0] > 0:
        import cupyx

        cupyx.scatter_add(
            d_ring_counts,
            d_valid_pm_idx,
            cp.ones(d_valid_pm_idx.shape[0], dtype=cp.int32),
        )
    del d_valid_pm_idx

    return d_ring_counts


# ---------------------------------------------------------------------------
# Offset building helpers (follow wkt_gpu.py patterns)
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


# ---------------------------------------------------------------------------
# Per-Placemark ring-level coordinate counting for Polygon assembly
# ---------------------------------------------------------------------------

def _count_coords_per_ring(
    structural: KmlStructuralResult,
    dim: int,
    d_bytes: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Count coordinate pairs per ring (coordinate region).

    For Polygon Placemarks, each coordinate region is one ring.
    Returns both the ring counts per Placemark and the coord pair
    counts per ring.

    Returns
    -------
    d_ring_counts : cp.ndarray
        Rings per Placemark, int32, shape ``(n_placemarks,)``.
    d_ring_coord_pair_counts : cp.ndarray
        Coordinate pairs per ring, int32, shape ``(n_coord_regions,)``.
    """
    n_regions = structural.d_coord_starts.shape[0]

    if n_regions == 0:
        return (
            cp.zeros(structural.n_placemarks, dtype=cp.int32),
            cp.empty(0, dtype=cp.int32),
        )

    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    # Count commas and spaces per region to get tuples per region
    cs_kernels = _count_commas_spaces_kernels()
    d_commas = cp.empty(n_regions, dtype=cp.int32)
    d_spaces = cp.empty(n_regions, dtype=cp.int32)

    _launch_kernel(
        runtime,
        cs_kernels["kml_count_commas_spaces"],
        n_regions,
        (
            (
                ptr(d_bytes),
                ptr(structural.d_coord_starts),
                ptr(structural.d_coord_ends),
                ptr(d_commas),
                ptr(d_spaces),
                np.int32(n_regions),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
    )

    # Tuples per region = spaces + 1 (for non-empty regions)
    d_region_lengths = (structural.d_coord_ends - structural.d_coord_starts).astype(
        cp.int32,
    )
    d_region_nonempty = (d_region_lengths > 0).astype(cp.int32)
    d_ring_coord_pair_counts = (d_spaces + d_region_nonempty).astype(cp.int32)
    del d_commas, d_spaces, d_region_lengths, d_region_nonempty

    # Ring counts per Placemark
    d_ring_counts = _count_rings_per_placemark(structural)

    return d_ring_counts, d_ring_coord_pair_counts


# ---------------------------------------------------------------------------
# Assembly into OwnedGeometryArray
# ---------------------------------------------------------------------------

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
        detail="GPU KML parse (empty)",
    )


def _assign_coord_region_to_placemark(
    structural: KmlStructuralResult,
) -> cp.ndarray:
    """Map each coordinate region to its parent Placemark index.

    Returns int32 array of shape ``(n_coord_regions,)``.
    Values are Placemark indices (0-based), or -1 for unassigned.
    """
    n_regions = structural.d_coord_starts.shape[0]
    n_placemarks = structural.n_placemarks

    if n_regions == 0 or n_placemarks == 0:
        return cp.full(n_regions, -1, dtype=cp.int32)

    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    kernels = _assign_coord_regions_kernels()
    d_region_pm_idx = cp.empty(n_regions, dtype=cp.int32)

    _launch_kernel(
        runtime,
        kernels["kml_assign_coord_regions"],
        n_regions,
        (
            (
                ptr(structural.d_coord_starts),
                ptr(structural.d_placemark_starts),
                ptr(structural.d_placemark_ends),
                ptr(d_region_pm_idx),
                np.int32(n_regions),
                np.int32(n_placemarks),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        ),
    )

    return d_region_pm_idx


def _assemble_kml_homogeneous(
    family: GeometryFamily,
    n_placemarks: int,
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_placemark_coord_counts: cp.ndarray,
    structural: KmlStructuralResult,
    dim: int,
    d_bytes: cp.ndarray,
) -> OwnedGeometryArray:
    """Build OwnedGeometryArray for a homogeneous KML file (single family)."""
    from vibespatial.io.pylibcudf import _build_device_single_family_owned

    d_empty_mask = (d_placemark_coord_counts == 0)
    d_validity = ~d_empty_mask

    if family == GeometryFamily.POINT:
        # Each Point has 1 coordinate pair
        d_geom_offsets = cp.arange(n_placemarks + 1, dtype=cp.int32)
        return _build_device_single_family_owned(
            family=GeometryFamily.POINT,
            validity_device=d_validity,
            x_device=cp.ascontiguousarray(d_x),
            y_device=cp.ascontiguousarray(d_y),
            geometry_offsets_device=d_geom_offsets,
            empty_mask_device=d_empty_mask,
            detail="GPU KML parse (Point)",
        )

    if family == GeometryFamily.LINESTRING:
        # geometry_offsets = prefix sum of per-Placemark coord counts
        d_geom_offsets = _device_compact_offsets(d_placemark_coord_counts)
        return _build_device_single_family_owned(
            family=GeometryFamily.LINESTRING,
            validity_device=d_validity,
            x_device=cp.ascontiguousarray(d_x),
            y_device=cp.ascontiguousarray(d_y),
            geometry_offsets_device=d_geom_offsets,
            empty_mask_device=d_empty_mask,
            detail="GPU KML parse (LineString)",
        )

    if family == GeometryFamily.POLYGON:
        # Polygon: each Placemark may have multiple <coordinates> blocks
        # (one per ring).  geometry_offsets = cumulative ring counts.
        # ring_offsets = cumulative coordinate pairs per ring.
        d_ring_counts, d_ring_coord_pair_counts = _count_coords_per_ring(
            structural, dim, d_bytes,
        )
        d_geom_offsets = _device_compact_offsets(d_ring_counts)
        d_ring_offsets = _device_compact_offsets(d_ring_coord_pair_counts)

        return _build_device_single_family_owned(
            family=GeometryFamily.POLYGON,
            validity_device=d_validity,
            x_device=cp.ascontiguousarray(d_x),
            y_device=cp.ascontiguousarray(d_y),
            geometry_offsets_device=d_geom_offsets,
            empty_mask_device=d_empty_mask,
            ring_offsets_device=d_ring_offsets,
            detail="GPU KML parse (Polygon)",
        )

    msg = f"Unsupported geometry family for KML assembly: {family}"
    raise ValueError(msg)


def _assemble_kml_mixed(
    n_placemarks: int,
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_placemark_coord_counts: cp.ndarray,
    structural: KmlStructuralResult,
    dim: int,
    d_bytes: cp.ndarray,
) -> OwnedGeometryArray:
    """Build OwnedGeometryArray for a mixed-type KML file.

    Partitions geometries by family and builds per-family buffers,
    then assembles them into a single mixed OwnedGeometryArray.
    """
    from vibespatial.geometry.owned import _device_gather_offset_slices
    from vibespatial.io.pylibcudf import _build_device_mixed_owned

    # Build per-Placemark coordinate offsets (n+1 array)
    d_coord_offsets = _device_compact_offsets(d_placemark_coord_counts)

    # Map KML tags to OGA tags (int8)
    d_oga_tags = structural.d_family_tags.copy().astype(cp.int8)
    # Mark unknown tags (-2) and MultiGeometry (6, unsupported) as -1 (invalid)
    d_oga_tags[d_oga_tags < 0] = -1
    d_oga_tags[d_oga_tags == KML_FAMILY_MULTI] = -1

    d_validity = (d_oga_tags >= 0)

    # Pre-compute 2D coordinate array for gather operations
    coords_2d = (
        cp.column_stack([d_x, d_y])
        if d_x.size > 0
        else cp.empty((0, 2), dtype=cp.float64)
    )

    family_devices: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}
    family_rows: dict[int, cp.ndarray] = {}

    # Only process families that are present
    d_valid_tags = structural.d_family_tags[structural.d_family_tags >= 0]
    d_valid_tags = d_valid_tags[d_valid_tags != KML_FAMILY_MULTI]
    if d_valid_tags.size == 0:
        return _build_empty_owned()
    unique_tags = cp.unique(d_valid_tags)
    h_unique_tags = unique_tags.get()

    for tag_val in h_unique_tags:
        tag_int = int(tag_val)
        if tag_int not in _KML_TAG_TO_FAMILY:
            continue
        family = _KML_TAG_TO_FAMILY[tag_int]

        rows = cp.flatnonzero(structural.d_family_tags == tag_int).astype(cp.int32)
        if rows.size == 0:
            continue
        family_rows[tag_int] = rows
        n_f = int(rows.size)

        if family == GeometryFamily.POINT:
            pt_starts = d_coord_offsets[rows]
            pt_x = d_x[pt_starts]
            pt_y = d_y[pt_starts]
            family_devices[family] = DeviceFamilyGeometryBuffer(
                family=family,
                x=cp.ascontiguousarray(pt_x),
                y=cp.ascontiguousarray(pt_y),
                geometry_offsets=cp.arange(n_f + 1, dtype=cp.int32),
                empty_mask=cp.zeros(n_f, dtype=cp.bool_),
            )

        elif family == GeometryFamily.LINESTRING:
            gathered, ls_geom_offsets = _device_gather_offset_slices(
                coords_2d, d_coord_offsets, rows,
            )
            family_devices[family] = DeviceFamilyGeometryBuffer(
                family=family,
                x=(
                    cp.ascontiguousarray(gathered[:, 0])
                    if gathered.size else cp.empty(0, dtype=cp.float64)
                ),
                y=(
                    cp.ascontiguousarray(gathered[:, 1])
                    if gathered.size else cp.empty(0, dtype=cp.float64)
                ),
                geometry_offsets=ls_geom_offsets,
                empty_mask=(ls_geom_offsets[1:] == ls_geom_offsets[:-1]),
            )

        elif family == GeometryFamily.POLYGON:
            # For Polygon: gather coordinates per Placemark, then build
            # ring offsets from the per-ring coordinate counts.
            gathered, sub_coord_offsets = _device_gather_offset_slices(
                coords_2d, d_coord_offsets, rows,
            )
            pg_x = (
                cp.ascontiguousarray(gathered[:, 0])
                if gathered.size else cp.empty(0, dtype=cp.float64)
            )
            pg_y = (
                cp.ascontiguousarray(gathered[:, 1])
                if gathered.size else cp.empty(0, dtype=cp.float64)
            )

            # Build ring counts and ring-level coord pair counts for the
            # polygon subset.
            sub_ring_counts = _count_rings_per_placemark_subset(
                structural, rows,
            )
            sub_geom_offsets = _device_compact_offsets(sub_ring_counts)

            sub_ring_coord_pair_counts = _count_ring_coords_for_subset(
                structural, rows, dim, d_bytes,
            )
            sub_ring_offsets = _device_compact_offsets(sub_ring_coord_pair_counts)

            family_devices[family] = DeviceFamilyGeometryBuffer(
                family=family,
                x=pg_x,
                y=pg_y,
                geometry_offsets=sub_geom_offsets,
                empty_mask=(sub_geom_offsets[1:] == sub_geom_offsets[:-1]),
                ring_offsets=sub_ring_offsets,
            )

    # Build tags and family_row_offsets
    d_family_row_offsets = cp.full(n_placemarks, -1, dtype=cp.int32)
    for tag_val, rows in family_rows.items():
        d_family_row_offsets[rows] = cp.arange(int(rows.size), dtype=cp.int32)

    return _build_device_mixed_owned(
        validity_device=d_validity,
        tags_device=d_oga_tags,
        family_row_offsets_device=d_family_row_offsets,
        family_devices=family_devices,
        detail="GPU KML parse (mixed)",
    )


def _count_rings_per_placemark_subset(
    structural: KmlStructuralResult,
    rows: cp.ndarray,
) -> cp.ndarray:
    """Count rings (coordinate regions) per Placemark for a subset of rows.

    Returns int32 array of shape ``(len(rows),)``.
    """
    # Get full ring counts and index into subset
    d_ring_counts = _count_rings_per_placemark(structural)
    return d_ring_counts[rows]


def _count_ring_coords_for_subset(
    structural: KmlStructuralResult,
    rows: cp.ndarray,
    dim: int,
    d_bytes: cp.ndarray,
) -> cp.ndarray:
    """Get per-ring coordinate pair counts for a subset of Placemarks.

    For each Placemark in `rows`, find its coordinate regions and count
    tuple pairs in each.  Returns a flat array of coordinate pair counts
    for all rings in the subset.
    """
    n_regions = structural.d_coord_starts.shape[0]
    if n_regions == 0:
        return cp.empty(0, dtype=cp.int32)

    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    # Get region-to-Placemark mapping
    d_region_pm_idx = _assign_coord_region_to_placemark(structural)

    # Count commas and spaces per region
    cs_kernels = _count_commas_spaces_kernels()
    d_commas = cp.empty(n_regions, dtype=cp.int32)
    d_spaces = cp.empty(n_regions, dtype=cp.int32)

    _launch_kernel(
        runtime,
        cs_kernels["kml_count_commas_spaces"],
        n_regions,
        (
            (
                ptr(d_bytes),
                ptr(structural.d_coord_starts),
                ptr(structural.d_coord_ends),
                ptr(d_commas),
                ptr(d_spaces),
                np.int32(n_regions),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
    )

    # Tuples per region
    d_region_lengths = (structural.d_coord_ends - structural.d_coord_starts).astype(
        cp.int32,
    )
    d_region_nonempty = (d_region_lengths > 0).astype(cp.int32)
    d_tuples_per_region = (d_spaces + d_region_nonempty).astype(cp.int32)
    del d_commas, d_spaces, d_region_lengths, d_region_nonempty

    # Build a set-membership mask for the subset rows
    d_row_set = cp.zeros(structural.n_placemarks, dtype=cp.uint8)
    d_row_set[rows] = 1

    # Filter to regions that belong to subset Placemarks
    d_valid = (d_region_pm_idx >= 0)
    d_in_subset = d_valid & (d_row_set[d_region_pm_idx.clip(0)] == 1)
    d_subset_region_indices = cp.flatnonzero(d_in_subset).astype(cp.int32)
    del d_valid, d_in_subset, d_row_set

    if d_subset_region_indices.shape[0] == 0:
        return cp.empty(0, dtype=cp.int32)

    return d_tuples_per_region[d_subset_region_indices]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_kml_gpu(d_bytes: cp.ndarray) -> OwnedGeometryArray:
    """Parse KML bytes on GPU and return device-resident geometry.

    Given a device-resident byte array containing a KML document, this
    function performs full GPU-accelerated parsing: structural analysis,
    coordinate extraction, and assembly into an ``OwnedGeometryArray``.

    Supported geometry types:

    - ``Point`` (full support)
    - ``LineString`` (full support)
    - ``Polygon`` (full support, including inner rings/holes)

    KML coordinate convention: longitude is FIRST (x=lon, y=lat).
    Altitude (3D) is detected and silently dropped.

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 array of KML file bytes, shape ``(n,)``.

    Returns
    -------
    OwnedGeometryArray
        Device-resident geometry array.  Coordinates are always fp64.
        Structural metadata (offsets, validity) is materialized on both
        host and device per the standard ``_build_device_*_owned``
        pattern.

    Notes
    -----
    Precision (ADR-0002):
        All coordinates are parsed and stored as fp64.  The structural
        analysis and counting kernels are integer-only byte
        classification -- no PrecisionPlan is needed for those stages.

    Tier classification (ADR-0033):
        Uses Tier 1 (custom NVRTC) for geometry-specific tag scanning
        and Tier 2 (CuPy) for element-wise operations.  Number parsing
        delegates to the gpu_parse primitives.

    Examples
    --------
    >>> import cupy as cp
    >>> kml = b'''<kml><Document>
    ...   <Placemark><Point><coordinates>-122.08,37.42,0</coordinates></Point></Placemark>
    ... </Document></kml>'''
    >>> d_bytes = cp.frombuffer(kml, dtype=cp.uint8)
    >>> owned = read_kml_gpu(d_bytes)
    >>> owned.row_count
    1
    """
    # ------------------------------------------------------------------
    # Stage 1: Structural analysis
    # ------------------------------------------------------------------
    structural = kml_structural_analysis(d_bytes)
    n_placemarks = structural.n_placemarks

    if n_placemarks == 0:
        return _build_empty_owned()

    # ------------------------------------------------------------------
    # Stage 2: Coordinate extraction
    # ------------------------------------------------------------------
    d_x, d_y, dim, d_placemark_coord_counts = _extract_kml_coordinates(
        d_bytes, structural,
    )

    if d_x.shape[0] == 0:
        return _build_empty_owned()

    # ------------------------------------------------------------------
    # Stage 3: Determine if homogeneous or mixed
    # ------------------------------------------------------------------
    # Filter to valid (known, supported) family tags
    d_valid_tags = structural.d_family_tags[structural.d_family_tags >= 0]
    d_valid_tags = d_valid_tags[d_valid_tags != KML_FAMILY_MULTI]

    if d_valid_tags.size == 0:
        return _build_empty_owned()

    d_unique_tags = cp.unique(d_valid_tags)
    n_unique = d_unique_tags.shape[0]
    h_unique_tags = d_unique_tags.get()

    if n_unique == 1:
        # Homogeneous file
        family = _KML_TAG_TO_FAMILY.get(int(h_unique_tags[0]))
        if family is None:
            msg = f"Unsupported KML geometry tag: {h_unique_tags[0]}"
            raise ValueError(msg)
        return _assemble_kml_homogeneous(
            family, n_placemarks, d_x, d_y,
            d_placemark_coord_counts, structural, dim, d_bytes,
        )

    # Mixed file
    return _assemble_kml_mixed(
        n_placemarks, d_x, d_y,
        d_placemark_coord_counts, structural, dim, d_bytes,
    )
