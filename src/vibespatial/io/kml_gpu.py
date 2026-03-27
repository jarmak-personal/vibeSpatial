"""GPU KML reader -- structural analysis for KML/GML XML geometry extraction.

GPU-accelerated KML structural parser.  Given a device-resident byte array
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

Precision (ADR-0002):
    All kernels in this module are integer-only byte classification.
    No floating-point coordinate computation occurs, so no PrecisionPlan
    is needed (same rationale as gpu_parse/structural.py and wkt_gpu.py).
    Coordinate parsing will be handled downstream by gpu_parse.parse_ascii_floats
    which always produces fp64 -- storage precision is always fp64 per ADR-0002.
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
from vibespatial.io.gpu_parse.pattern import pattern_match

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

    # Suppress matches inside XML comments
    if d_comment_mask is not None:
        combined_hits = combined_hits & (~d_comment_mask)

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
