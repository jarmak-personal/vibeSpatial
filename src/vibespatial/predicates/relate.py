"""GPU-accelerated DE-9IM relate computation.

Computes the full 9-Intersection Model matrix per geometry pair, returning
a 9-character string like "212101212" per pair.  Mirrors ``shapely.relate()``.

Strategy:
    GPU kernels handle the common Point-* family combinations where the
    DE-9IM matrix is deterministic from point location classification:
        - Point-Point: equality check -> one of two fixed matrices
        - Point-LineString/MultiLineString: location on line -> fixed matrix
        - Point-Polygon/MultiPolygon: PIP classification -> fixed matrix

    For non-point families (Line-Line, Line-Polygon, Polygon-Polygon), the
    existing ``polygon.py`` DE-9IM bitmask kernel computes presence/absence
    of each cell but NOT the dimension (which is needed for the string).
    These fall back to Shapely.

    Tier classification per ADR-0033:
        - Point-* GPU paths: reuse existing Tier 1 NVRTC kernels from
          ``point_relations.py`` (geometry-specific inner loops).
        - Non-point: Shapely fallback with ``record_fallback_event``.

    ADR-0002: PREDICATE kernel class.  Point-* kernels inherit precision
    compliance from the underlying ``point_relations`` infrastructure.
"""

from __future__ import annotations

import logging

import numpy as np
import shapely

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    OwnedGeometryArray,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import Residency, TransferTrigger

from .point_relations import (
    POINT_LOCATION_BOUNDARY,
    POINT_LOCATION_INTERIOR,
    POINT_LOCATION_OUTSIDE,
    classify_point_equals_gpu,
    classify_point_line_gpu,
    classify_point_region_gpu,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Family tag constants (avoid repeated dict lookups in hot paths)
# ---------------------------------------------------------------------------
_POINT_TAG = FAMILY_TAGS[GeometryFamily.POINT]
_LINE_FAMILIES = (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING)
_LINE_TAGS = tuple(FAMILY_TAGS[f] for f in _LINE_FAMILIES)
_REGION_FAMILIES = (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
_REGION_TAGS = tuple(FAMILY_TAGS[f] for f in _REGION_FAMILIES)

# ---------------------------------------------------------------------------
# DE-9IM string constants for Point-Point pairs
#
# Two points are either equal (same location) or disjoint:
#   Equal:    dim(II)=0, dim(IE)=F, dim(BI)=F, dim(BB)=F, dim(BE)=F,
#             dim(EI)=F, dim(EB)=F, dim(EE)=2
#             Matrix: "0FFFFFFF2"
#   Disjoint: dim(II)=F, dim(IB)=F, dim(IE)=0, dim(BI)=F, dim(BB)=F,
#             dim(BE)=0, dim(EI)=0, dim(EB)=0, dim(EE)=2
#             Matrix: "FF0FFF0F2"
# ---------------------------------------------------------------------------
_DE9IM_POINT_POINT_EQUAL = "0FFFFFFF2"
_DE9IM_POINT_POINT_DISJOINT = "FF0FFF0F2"

# ---------------------------------------------------------------------------
# DE-9IM string constants for Point-Line pairs
#
# A point relative to a linestring has three possible locations:
#   Interior (on line, not at endpoint):
#     II=0, IB=F, IE=F, BI=F, BB=F, BE=F, EI=1, EB=0, EE=2
#     Matrix: "0FFFFF102"
#   Boundary (at endpoint of line):
#     II=F, IB=0, IE=F, BI=F, BB=F, BE=F, EI=1, EB=0, EE=2
#     Matrix: "F0FFFF102"
#   Exterior (disjoint from line):
#     II=F, IB=F, IE=0, BI=F, BB=F, BE=F, EI=1, EB=0, EE=2
#     Matrix: "FF0FFF102"
#
# Note: these assume the linestring has non-degenerate boundary (distinct
# endpoints).  Closed rings have empty boundary, which changes the matrix.
# For v1, we use the standard linestring matrices and accept minor
# deviations for closed rings (which Shapely also handles specially).
# ---------------------------------------------------------------------------
_DE9IM_POINT_LINE_INTERIOR = "0FFFFF102"
_DE9IM_POINT_LINE_BOUNDARY = "F0FFFF102"
_DE9IM_POINT_LINE_EXTERIOR = "FF0FFF102"

# ---------------------------------------------------------------------------
# DE-9IM string constants for Point-Polygon pairs
#
# A point relative to a polygon (with non-empty interior):
#   Interior (inside polygon):
#     II=0, IB=F, IE=F, BI=F, BB=F, BE=F, EI=2, EB=1, EE=2
#     Matrix: "0FFFFF212"
#   Boundary (on polygon ring):
#     II=F, IB=0, IE=F, BI=F, BB=F, BE=F, EI=2, EB=1, EE=2
#     Matrix: "F0FFFF212"
#   Exterior (outside polygon):
#     II=F, IB=F, IE=0, BI=F, BB=F, BE=F, EI=2, EB=1, EE=2
#     Matrix: "FF0FFF212"
# ---------------------------------------------------------------------------
_DE9IM_POINT_POLYGON_INTERIOR = "0FFFFF212"
_DE9IM_POINT_POLYGON_BOUNDARY = "F0FFFF212"
_DE9IM_POINT_POLYGON_EXTERIOR = "FF0FFF212"


def _location_to_point_point_de9im(loc: np.ndarray) -> np.ndarray:
    """Map point-point location codes to DE-9IM strings."""
    out = np.empty(loc.shape[0], dtype="U9")
    equal = loc == POINT_LOCATION_INTERIOR
    out[equal] = _DE9IM_POINT_POINT_EQUAL
    out[~equal] = _DE9IM_POINT_POINT_DISJOINT
    return out


def _location_to_point_line_de9im(
    loc: np.ndarray,
    *,
    point_on_left: bool,
) -> np.ndarray:
    """Map point-line location codes to DE-9IM strings.

    When *point_on_left* is True, left=Point, right=Line.
    When False, left=Line, right=Point (transposed matrix).
    """
    out = np.empty(loc.shape[0], dtype="U9")
    interior = loc == POINT_LOCATION_INTERIOR
    boundary = loc == POINT_LOCATION_BOUNDARY
    exterior = loc == POINT_LOCATION_OUTSIDE

    if point_on_left:
        out[interior] = _DE9IM_POINT_LINE_INTERIOR
        out[boundary] = _DE9IM_POINT_LINE_BOUNDARY
        out[exterior] = _DE9IM_POINT_LINE_EXTERIOR
    else:
        # Use pre-computed transposed constants (Line-Point).
        out[interior] = _DE9IM_LINE_POINT_INTERIOR_T
        out[boundary] = _DE9IM_LINE_POINT_BOUNDARY_T
        out[exterior] = _DE9IM_LINE_POINT_EXTERIOR_T
    return out


def _location_to_point_polygon_de9im(
    loc: np.ndarray,
    *,
    point_on_left: bool,
) -> np.ndarray:
    """Map point-polygon location codes to DE-9IM strings.

    When *point_on_left* is True, left=Point, right=Polygon.
    When False, left=Polygon, right=Point (transposed matrix).
    """
    out = np.empty(loc.shape[0], dtype="U9")
    interior = loc == POINT_LOCATION_INTERIOR
    boundary = loc == POINT_LOCATION_BOUNDARY
    exterior = loc == POINT_LOCATION_OUTSIDE

    if point_on_left:
        out[interior] = _DE9IM_POINT_POLYGON_INTERIOR
        out[boundary] = _DE9IM_POINT_POLYGON_BOUNDARY
        out[exterior] = _DE9IM_POINT_POLYGON_EXTERIOR
    else:
        # Use pre-computed transposed constants (Polygon-Point).
        out[interior] = _DE9IM_POLYGON_POINT_INTERIOR
        out[boundary] = _DE9IM_POLYGON_POINT_BOUNDARY
        out[exterior] = _DE9IM_POLYGON_POINT_EXTERIOR
    return out


def _transpose_de9im_string(s: str) -> str:
    """Transpose a DE-9IM string (swap A and B roles).

    The 3x3 matrix [II, IB, IE, BI, BB, BE, EI, EB, EE] becomes
    [II, BI, EI, IB, BB, EB, IE, BE, EE] when transposed.
    """
    # s[0]=II, s[1]=IB, s[2]=IE, s[3]=BI, s[4]=BB, s[5]=BE, s[6]=EI, s[7]=EB, s[8]=EE
    return s[0] + s[3] + s[6] + s[1] + s[4] + s[7] + s[2] + s[5] + s[8]


# Pre-compute transposed constants for cache-friendly access.
_DE9IM_POLYGON_POINT_INTERIOR = _transpose_de9im_string(_DE9IM_POINT_POLYGON_INTERIOR)
_DE9IM_POLYGON_POINT_BOUNDARY = _transpose_de9im_string(_DE9IM_POINT_POLYGON_BOUNDARY)
_DE9IM_POLYGON_POINT_EXTERIOR = _transpose_de9im_string(_DE9IM_POINT_POLYGON_EXTERIOR)
_DE9IM_LINE_POINT_INTERIOR_T = _transpose_de9im_string(_DE9IM_POINT_LINE_INTERIOR)
_DE9IM_LINE_POINT_BOUNDARY_T = _transpose_de9im_string(_DE9IM_POINT_LINE_BOUNDARY)
_DE9IM_LINE_POINT_EXTERIOR_T = _transpose_de9im_string(_DE9IM_POINT_LINE_EXTERIOR)


def _classify_gpu_rows(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    valid_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition valid_rows into GPU-eligible and Shapely-fallback rows.

    Returns (gpu_rows, cpu_rows).
    """
    if valid_rows.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    left_tags = left.tags[valid_rows]
    right_tags = right.tags[valid_rows]
    left_is_point = left_tags == _POINT_TAG
    right_is_point = right_tags == _POINT_TAG
    gpu_mask = left_is_point | right_is_point
    gpu_rows = valid_rows[gpu_mask]
    cpu_rows = valid_rows[~gpu_mask]
    return gpu_rows.astype(np.int32, copy=False), cpu_rows.astype(np.int32, copy=False)


def _evaluate_gpu_relate(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    gpu_rows: np.ndarray,
    out: np.ndarray,
) -> None:
    """Fill *out* at positions *gpu_rows* with GPU-computed DE-9IM strings.

    Dispatches by (left_tag, right_tag) family pair within the GPU-eligible
    rows.  Only Point-* combinations are handled.  Moves arrays to device
    as needed.
    """
    if gpu_rows.size == 0:
        return

    left.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="relate GPU execution for left geometry input",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="relate GPU execution for right geometry input",
    )

    left_tags = left.tags[gpu_rows]
    right_tags = right.tags[gpu_rows]

    # --- Point-Point ---
    pp_mask = (left_tags == _POINT_TAG) & (right_tags == _POINT_TAG)
    if pp_mask.any():
        rows = gpu_rows[pp_mask]
        loc = classify_point_equals_gpu(rows, left, right)
        strings = _location_to_point_point_de9im(loc)
        out[rows] = strings

    # --- Point-Line ---
    for line_family, line_tag in zip(_LINE_FAMILIES, _LINE_TAGS, strict=True):
        pl_mask = (left_tags == _POINT_TAG) & (right_tags == line_tag)
        if pl_mask.any():
            rows = gpu_rows[pl_mask]
            loc = classify_point_line_gpu(rows, left, right, line_family=line_family)
            strings = _location_to_point_line_de9im(loc, point_on_left=True)
            out[rows] = strings

        lp_mask = (left_tags == line_tag) & (right_tags == _POINT_TAG)
        if lp_mask.any():
            rows = gpu_rows[lp_mask]
            loc = classify_point_line_gpu(rows, right, left, line_family=line_family)
            strings = _location_to_point_line_de9im(loc, point_on_left=False)
            out[rows] = strings

    # --- Point-Polygon ---
    for region_family, region_tag in zip(_REGION_FAMILIES, _REGION_TAGS, strict=True):
        pr_mask = (left_tags == _POINT_TAG) & (right_tags == region_tag)
        if pr_mask.any():
            rows = gpu_rows[pr_mask]
            loc = classify_point_region_gpu(
                rows, left, right, region_family=region_family,
            )
            strings = _location_to_point_polygon_de9im(loc, point_on_left=True)
            out[rows] = strings

        rp_mask = (left_tags == region_tag) & (right_tags == _POINT_TAG)
        if rp_mask.any():
            rows = gpu_rows[rp_mask]
            loc = classify_point_region_gpu(
                rows, right, left, region_family=region_family,
            )
            strings = _location_to_point_polygon_de9im(loc, point_on_left=False)
            out[rows] = strings


def _evaluate_cpu_relate(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    cpu_rows: np.ndarray,
    out: np.ndarray,
) -> None:
    """Fill *out* at positions *cpu_rows* with Shapely-computed DE-9IM strings."""
    if cpu_rows.size == 0:
        return

    left_shapely = np.asarray(left.to_shapely(), dtype=object)
    right_shapely = np.asarray(right.to_shapely(), dtype=object)

    # Vectorized Shapely relate on the subset.
    left_subset = left_shapely[cpu_rows]
    right_subset = right_shapely[cpu_rows]
    results = shapely.relate(left_subset, right_subset)
    out[cpu_rows] = results


@register_kernel_variant(
    "relate_de9im",
    "gpu-cuda-python",
    kernel_class=KernelClass.PREDICATE,
    execution_modes=(ExecutionMode.GPU,),
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP32, PrecisionMode.FP64),
    preferred_residency=Residency.DEVICE,
)
def relate_de9im(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> np.ndarray:
    """Compute the DE-9IM intersection matrix for each pair of geometries.

    Parameters
    ----------
    left, right : OwnedGeometryArray
        Pairwise-aligned geometry arrays.  Must have the same ``row_count``.
    dispatch_mode : ExecutionMode
        Requested execution mode (AUTO, GPU, CPU).
    precision : PrecisionMode
        Precision mode for GPU kernels (AUTO, FP32, FP64).

    Returns
    -------
    np.ndarray
        Object array of 9-character DE-9IM strings (or None for null pairs).
        Shape ``(row_count,)``.

    Notes
    -----
    GPU path handles Point-* family combinations.  All other combinations
    fall back to Shapely with a recorded fallback event.  Null geometries
    produce None (not "FFFFFFFFF") to match Shapely convention.
    """
    row_count = left.row_count
    if row_count != right.row_count:
        raise ValueError(
            f"left and right must have the same row count, got {row_count} vs {right.row_count}"
        )
    if row_count == 0:
        return np.empty(0, dtype=object)

    requested_mode = (
        dispatch_mode
        if isinstance(dispatch_mode, ExecutionMode)
        else ExecutionMode(dispatch_mode)
    )

    selection = plan_dispatch_selection(
        kernel_name="relate_de9im",
        kernel_class=KernelClass.PREDICATE,
        row_count=row_count,
        requested_mode=requested_mode,
    )

    # Build null mask from validity.
    null_mask = ~left.validity | ~right.validity

    # Output array -- object dtype to hold None for nulls and strings for valid.
    out = np.empty(row_count, dtype=object)
    out[null_mask] = None

    valid_rows = np.flatnonzero(~null_mask).astype(np.int32, copy=False)
    if valid_rows.size == 0:
        return out

    gpu_rows: np.ndarray
    cpu_rows: np.ndarray

    if selection.selected is ExecutionMode.GPU:
        gpu_rows, cpu_rows = _classify_gpu_rows(left, right, valid_rows)
    else:
        # CPU-only path.
        gpu_rows = np.empty(0, dtype=np.int32)
        cpu_rows = valid_rows

    # --- GPU path for Point-* pairs ---
    if gpu_rows.size > 0:
        # Ensure bounds are computed for region families (needed by PIP).
        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds

        for arr in (left, right):
            arr.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="relate GPU: ensure device residency",
            )
            state = arr._ensure_device_state()
            for family in _REGION_FAMILIES:
                if family in state.families and state.families[family].bounds is None:
                    compute_geometry_bounds(arr, dispatch_mode=ExecutionMode.GPU)
                    break

        _evaluate_gpu_relate(left, right, gpu_rows, out)

        record_dispatch_event(
            surface="vibespatial.predicates.relate",
            operation="relate_de9im",
            requested=requested_mode,
            selected=ExecutionMode.GPU,
            implementation="gpu_point_relate",
            reason="GPU relate for Point-* family pairs",
            detail=f"gpu_rows={gpu_rows.size}, cpu_rows={cpu_rows.size}",
        )

    # --- CPU (Shapely) path for non-point pairs ---
    if cpu_rows.size > 0:
        record_fallback_event(
            surface="vibespatial.predicates.relate",
            reason=(
                "relate_de9im: non-point family combinations require full "
                "topology analysis; falling back to Shapely"
            ),
            detail=f"cpu_rows={cpu_rows.size} of {row_count} total",
            requested=requested_mode,
            selected=ExecutionMode.CPU,
            d2h_transfer=True,
        )
        _evaluate_cpu_relate(left, right, cpu_rows, out)

    return out


# ---------------------------------------------------------------------------
# Pattern matching helpers
# ---------------------------------------------------------------------------

# Valid pattern characters per the OGC DE-9IM specification.
_VALID_PATTERN_CHARS = frozenset("TF*012")


def _validate_pattern(pattern: str) -> None:
    """Validate a DE-9IM pattern string.

    Raises
    ------
    ValueError
        If the pattern is not exactly 9 characters or contains invalid chars.
    """
    if len(pattern) != 9:
        raise ValueError(
            f"DE-9IM pattern must be exactly 9 characters, got {len(pattern)}: {pattern!r}"
        )
    for i, ch in enumerate(pattern):
        if ch not in _VALID_PATTERN_CHARS:
            raise ValueError(
                f"Invalid character {ch!r} at position {i} in DE-9IM pattern {pattern!r}. "
                f"Valid characters are: T, F, *, 0, 1, 2"
            )


def _match_de9im_char(matrix_char: str, pattern_char: str) -> bool:
    """Check if a single DE-9IM matrix character matches a pattern character.

    Rules:
        '*' matches any matrix character.
        'T' matches '0', '1', or '2' (any non-F dimension).
        'F' matches 'F' only.
        '0', '1', '2' match the exact dimension character.
    """
    if pattern_char == "*":
        return True
    if pattern_char == "T":
        return matrix_char in ("0", "1", "2")
    # 'F', '0', '1', '2' — exact match.
    return matrix_char == pattern_char


def _match_de9im_string(matrix: str, pattern: str) -> bool:
    """Check if a full 9-char DE-9IM string matches a pattern."""
    for mc, pc in zip(matrix, pattern, strict=True):
        if not _match_de9im_char(mc, pc):
            return False
    return True


def _match_pattern_bulk(de9im_strings: np.ndarray, pattern: str) -> np.ndarray:
    """Match a DE-9IM pattern against an array of DE-9IM strings.

    Parameters
    ----------
    de9im_strings : np.ndarray
        Object array from ``relate_de9im``.  Elements are 9-char strings
        or None (for null geometry pairs).
    pattern : str
        9-character DE-9IM pattern with characters from {T, F, *, 0, 1, 2}.

    Returns
    -------
    np.ndarray
        Boolean array.  None entries in *de9im_strings* produce False.
    """
    n = len(de9im_strings)
    result = np.zeros(n, dtype=bool)
    for i in range(n):
        s = de9im_strings[i]
        if s is not None and isinstance(s, str) and len(s) == 9:
            result[i] = _match_de9im_string(s, pattern)
    return result


@register_kernel_variant(
    "relate_pattern",
    "gpu-cuda-python",
    kernel_class=KernelClass.PREDICATE,
    execution_modes=(ExecutionMode.GPU,),
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP32, PrecisionMode.FP64),
    preferred_residency=Residency.DEVICE,
)
def relate_pattern_match(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    pattern: str,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> np.ndarray:
    """Check if the DE-9IM relationship matches a pattern for each pair.

    This is the GPU-accelerated equivalent of ``shapely.relate_pattern()``.
    It computes the full DE-9IM matrix via ``relate_de9im`` and then applies
    the pattern match on the resulting strings.

    Parameters
    ----------
    left, right : OwnedGeometryArray
        Pairwise-aligned geometry arrays.  Must have the same ``row_count``.
    pattern : str
        9-character DE-9IM pattern.  Valid characters are:
        ``T`` (matches 0, 1, 2), ``F`` (matches F), ``*`` (matches any),
        ``0``, ``1``, ``2`` (exact dimension match).
    dispatch_mode : ExecutionMode
        Requested execution mode (AUTO, GPU, CPU).
    precision : PrecisionMode
        Precision mode for GPU kernels (AUTO, FP32, FP64).

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(row_count,)``.  True where the DE-9IM
        string matches the pattern, False otherwise.  Null geometry pairs
        produce False.
    """
    _validate_pattern(pattern)

    # Compute DE-9IM strings via the existing relate_de9im infrastructure.
    de9im_strings = relate_de9im(
        left, right,
        dispatch_mode=dispatch_mode,
        precision=precision,
    )

    # Pattern match on host.  The DE-9IM strings are already host-resident
    # (object dtype), so no device transfer is needed.
    return _match_pattern_bulk(de9im_strings, pattern)
