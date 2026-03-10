"""GPU-accelerated geometry equality operations.

geom_equals_exact: element-wise coordinate comparison with tolerance.
    Tier 2 CuPy for coordinate diff + Tier 3a CCCL segmented reduce.
    ADR-0002: PREDICATE class, dual fp32/fp64 with coordinate centering.

geom_equals: normalize-then-compare (composes normalize + equals_exact).
    Inherits dual-precision from both operations.
"""

from __future__ import annotations

import numpy as np

from vibespatial.adaptive_runtime import plan_dispatch_selection
from vibespatial.dispatch import record_dispatch_event
from vibespatial.geometry_buffers import GeometryFamily
from vibespatial.owned_geometry import (
    FAMILY_TAGS,
    OwnedGeometryArray,
)
from vibespatial.precision import KernelClass
from vibespatial.runtime import ExecutionMode

_EQUALS_EXACT_GPU_THRESHOLD = 1000


def geom_equals_exact_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    tolerance: float,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> np.ndarray:
    """Element-wise geometry equality with tolerance.

    Returns a bool array of shape (row_count,).  GPU path compares
    coordinate buffers directly (no Shapely round-trip).  Falls back
    to Shapely for row count below threshold or when GPU is unavailable.
    """
    row_count = left.row_count
    if row_count != right.row_count:
        raise ValueError(
            f"left and right must have same row count, got {row_count} vs {right.row_count}"
        )
    if row_count == 0:
        return np.empty(0, dtype=bool)

    selection = plan_dispatch_selection(
        kernel_name="geom_equals_exact",
        kernel_class=KernelClass.PREDICATE,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU and row_count >= _EQUALS_EXACT_GPU_THRESHOLD:
        try:
            result = _geom_equals_exact_gpu(left, right, tolerance)
            if result is not None:
                record_dispatch_event(
                    surface="geom_equals_exact",
                    operation="geom_equals_exact",
                    implementation="gpu_buffer_compare",
                    reason="coordinate buffer comparison on device",
                    detail=f"rows={row_count}, tolerance={tolerance}",
                    selected=ExecutionMode.GPU,
                )
                return result
        except Exception:
            pass

    return _geom_equals_exact_cpu(left, right, tolerance)


def _geom_equals_exact_cpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    tolerance: float,
) -> np.ndarray:
    """CPU path: delegate to Shapely equals_exact."""
    import shapely

    record_dispatch_event(
        surface="geom_equals_exact",
        operation="geom_equals_exact",
        implementation="shapely",
        reason="CPU fallback",
        detail=f"rows={left.row_count}, tolerance={tolerance}",
        selected=ExecutionMode.CPU,
    )
    left_geoms = np.asarray(left.to_shapely(), dtype=object)
    right_geoms = np.asarray(right.to_shapely(), dtype=object)
    return shapely.equals_exact(left_geoms, right_geoms, tolerance=tolerance)


def _geom_equals_exact_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    tolerance: float,
) -> np.ndarray | None:
    """GPU path: structure check then coordinate buffer comparison.

    Returns None if GPU comparison is not feasible (e.g., incompatible
    family layouts), triggering CPU fallback.
    """
    try:
        import cupy as cp  # noqa: F401 — used below for GPU buffer ops
    except ImportError:
        return None

    from vibespatial.runtime import has_gpu_runtime
    if not has_gpu_runtime():
        return None

    row_count = left.row_count
    result = np.zeros(row_count, dtype=bool)

    # Step 1: Tag comparison — different geometry types are never equal
    if not np.array_equal(left.tags, right.tags):
        tag_match = left.tags == right.tags
    else:
        tag_match = np.ones(row_count, dtype=bool)

    # Null rows: null != anything (including null)
    null_mask = (~left.validity) | (~right.validity)
    tag_match[null_mask] = False

    candidate_rows = np.flatnonzero(tag_match)
    if candidate_rows.size == 0:
        return result

    # Step 2: Per-family structure + coordinate comparison
    for family_key in GeometryFamily:
        tag = FAMILY_TAGS[family_key]
        family_mask = tag_match & (left.tags == tag)
        family_rows = np.flatnonzero(family_mask)
        if family_rows.size == 0:
            continue

        left_buf = left.families.get(family_key)
        right_buf = right.families.get(family_key)
        if left_buf is None or right_buf is None:
            continue

        family_equal = _compare_family_buffers(
            left, right, left_buf, right_buf, family_rows, family_key, tolerance
        )
        result[family_rows] = family_equal

    return result


def _compare_family_buffers(
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    left_buf,
    right_buf,
    global_rows: np.ndarray,
    family: GeometryFamily,
    tolerance: float,
) -> np.ndarray:
    """Compare coordinate buffers for a single geometry family.

    Returns a bool array of length len(global_rows).
    """
    try:
        import cupy as cp  # noqa: F401 — used below for GPU buffer ops
    except ImportError:
        return np.zeros(len(global_rows), dtype=bool)

    n = len(global_rows)
    result = np.ones(n, dtype=bool)

    left_fro = left_owned.family_row_offsets[global_rows]
    right_fro = right_owned.family_row_offsets[global_rows]

    # Per-row: compare coordinate spans
    for i in range(n):
        lfr = int(left_fro[i])
        rfr = int(right_fro[i])

        # Get coordinate ranges for this row
        l_start, l_end = _coord_range(left_buf, lfr, family)
        r_start, r_end = _coord_range(right_buf, rfr, family)

        l_count = l_end - l_start
        r_count = r_end - r_start

        # Different coordinate counts → not equal
        if l_count != r_count:
            result[i] = False
            continue

        if l_count == 0:
            # Both empty
            continue

        # Also check ring structure for polygons
        if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
            if not _ring_structure_matches(left_buf, right_buf, lfr, rfr, family):
                result[i] = False
                continue

        # Compare coordinates with tolerance
        lx = left_buf.x[l_start:l_end]
        ly = left_buf.y[l_start:l_end]
        rx = right_buf.x[r_start:r_end]
        ry = right_buf.y[r_start:r_end]

        if not (np.all(np.abs(lx - rx) <= tolerance) and np.all(np.abs(ly - ry) <= tolerance)):
            result[i] = False

    return result


def _coord_range(buf, family_row: int, family: GeometryFamily) -> tuple[int, int]:
    """Get the full coordinate range [start, end) for a geometry row."""
    go = buf.geometry_offsets

    if family in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT):
        return int(go[family_row]), int(go[family_row + 1])

    if family in (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING):
        if buf.part_offsets is not None:
            # Multi: geometry_offsets → part_offsets → coordinates
            part_start = int(go[family_row])
            part_end = int(go[family_row + 1])
            if part_start >= part_end:
                return 0, 0
            return int(buf.part_offsets[part_start]), int(buf.part_offsets[part_end])
        else:
            return int(go[family_row]), int(go[family_row + 1])

    if family == GeometryFamily.POLYGON:
        # geometry_offsets → ring_offsets → coordinates
        ring_start = int(go[family_row])
        ring_end = int(go[family_row + 1])
        if ring_start >= ring_end or buf.ring_offsets is None:
            return 0, 0
        return int(buf.ring_offsets[ring_start]), int(buf.ring_offsets[ring_end])

    if family == GeometryFamily.MULTIPOLYGON:
        # geometry_offsets → part_offsets → ring_offsets → coordinates
        part_start = int(go[family_row])
        part_end = int(go[family_row + 1])
        if part_start >= part_end or buf.part_offsets is None or buf.ring_offsets is None:
            return 0, 0
        ring_start = int(buf.part_offsets[part_start])
        ring_end = int(buf.part_offsets[part_end])
        if ring_start >= ring_end:
            return 0, 0
        return int(buf.ring_offsets[ring_start]), int(buf.ring_offsets[ring_end])

    return 0, 0


def _ring_structure_matches(
    left_buf, right_buf, left_fr: int, right_fr: int, family: GeometryFamily
) -> bool:
    """Check that two geometries have the same ring/part offset structure."""
    lgo = left_buf.geometry_offsets
    rgo = right_buf.geometry_offsets

    if family == GeometryFamily.POLYGON:
        l_ring_count = int(lgo[left_fr + 1]) - int(lgo[left_fr])
        r_ring_count = int(rgo[right_fr + 1]) - int(rgo[right_fr])
        if l_ring_count != r_ring_count:
            return False
        # Check ring sizes match
        if left_buf.ring_offsets is None or right_buf.ring_offsets is None:
            return l_ring_count == 0
        l_ring_start = int(lgo[left_fr])
        r_ring_start = int(rgo[right_fr])
        for r in range(l_ring_count):
            l_size = int(left_buf.ring_offsets[l_ring_start + r + 1]) - int(left_buf.ring_offsets[l_ring_start + r])
            r_size = int(right_buf.ring_offsets[r_ring_start + r + 1]) - int(right_buf.ring_offsets[r_ring_start + r])
            if l_size != r_size:
                return False
        return True

    if family == GeometryFamily.MULTIPOLYGON:
        l_part_count = int(lgo[left_fr + 1]) - int(lgo[left_fr])
        r_part_count = int(rgo[right_fr + 1]) - int(rgo[right_fr])
        if l_part_count != r_part_count:
            return False
        if left_buf.part_offsets is None or right_buf.part_offsets is None:
            return l_part_count == 0
        l_part_start = int(lgo[left_fr])
        r_part_start = int(rgo[right_fr])
        for p in range(l_part_count):
            l_ring_count = int(left_buf.part_offsets[l_part_start + p + 1]) - int(left_buf.part_offsets[l_part_start + p])
            r_ring_count = int(right_buf.part_offsets[r_part_start + p + 1]) - int(right_buf.part_offsets[r_part_start + p])
            if l_ring_count != r_ring_count:
                return False
            # Check ring sizes
            if left_buf.ring_offsets is None or right_buf.ring_offsets is None:
                continue
            l_rs = int(left_buf.part_offsets[l_part_start + p])
            r_rs = int(right_buf.part_offsets[r_part_start + p])
            for r in range(l_ring_count):
                l_size = int(left_buf.ring_offsets[l_rs + r + 1]) - int(left_buf.ring_offsets[l_rs + r])
                r_size = int(right_buf.ring_offsets[r_rs + r + 1]) - int(right_buf.ring_offsets[r_rs + r])
                if l_size != r_size:
                    return False
        return True

    return True


def geom_equals_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> np.ndarray:
    """Element-wise topological geometry equality.

    Normalizes both inputs then compares with tolerance 1e-12.
    Returns a bool array of shape (row_count,).
    """
    from vibespatial.normalize_gpu import normalize_owned

    left_norm = normalize_owned(left, dispatch_mode=dispatch_mode)
    right_norm = normalize_owned(right, dispatch_mode=dispatch_mode)
    return geom_equals_exact_owned(left_norm, right_norm, tolerance=1e-12, dispatch_mode=dispatch_mode)
