from __future__ import annotations

import numpy as np

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS, OwnedGeometryArray


def estimate_pip_work_polygon(
    polygon_geometry_offsets: np.ndarray,
    polygon_ring_offsets: np.ndarray,
    candidate_right_family_rows: np.ndarray,
) -> np.ndarray:
    ring_start = polygon_geometry_offsets[candidate_right_family_rows]
    ring_end = polygon_geometry_offsets[candidate_right_family_rows + 1]
    coord_start = polygon_ring_offsets[ring_start]
    coord_end = polygon_ring_offsets[ring_end]
    return (coord_end - coord_start).astype(np.int32)


def estimate_pip_work_multipolygon(
    multipolygon_geometry_offsets: np.ndarray,
    multipolygon_part_offsets: np.ndarray,
    multipolygon_ring_offsets: np.ndarray,
    candidate_right_family_rows: np.ndarray,
) -> np.ndarray:
    poly_start = multipolygon_geometry_offsets[candidate_right_family_rows]
    poly_end = multipolygon_geometry_offsets[candidate_right_family_rows + 1]
    ring_start = multipolygon_part_offsets[poly_start]
    ring_end = multipolygon_part_offsets[poly_end]
    coord_start = multipolygon_ring_offsets[ring_start]
    coord_end = multipolygon_ring_offsets[ring_end]
    return (coord_end - coord_start).astype(np.int32)


def should_bin_dispatch(work_estimates: np.ndarray) -> bool:
    if len(work_estimates) < 1024:
        return False
    mean_work = work_estimates.mean()
    if mean_work < 1:
        return False
    cv = work_estimates.std() / mean_work
    return cv > 2.0


def compute_work_estimates_for_candidates(
    candidate_rows_host: np.ndarray,
    right_array: OwnedGeometryArray,
) -> np.ndarray:
    tags = right_array.tags[candidate_rows_host]
    family_row_offsets = right_array.family_row_offsets[candidate_rows_host]
    work = np.zeros(len(candidate_rows_host), dtype=np.int32)

    if GeometryFamily.POLYGON in right_array.families:
        poly_buf = right_array.families[GeometryFamily.POLYGON]
        poly_mask = tags == FAMILY_TAGS[GeometryFamily.POLYGON]
        poly_family_rows = family_row_offsets[poly_mask]
        valid = poly_family_rows >= 0
        if valid.any():
            poly_work = estimate_pip_work_polygon(
                poly_buf.geometry_offsets,
                poly_buf.ring_offsets,
                poly_family_rows[valid],
            )
            poly_indices = np.flatnonzero(poly_mask)
            work[poly_indices[valid]] = poly_work

    if GeometryFamily.MULTIPOLYGON in right_array.families:
        mp_buf = right_array.families[GeometryFamily.MULTIPOLYGON]
        mp_mask = tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
        mp_family_rows = family_row_offsets[mp_mask]
        valid = mp_family_rows >= 0
        if valid.any():
            mp_work = estimate_pip_work_multipolygon(
                mp_buf.geometry_offsets,
                mp_buf.part_offsets,
                mp_buf.ring_offsets,
                mp_family_rows[valid],
            )
            mp_indices = np.flatnonzero(mp_mask)
            work[mp_indices[valid]] = mp_work

    return work


def to_python_result(values: np.ndarray) -> list[bool | None]:
    null_mask = np.equal(values, None)
    result = np.empty(len(values), dtype=object)
    result[:] = np.where(null_mask, False, values).astype(bool)
    result[null_mask] = None
    return list(result)


def candidate_rows_by_family(
    right: OwnedGeometryArray,
    candidate_rows: np.ndarray,
) -> dict[GeometryFamily, np.ndarray]:
    rows_by_family: dict[GeometryFamily, np.ndarray] = {}
    if candidate_rows.size == 0:
        return rows_by_family
    candidate_tags = right.tags[candidate_rows]
    for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        rows = candidate_rows[candidate_tags == FAMILY_TAGS[family]]
        if rows.size:
            rows_by_family[family] = rows.astype(np.int32, copy=False)
    return rows_by_family


def select_gpu_strategy(
    row_count: int,
    *,
    strategy: str,
    right_array: OwnedGeometryArray | None = None,
) -> str:
    del row_count
    if strategy != "auto":
        return strategy
    if right_array is not None:
        try:
            work_samples: list[np.ndarray] = []
            if GeometryFamily.POLYGON in right_array.families:
                buf = right_array.families[GeometryFamily.POLYGON]
                if buf.row_count > 0:
                    ring_s = buf.geometry_offsets[:-1]
                    ring_e = buf.geometry_offsets[1:]
                    coord_s = buf.ring_offsets[ring_s]
                    coord_e = buf.ring_offsets[ring_e]
                    work_samples.append((coord_e - coord_s).astype(np.int32))
            if GeometryFamily.MULTIPOLYGON in right_array.families:
                buf = right_array.families[GeometryFamily.MULTIPOLYGON]
                if buf.row_count > 0:
                    poly_s = buf.geometry_offsets[:-1]
                    poly_e = buf.geometry_offsets[1:]
                    ring_s = buf.part_offsets[poly_s]
                    ring_e = buf.part_offsets[poly_e]
                    coord_s = buf.ring_offsets[ring_s]
                    coord_e = buf.ring_offsets[ring_e]
                    work_samples.append((coord_e - coord_s).astype(np.int32))
            if work_samples:
                all_work = np.concatenate(work_samples)
                if should_bin_dispatch(all_work):
                    return "binned"
        except Exception:
            pass
    return "fused"


def compute_pip_center(points: OwnedGeometryArray, right_array: OwnedGeometryArray) -> tuple[float, float]:
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for owned in (points, right_array):
        for buffer in owned.families.values():
            if buffer.x.size > 0:
                all_x.append(buffer.x)
                all_y.append(buffer.y)
    if not all_x:
        return 0.0, 0.0
    combined_x = np.concatenate(all_x)
    combined_y = np.concatenate(all_y)
    cx = (float(np.nanmin(combined_x)) + float(np.nanmax(combined_x))) * 0.5
    cy = (float(np.nanmin(combined_y)) + float(np.nanmax(combined_y))) * 0.5
    return cx, cy


def initialize_coarse_result(points: OwnedGeometryArray, null_mask: np.ndarray) -> np.ndarray:
    coarse = np.zeros(points.row_count, dtype=object)
    coarse[:] = False
    coarse[null_mask] = None
    return coarse


def candidate_rows_from_coarse(coarse: np.ndarray) -> np.ndarray:
    return np.flatnonzero(coarse == True)  # noqa: E712


def dense_true_mask(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=bool)


def empty_gpu_bounds_placeholder() -> np.ndarray:
    return np.empty(0)


def empty_bool_mask_like(validity: np.ndarray) -> np.ndarray:
    return np.zeros(validity.shape[0], dtype=bool)


def cpu_return_device_fallback(values: list[bool | None]) -> np.ndarray:
    return np.array([bool(v) if v is not None else False for v in values], dtype=bool)


def work_cv(work_estimates: np.ndarray) -> float:
    return float(work_estimates.std() / max(work_estimates.mean(), 1e-9))
