from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vibespatial.cuda._runtime import get_cuda_runtime
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.materialization import (
    MaterializationBoundary,
    record_materialization_event,
)

SUPPORTED_GEOM_TYPES = {
    "Point",
    "LineString",
    "Polygon",
    "MultiPoint",
    "MultiLineString",
    "MultiPolygon",
    "GeometryCollection",
}

# Predicates that can be evaluated via GPU DE-9IM bitmask for polygon pairs.
_POLYGON_DE9IM_PREDICATES = frozenset({
    "intersects", "contains", "within", "touches",
    "covers", "covered_by", "overlaps", "disjoint",
    "contains_properly",
})


def _device_array_detail(values: Any, *, side: str) -> str:
    size = int(getattr(values, "size", 0))
    itemsize = int(getattr(getattr(values, "dtype", None), "itemsize", 0))
    return f"side={side}, rows={size}, bytes={size * itemsize}"


def _record_device_join_materialization(
    values: Any,
    *,
    side: str,
    surface: str,
    operation: str,
    reason: str,
) -> None:
    record_materialization_event(
        surface=surface,
        boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
        operation=operation,
        reason=reason,
        detail=_device_array_detail(values, side=side),
        d2h_transfer=True,
        strict_disallowed=False,
    )


@dataclass(frozen=True)
class SpatialQueryExecution:
    requested: ExecutionMode
    selected: ExecutionMode
    implementation: str
    reason: str


@dataclass(frozen=True)
class RegularGridPointIndex:
    origin_x: float
    origin_y: float
    cell_width: float
    cell_height: float
    cols: int
    rows: int
    size: int


@dataclass(frozen=True)
class _DeviceCandidates:
    """Device-resident candidate pair indices from GPU bbox overlap."""
    d_left: Any  # CuPy int32 device array
    d_right: Any  # CuPy int32 device array
    total_pairs: int

    def to_host(self) -> tuple[np.ndarray, np.ndarray]:
        """Copy indices to host as numpy arrays."""
        runtime = get_cuda_runtime()
        _record_device_join_materialization(
            self.d_left,
            side="left",
            surface="vibespatial.spatial.query_types._DeviceCandidates.to_host",
            operation="device_candidates_to_host",
            reason="device candidate pairs crossed to host for CPU compatibility refinement",
        )
        left = runtime.copy_device_to_host(
            self.d_left,
            reason="device candidate left pairs host export",
        ).astype(np.int32, copy=False)
        _record_device_join_materialization(
            self.d_right,
            side="right",
            surface="vibespatial.spatial.query_types._DeviceCandidates.to_host",
            operation="device_candidates_to_host",
            reason="device candidate pairs crossed to host for CPU compatibility refinement",
        )
        right = runtime.copy_device_to_host(
            self.d_right,
            reason="device candidate right pairs host export",
        ).astype(np.int32, copy=False)
        return left, right


@dataclass(frozen=True)
class DeviceSpatialJoinResult:
    """Device-resident spatial join index pairs (Phase 2 zero-copy overlay).

    Holds left and right index arrays as CuPy int32 device arrays,
    eliminating the D->H->D round-trip when both sides of an overlay
    have owned (device-resident) geometry backing.

    Use :meth:`to_host` when host-side numpy arrays are needed (e.g.,
    for pandas attribute assembly or Shapely fallback paths).
    """

    d_left_idx: Any   # CuPy int32 device array
    d_right_idx: Any  # CuPy int32 device array

    def to_host(self) -> tuple[np.ndarray, np.ndarray]:
        """Copy index arrays to host as numpy int32 arrays."""
        runtime = get_cuda_runtime()
        _record_device_join_materialization(
            self.d_left_idx,
            side="left",
            surface="vibespatial.spatial.query_types.DeviceSpatialJoinResult.to_host",
            operation="device_spatial_join_indices_to_host",
            reason="device spatial join pairs crossed to host for public index export",
        )
        left = runtime.copy_device_to_host(
            self.d_left_idx,
            reason="device spatial join left-index host export",
        ).astype(np.int32, copy=False)
        _record_device_join_materialization(
            self.d_right_idx,
            side="right",
            surface="vibespatial.spatial.query_types.DeviceSpatialJoinResult.to_host",
            operation="device_spatial_join_indices_to_host",
            reason="device spatial join pairs crossed to host for public index export",
        )
        right = runtime.copy_device_to_host(
            self.d_right_idx,
            reason="device spatial join right-index host export",
        ).astype(np.int32, copy=False)
        return left, right

    @property
    def size(self) -> int:
        """Number of index pairs."""
        return int(self.d_left_idx.size)


@dataclass(frozen=True)
class SpatialJoinIndices:
    """ADR-0042: Typed contract for low-level spatial kernel index output.

    Device-native workflows may carry richer result objects, but the low-level
    spatial-query seam still uses integer index arrays. This frozen dataclass
    enforces that dtype invariant at construction time.
    """

    left: np.ndarray   # dtype np.intp
    right: np.ndarray  # dtype np.intp

    def __post_init__(self):
        object.__setattr__(self, "left", np.asarray(self.left, dtype=np.intp))
        object.__setattr__(self, "right", np.asarray(self.right, dtype=np.intp))

    @classmethod
    def from_raw(cls, left, right) -> SpatialJoinIndices:
        """Validate and coerce raw arrays into the canonical index form."""
        return cls(
            left=np.asarray(left, dtype=np.intp),
            right=np.asarray(right, dtype=np.intp),
        )


class _DeviceJoinResult:
    """Lazy device-to-host wrapper for join index pairs and optional distances.

    Keeps join results device-resident (zero-copy per ADR-0005) until pandas
    assembly actually needs numpy arrays for ``_reindex_with_indexers``.
    """

    __slots__ = (
        "_d_distances",
        "_d_left",
        "_d_right",
        "_h_distances",
        "_h_left",
        "_h_right",
    )

    def __init__(self, d_left, d_right, d_distances=None):
        self._d_left = d_left
        self._d_right = d_right
        self._d_distances = d_distances
        self._h_left = None
        self._h_right = None
        self._h_distances = None

    def _materialize(self):
        if self._h_left is not None:
            return
        runtime = get_cuda_runtime()
        _record_device_join_materialization(
            self._d_left,
            side="left",
            surface="vibespatial.spatial.query_types._DeviceJoinResult._materialize",
            operation="device_join_indices_to_host",
            reason="lazy device join pairs crossed to host for pandas-compatible assembly",
        )
        self._h_left = runtime.copy_device_to_host(
            self._d_left,
            reason="lazy device join left-index host export",
        ).astype(np.intp, copy=False)
        _record_device_join_materialization(
            self._d_right,
            side="right",
            surface="vibespatial.spatial.query_types._DeviceJoinResult._materialize",
            operation="device_join_indices_to_host",
            reason="lazy device join pairs crossed to host for pandas-compatible assembly",
        )
        self._h_right = runtime.copy_device_to_host(
            self._d_right,
            reason="lazy device join right-index host export",
        ).astype(np.intp, copy=False)
        if self._d_distances is not None:
            _record_device_join_materialization(
                self._d_distances,
                side="distances",
                surface="vibespatial.spatial.query_types._DeviceJoinResult._materialize",
                operation="device_join_distances_to_host",
                reason="lazy device join distances crossed to host for pandas-compatible assembly",
            )
            self._h_distances = runtime.copy_device_to_host(
                self._d_distances,
                reason="lazy device join distance host export",
            ).astype(np.float64, copy=False)

    @property
    def left(self) -> np.ndarray:
        self._materialize()
        return self._h_left

    @property
    def right(self) -> np.ndarray:
        self._materialize()
        return self._h_right

    @property
    def distances(self) -> np.ndarray | None:
        self._materialize()
        return self._h_distances

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (left, right) host arrays."""
        return self.left, self.right
