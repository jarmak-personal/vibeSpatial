"""Shared overlay pipeline data structures.

Extracted from overlay_gpu.py so that multiple modules (overlay_gpu,
make_valid_gpu) can import lightweight data structures without pulling
in the full overlay pipeline and its CUDA kernel strings.

Phase 8 (vibeSpatial-p23.8): All four overlay data structures are
device-primary with lazy host materialization.  GPU-only consumers
never trigger D->H copies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vibespatial.cuda._runtime import DeviceArray, get_cuda_runtime
from vibespatial.runtime import RuntimeSelection


def _runtime_host_array(runtime, value, dtype, *, reason: str):
    if value is None:
        return None
    return np.asarray(runtime.copy_device_to_host(value, reason=reason), dtype=dtype)


@dataclass(frozen=True)
class SplitEventDeviceState:
    source_segment_ids: DeviceArray
    t: DeviceArray
    x: DeviceArray
    y: DeviceArray
    source_side: DeviceArray | None = None
    row_indices: DeviceArray | None = None
    part_indices: DeviceArray | None = None
    ring_indices: DeviceArray | None = None


@dataclass
class SplitEventTable:
    """Split event table with lazy host materialization.

    When produced by the GPU pipeline, all arrays live in ``device_state``
    and host numpy arrays are lazily copied on first property access.
    GPU-only consumers that read only ``device_state``, ``count``,
    ``left_segment_count``, ``right_segment_count``, and
    ``runtime_selection`` never trigger device-to-host copies.
    """

    left_segment_count: int
    right_segment_count: int
    runtime_selection: RuntimeSelection
    device_state: SplitEventDeviceState
    _count: int = 0
    # Host arrays — lazily materialized from device_state on first access.
    _source_segment_ids: np.ndarray | None = None
    _source_side: np.ndarray | None = None
    _row_indices: np.ndarray | None = None
    _part_indices: np.ndarray | None = None
    _ring_indices: np.ndarray | None = None
    _t: np.ndarray | None = None
    _x: np.ndarray | None = None
    _y: np.ndarray | None = None

    def _ensure_host(self) -> None:
        """Lazily copy host arrays from device_state on first access."""
        if self._source_segment_ids is not None:
            return
        ds = self.device_state
        if ds is None:
            return
        runtime = get_cuda_runtime()
        self._source_segment_ids = _runtime_host_array(
            runtime,
            ds.source_segment_ids,
            np.int32,
            reason="overlay split-event source-segment host export",
        )
        self._t = _runtime_host_array(
            runtime,
            ds.t,
            np.float64,
            reason="overlay split-event parameter host export",
        )
        self._x = _runtime_host_array(
            runtime,
            ds.x,
            np.float64,
            reason="overlay split-event x-coordinate host export",
        )
        self._y = _runtime_host_array(
            runtime,
            ds.y,
            np.float64,
            reason="overlay split-event y-coordinate host export",
        )
        if ds.source_side is not None:
            self._source_side = _runtime_host_array(
                runtime,
                ds.source_side,
                np.int8,
                reason="overlay split-event source-side host export",
            )
        else:
            # Derive from source_segment_ids + left_segment_count
            ids = self._source_segment_ids
            self._source_side = np.where(ids < self.left_segment_count, 1, 2).astype(np.int8)
        if ds.row_indices is not None:
            self._row_indices = _runtime_host_array(
                runtime,
                ds.row_indices,
                np.int32,
                reason="overlay split-event row-index host export",
            )
        else:
            self._row_indices = np.empty(0, dtype=np.int32)
        if ds.part_indices is not None:
            self._part_indices = _runtime_host_array(
                runtime,
                ds.part_indices,
                np.int32,
                reason="overlay split-event part-index host export",
            )
        else:
            self._part_indices = np.empty(0, dtype=np.int32)
        if ds.ring_indices is not None:
            self._ring_indices = _runtime_host_array(
                runtime,
                ds.ring_indices,
                np.int32,
                reason="overlay split-event ring-index host export",
            )
        else:
            self._ring_indices = np.empty(0, dtype=np.int32)

    @property
    def source_segment_ids(self) -> np.ndarray:
        self._ensure_host()
        return self._source_segment_ids  # type: ignore[return-value]

    @property
    def source_side(self) -> np.ndarray:
        self._ensure_host()
        return self._source_side  # type: ignore[return-value]

    @property
    def row_indices(self) -> np.ndarray:
        self._ensure_host()
        return self._row_indices  # type: ignore[return-value]

    @property
    def part_indices(self) -> np.ndarray:
        self._ensure_host()
        return self._part_indices  # type: ignore[return-value]

    @property
    def ring_indices(self) -> np.ndarray:
        self._ensure_host()
        return self._ring_indices  # type: ignore[return-value]

    @property
    def t(self) -> np.ndarray:
        self._ensure_host()
        return self._t  # type: ignore[return-value]

    @property
    def x(self) -> np.ndarray:
        self._ensure_host()
        return self._x  # type: ignore[return-value]

    @property
    def y(self) -> np.ndarray:
        self._ensure_host()
        return self._y  # type: ignore[return-value]

    @property
    def count(self) -> int:
        if self._count > 0:
            return self._count
        if self.device_state is not None and self.device_state.source_segment_ids is not None:
            return int(self.device_state.source_segment_ids.size)
        if self._source_segment_ids is not None:
            return int(self._source_segment_ids.size)
        return 0


@dataclass(frozen=True)
class AtomicEdgeDeviceState:
    source_segment_ids: DeviceArray
    direction: DeviceArray
    src_x: DeviceArray
    src_y: DeviceArray
    dst_x: DeviceArray
    dst_y: DeviceArray
    # Metadata arrays — stored on device to avoid D->H transfers in
    # GPU-only consumers (e.g. build_gpu_half_edge_graph).
    row_indices: DeviceArray | None = None
    part_indices: DeviceArray | None = None
    ring_indices: DeviceArray | None = None
    source_side: DeviceArray | None = None
    # Bitset over all source atoms collapsed into this geometric edge:
    # bit 0 = left, bit 1 = right. Coincident boundaries therefore retain
    # dual-source provenance after geometric deduplication.
    source_membership: DeviceArray | None = None


@dataclass
class AtomicEdgeTable:
    """Atomic edge table with lazy host materialization.

    Host numpy arrays are lazily copied from device_state on first access,
    matching the HalfEdgeGraph lazy pattern.  GPU-only consumers that read
    only ``device_state``, ``count``, ``left_segment_count``,
    ``right_segment_count``, and ``runtime_selection`` never trigger the
    device-to-host copies.
    """

    left_segment_count: int
    right_segment_count: int
    runtime_selection: RuntimeSelection
    device_state: AtomicEdgeDeviceState
    _count: int = 0
    # Host arrays — lazily materialized from device_state on first access.
    _source_segment_ids: np.ndarray | None = None
    _source_side: np.ndarray | None = None
    _row_indices: np.ndarray | None = None
    _part_indices: np.ndarray | None = None
    _ring_indices: np.ndarray | None = None
    _direction: np.ndarray | None = None
    _src_x: np.ndarray | None = None
    _src_y: np.ndarray | None = None
    _dst_x: np.ndarray | None = None
    _dst_y: np.ndarray | None = None

    def _ensure_host(self) -> None:
        """Lazily copy host arrays from device_state on first access."""
        if self._source_segment_ids is not None:
            return
        ds = self.device_state
        if ds is None:
            return
        runtime = get_cuda_runtime()
        self._source_segment_ids = _runtime_host_array(
            runtime,
            ds.source_segment_ids,
            np.int32,
            reason="overlay atomic-edge source-segment host export",
        )
        self._direction = _runtime_host_array(
            runtime,
            ds.direction,
            np.int8,
            reason="overlay atomic-edge direction host export",
        )
        self._src_x = _runtime_host_array(
            runtime,
            ds.src_x,
            np.float64,
            reason="overlay atomic-edge source-x host export",
        )
        self._src_y = _runtime_host_array(
            runtime,
            ds.src_y,
            np.float64,
            reason="overlay atomic-edge source-y host export",
        )
        self._dst_x = _runtime_host_array(
            runtime,
            ds.dst_x,
            np.float64,
            reason="overlay atomic-edge target-x host export",
        )
        self._dst_y = _runtime_host_array(
            runtime,
            ds.dst_y,
            np.float64,
            reason="overlay atomic-edge target-y host export",
        )

    @property
    def source_segment_ids(self) -> np.ndarray:
        self._ensure_host()
        return self._source_segment_ids  # type: ignore[return-value]

    @property
    def source_side(self) -> np.ndarray:
        if self._source_side is None:
            # Check device_state first to avoid triggering _ensure_host
            # just to derive source_side from source_segment_ids.
            ds = self.device_state
            if ds is not None and ds.source_side is not None:
                runtime = get_cuda_runtime()
                self._source_side = _runtime_host_array(
                    runtime,
                    ds.source_side,
                    np.int8,
                    reason="overlay atomic-edge source-side host export",
                )
            else:
                # Derive from source_segment_ids + left_segment_count
                ids = self.source_segment_ids
                self._source_side = np.where(ids < self.left_segment_count, 1, 2).astype(np.int8)
        return self._source_side  # type: ignore[return-value]

    @property
    def row_indices(self) -> np.ndarray:
        if self._row_indices is None:
            ds = self.device_state
            if ds is not None and ds.row_indices is not None:
                runtime = get_cuda_runtime()
                self._row_indices = _runtime_host_array(
                    runtime,
                    ds.row_indices,
                    np.int32,
                    reason="overlay atomic-edge row-index host export",
                )
            else:
                return np.empty(0, dtype=np.int32)
        return self._row_indices

    @property
    def part_indices(self) -> np.ndarray:
        if self._part_indices is None:
            ds = self.device_state
            if ds is not None and ds.part_indices is not None:
                runtime = get_cuda_runtime()
                self._part_indices = _runtime_host_array(
                    runtime,
                    ds.part_indices,
                    np.int32,
                    reason="overlay atomic-edge part-index host export",
                )
            else:
                return np.empty(0, dtype=np.int32)
        return self._part_indices

    @property
    def ring_indices(self) -> np.ndarray:
        if self._ring_indices is None:
            ds = self.device_state
            if ds is not None and ds.ring_indices is not None:
                runtime = get_cuda_runtime()
                self._ring_indices = _runtime_host_array(
                    runtime,
                    ds.ring_indices,
                    np.int32,
                    reason="overlay atomic-edge ring-index host export",
                )
            else:
                return np.empty(0, dtype=np.int32)
        return self._ring_indices

    @property
    def direction(self) -> np.ndarray:
        self._ensure_host()
        return self._direction  # type: ignore[return-value]

    @property
    def src_x(self) -> np.ndarray:
        self._ensure_host()
        return self._src_x  # type: ignore[return-value]

    @property
    def src_y(self) -> np.ndarray:
        self._ensure_host()
        return self._src_y  # type: ignore[return-value]

    @property
    def dst_x(self) -> np.ndarray:
        self._ensure_host()
        return self._dst_x  # type: ignore[return-value]

    @property
    def dst_y(self) -> np.ndarray:
        self._ensure_host()
        return self._dst_y  # type: ignore[return-value]

    @property
    def count(self) -> int:
        if self._count > 0:
            return self._count
        if self.device_state is not None and self.device_state.source_segment_ids is not None:
            return int(self.device_state.source_segment_ids.size)
        if self._source_segment_ids is not None:
            return int(self._source_segment_ids.size)
        return 0


@dataclass(frozen=True)
class HalfEdgeGraphDeviceState:
    node_x: DeviceArray | None
    node_y: DeviceArray | None
    src_node_ids: DeviceArray | None
    dst_node_ids: DeviceArray | None
    angle: DeviceArray | None
    sorted_edge_ids: DeviceArray | None
    edge_positions: DeviceArray | None
    next_edge_ids: DeviceArray
    src_x: DeviceArray
    src_y: DeviceArray
    # Metadata arrays — carried from AtomicEdgeTable to avoid D->H
    # round-trips when GPU consumers need per-edge source metadata.
    source_segment_ids: DeviceArray | None = None
    source_side: DeviceArray | None = None
    source_membership: DeviceArray | None = None
    row_indices: DeviceArray | None = None
    part_indices: DeviceArray | None = None
    ring_indices: DeviceArray | None = None
    direction: DeviceArray | None = None


@dataclass
class HalfEdgeGraph:
    """Half-edge graph with device-primary storage and lazy host materialization.

    All arrays (both topology and per-edge metadata) are stored on device
    via ``device_state`` and lazily copied to host on first property access.
    GPU-only consumers that read ``device_state``, ``edge_count``,
    ``node_count``, ``left_segment_count``, ``right_segment_count``, and
    ``runtime_selection`` never trigger device-to-host copies.
    """

    left_segment_count: int
    right_segment_count: int
    runtime_selection: RuntimeSelection
    device_state: HalfEdgeGraphDeviceState
    _edge_count: int = 0
    _node_count: int = 0
    isolate_rows: bool = False
    # Host arrays — lazily materialized from device_state on first access.
    _source_segment_ids: np.ndarray | None = None
    _source_side: np.ndarray | None = None
    _row_indices: np.ndarray | None = None
    _part_indices: np.ndarray | None = None
    _ring_indices: np.ndarray | None = None
    _direction: np.ndarray | None = None
    _src_x: np.ndarray | None = None
    _src_y: np.ndarray | None = None
    _dst_x: np.ndarray | None = None
    _dst_y: np.ndarray | None = None
    _node_x: np.ndarray | None = None
    _node_y: np.ndarray | None = None
    _src_node_ids: np.ndarray | None = None
    _dst_node_ids: np.ndarray | None = None
    _angle: np.ndarray | None = None
    _sorted_edge_ids: np.ndarray | None = None
    _edge_positions: np.ndarray | None = None
    _next_edge_ids: np.ndarray | None = None

    def _ensure_host_topology(self) -> None:
        """Lazily copy topology arrays from device to host on first access."""
        if self._next_edge_ids is not None:
            return
        ds = self.device_state
        if ds is None:
            return
        runtime = get_cuda_runtime()
        self._src_x = _runtime_host_array(
            runtime,
            ds.src_x,
            np.float64,
            reason="overlay half-edge source-x host export",
        )
        self._src_y = _runtime_host_array(
            runtime,
            ds.src_y,
            np.float64,
            reason="overlay half-edge source-y host export",
        )
        self._dst_x = _runtime_host_array(
            runtime,
            getattr(ds, "dst_x", None),
            np.float64,
            reason="overlay half-edge target-x host export",
        )
        self._dst_y = _runtime_host_array(
            runtime,
            getattr(ds, "dst_y", None),
            np.float64,
            reason="overlay half-edge target-y host export",
        )
        self._next_edge_ids = _runtime_host_array(
            runtime,
            ds.next_edge_ids,
            np.int32,
            reason="overlay half-edge next-edge host export",
        )
        if ds.node_x is not None:
            self._node_x = _runtime_host_array(
                runtime,
                ds.node_x,
                np.float64,
                reason="overlay half-edge node-x host export",
            )
            self._node_y = _runtime_host_array(
                runtime,
                ds.node_y,
                np.float64,
                reason="overlay half-edge node-y host export",
            )
            self._src_node_ids = _runtime_host_array(
                runtime,
                ds.src_node_ids,
                np.int32,
                reason="overlay half-edge source-node host export",
            )
            self._dst_node_ids = _runtime_host_array(
                runtime,
                ds.dst_node_ids,
                np.int32,
                reason="overlay half-edge target-node host export",
            )
            self._angle = _runtime_host_array(
                runtime,
                ds.angle,
                np.float64,
                reason="overlay half-edge angle host export",
            )
            self._sorted_edge_ids = _runtime_host_array(
                runtime,
                ds.sorted_edge_ids,
                np.int32,
                reason="overlay half-edge sorted-edge host export",
            )
            self._edge_positions = _runtime_host_array(
                runtime,
                ds.edge_positions,
                np.int32,
                reason="overlay half-edge edge-position host export",
            )
            return

        edge_count = int(self._src_x.size)
        edge_ids = np.arange(edge_count, dtype=np.int32)
        twin_edge_ids = edge_ids ^ np.int32(1)
        self._dst_x = self._src_x[twin_edge_ids]
        self._dst_y = self._src_y[twin_edge_ids]
        if self.isolate_rows:
            self._ensure_host_metadata()
            point_order = np.lexsort((self._src_y, self._src_x, self._row_indices))
        else:
            point_order = np.lexsort((self._src_y, self._src_x))
        sorted_x = self._src_x[point_order]
        sorted_y = self._src_y[point_order]
        point_start = np.empty(edge_count, dtype=bool)
        if edge_count:
            point_start[0] = True
            point_start[1:] = (sorted_x[1:] != sorted_x[:-1]) | (sorted_y[1:] != sorted_y[:-1])
            if self.isolate_rows:
                sorted_rows = self._row_indices[point_order]
                point_start[1:] |= sorted_rows[1:] != sorted_rows[:-1]
        point_node_ids = np.cumsum(point_start, dtype=np.int32) - 1
        self._src_node_ids = np.empty(edge_count, dtype=np.int32)
        self._src_node_ids[point_order] = point_node_ids
        self._dst_node_ids = self._src_node_ids[twin_edge_ids]
        self._node_x = sorted_x[point_start]
        self._node_y = sorted_y[point_start]
        self._angle = np.arctan2(
            self._dst_y - self._src_y,
            self._dst_x - self._src_x,
        )
        self._sorted_edge_ids = np.lexsort((edge_ids, self._angle, self._src_node_ids)).astype(
            np.int32, copy=False
        )
        self._edge_positions = np.empty(edge_count, dtype=np.int32)
        self._edge_positions[self._sorted_edge_ids] = edge_ids

    def _ensure_host_metadata(self) -> None:
        """Lazily copy per-edge metadata arrays from device to host on first access."""
        if self._source_segment_ids is not None:
            return
        ds = self.device_state
        if ds is None:
            return
        runtime = get_cuda_runtime()
        self._source_segment_ids = _runtime_host_array(
            runtime,
            ds.source_segment_ids,
            np.int32,
            reason="overlay half-edge source-segment host export",
        )
        self._source_side = _runtime_host_array(
            runtime,
            ds.source_side,
            np.int8,
            reason="overlay half-edge source-side host export",
        )
        self._row_indices = _runtime_host_array(
            runtime,
            ds.row_indices,
            np.int32,
            reason="overlay half-edge row-index host export",
        )
        self._part_indices = _runtime_host_array(
            runtime,
            ds.part_indices,
            np.int32,
            reason="overlay half-edge part-index host export",
        )
        self._ring_indices = _runtime_host_array(
            runtime,
            ds.ring_indices,
            np.int32,
            reason="overlay half-edge ring-index host export",
        )
        self._direction = _runtime_host_array(
            runtime,
            ds.direction,
            np.int8,
            reason="overlay half-edge direction host export",
        )

    @property
    def source_segment_ids(self) -> np.ndarray:
        if self._source_segment_ids is None:
            self._ensure_host_metadata()
        if self._source_segment_ids is None:
            return np.empty(0, dtype=np.int32)
        return self._source_segment_ids

    @property
    def source_side(self) -> np.ndarray:
        if self._source_side is None:
            self._ensure_host_metadata()
        if self._source_side is None:
            return np.empty(0, dtype=np.int8)
        return self._source_side

    @property
    def row_indices(self) -> np.ndarray:
        if self._row_indices is None:
            self._ensure_host_metadata()
        if self._row_indices is None:
            return np.empty(0, dtype=np.int32)
        return self._row_indices

    @property
    def part_indices(self) -> np.ndarray:
        if self._part_indices is None:
            self._ensure_host_metadata()
        if self._part_indices is None:
            return np.empty(0, dtype=np.int32)
        return self._part_indices

    @property
    def ring_indices(self) -> np.ndarray:
        if self._ring_indices is None:
            self._ensure_host_metadata()
        if self._ring_indices is None:
            return np.empty(0, dtype=np.int32)
        return self._ring_indices

    @property
    def direction(self) -> np.ndarray:
        if self._direction is None:
            self._ensure_host_metadata()
        if self._direction is None:
            return np.empty(0, dtype=np.int8)
        return self._direction

    @property
    def src_x(self) -> np.ndarray:
        self._ensure_host_topology()
        return self._src_x  # type: ignore[return-value]

    @property
    def src_y(self) -> np.ndarray:
        self._ensure_host_topology()
        return self._src_y  # type: ignore[return-value]

    @property
    def dst_x(self) -> np.ndarray:
        self._ensure_host_topology()
        return self._dst_x  # type: ignore[return-value]

    @property
    def dst_y(self) -> np.ndarray:
        self._ensure_host_topology()
        return self._dst_y  # type: ignore[return-value]

    @property
    def node_x(self) -> np.ndarray:
        self._ensure_host_topology()
        return self._node_x  # type: ignore[return-value]

    @property
    def node_y(self) -> np.ndarray:
        self._ensure_host_topology()
        return self._node_y  # type: ignore[return-value]

    @property
    def src_node_ids(self) -> np.ndarray:
        self._ensure_host_topology()
        return self._src_node_ids  # type: ignore[return-value]

    @property
    def dst_node_ids(self) -> np.ndarray:
        self._ensure_host_topology()
        return self._dst_node_ids  # type: ignore[return-value]

    @property
    def angle(self) -> np.ndarray:
        self._ensure_host_topology()
        return self._angle  # type: ignore[return-value]

    @property
    def sorted_edge_ids(self) -> np.ndarray:
        self._ensure_host_topology()
        return self._sorted_edge_ids  # type: ignore[return-value]

    @property
    def edge_positions(self) -> np.ndarray:
        self._ensure_host_topology()
        return self._edge_positions  # type: ignore[return-value]

    @property
    def next_edge_ids(self) -> np.ndarray:
        self._ensure_host_topology()
        return self._next_edge_ids  # type: ignore[return-value]

    @property
    def edge_count(self) -> int:
        return self._edge_count

    @property
    def node_count(self) -> int:
        if self._node_count > 0:
            return self._node_count
        if self._node_x is not None:
            return int(self._node_x.size)
        if self.device_state is not None and self.device_state.node_x is not None:
            return int(self.device_state.node_x.size)
        if self.device_state is not None:
            self._ensure_host_topology()
            if self._node_x is not None:
                return int(self._node_x.size)
        return 0


@dataclass(frozen=True)
class OverlayFaceDeviceState:
    face_offsets: DeviceArray
    face_edge_ids: DeviceArray
    bounded_mask: DeviceArray
    signed_area: DeviceArray
    centroid_x: DeviceArray
    centroid_y: DeviceArray
    left_covered: DeviceArray
    right_covered: DeviceArray


@dataclass
class OverlayFaceTable:
    runtime_selection: RuntimeSelection
    device_state: OverlayFaceDeviceState
    _face_count: int = 0
    _face_offsets: np.ndarray | None = None
    _face_edge_ids: np.ndarray | None = None
    _bounded_mask: np.ndarray | None = None
    _signed_area: np.ndarray | None = None
    _centroid_x: np.ndarray | None = None
    _centroid_y: np.ndarray | None = None
    _left_covered: np.ndarray | None = None
    _right_covered: np.ndarray | None = None

    def _ensure_host(self) -> None:
        if self._face_offsets is not None:
            return
        ds = self.device_state
        if ds is None:
            return
        runtime = get_cuda_runtime()
        self._face_offsets = _runtime_host_array(
            runtime,
            ds.face_offsets,
            np.int32,
            reason="overlay face-table face-offset host export",
        )
        self._face_edge_ids = _runtime_host_array(
            runtime,
            ds.face_edge_ids,
            np.int32,
            reason="overlay face-table face-edge host export",
        )
        self._bounded_mask = _runtime_host_array(
            runtime,
            ds.bounded_mask,
            np.int8,
            reason="overlay face-table bounded-mask host export",
        )
        self._signed_area = _runtime_host_array(
            runtime,
            ds.signed_area,
            np.float64,
            reason="overlay face-table signed-area host export",
        )
        self._centroid_x = _runtime_host_array(
            runtime,
            ds.centroid_x,
            np.float64,
            reason="overlay face-table centroid-x host export",
        )
        self._centroid_y = _runtime_host_array(
            runtime,
            ds.centroid_y,
            np.float64,
            reason="overlay face-table centroid-y host export",
        )
        self._left_covered = _runtime_host_array(
            runtime,
            ds.left_covered,
            np.int8,
            reason="overlay face-table left-coverage host export",
        )
        self._right_covered = _runtime_host_array(
            runtime,
            ds.right_covered,
            np.int8,
            reason="overlay face-table right-coverage host export",
        )

    @property
    def face_offsets(self) -> np.ndarray:
        self._ensure_host()
        return self._face_offsets  # type: ignore[return-value]

    @property
    def face_edge_ids(self) -> np.ndarray:
        self._ensure_host()
        return self._face_edge_ids  # type: ignore[return-value]

    @property
    def bounded_mask(self) -> np.ndarray:
        self._ensure_host()
        return self._bounded_mask  # type: ignore[return-value]

    @property
    def signed_area(self) -> np.ndarray:
        self._ensure_host()
        return self._signed_area  # type: ignore[return-value]

    @property
    def centroid_x(self) -> np.ndarray:
        self._ensure_host()
        return self._centroid_x  # type: ignore[return-value]

    @property
    def centroid_y(self) -> np.ndarray:
        self._ensure_host()
        return self._centroid_y  # type: ignore[return-value]

    @property
    def left_covered(self) -> np.ndarray:
        if self._left_covered is None:
            self._ensure_host()
        if (
            self._left_covered is None
            and self.device_state is not None
            and self.device_state.left_covered is not None
        ):
            runtime = get_cuda_runtime()
            self._left_covered = _runtime_host_array(
                runtime,
                self.device_state.left_covered,
                np.int8,
                reason="overlay face-table left-coverage host export",
            )
        return self._left_covered  # type: ignore[return-value]

    @property
    def right_covered(self) -> np.ndarray:
        if self._right_covered is None:
            self._ensure_host()
        if (
            self._right_covered is None
            and self.device_state is not None
            and self.device_state.right_covered is not None
        ):
            runtime = get_cuda_runtime()
            self._right_covered = _runtime_host_array(
                runtime,
                self.device_state.right_covered,
                np.int8,
                reason="overlay face-table right-coverage host export",
            )
        return self._right_covered  # type: ignore[return-value]

    @property
    def face_count(self) -> int:
        return self._face_count


@dataclass(frozen=True)
class OverlayExecutionPlan:
    """Reusable overlay topology plan for one left/right workload."""

    split_events: SplitEventTable | None
    atomic_edges: AtomicEdgeTable | None
    half_edge_graph: HalfEdgeGraph
    faces: OverlayFaceTable
    row_isolated: bool = False


@dataclass(frozen=True)
class ComponentOverlayExecutionPlan:
    """One oversized logical row decomposed into disjoint topology rows.

    ``left`` and ``right`` contain aligned synthetic MultiPolygon rows. Their
    combined polygon-part x intervals are strictly separated between rows, so
    each row owns an independent face graph and the results can be packed back
    into one geometry without another constructive union.
    """

    left: object
    right: object
    component_count: int
    max_left_segments_per_component: int
    max_right_segments_per_component: int
    dispatch_mode: object
    include_same_side_splits: bool = False
    row_isolated: bool = True


@dataclass(frozen=True)
class MicrocellOverlayExecutionPlan:
    """One connected oversized row reconstructed from paged microcell bands."""

    left: object
    right: object
    max_left_segments: int
    max_right_segments: int
    dispatch_mode: object
    row_isolated: bool = True


@dataclass(frozen=True)
class PagedOverlayExecutionPlan:
    """Independent row-isolated topology plans bounded by live-event work.

    Every page owns complete logical rows, source-segment runs, and face graphs.
    Page boundaries are algebraic from ``rows_per_page``; no device row/event
    metadata is exported and no row-shaped host offset table is retained.
    """

    left: object
    right: object
    row_count: int
    rows_per_page: int
    max_left_segments_per_row: int
    max_right_segments_per_row: int
    dispatch_mode: object
    use_same_row_fast_path: bool | None = None
    include_same_side_splits: bool = False
    right_geometry_source_rows: DeviceArray | np.ndarray | None = None
    right_segment_source_rows: DeviceArray | np.ndarray | None = None
    right_segment_broadcast: object | None = None
    allow_component_decomposition: bool = True
    row_isolated: bool = True

    @property
    def page_count(self) -> int:
        if self.row_count == 0:
            return 0
        return (self.row_count + self.rows_per_page - 1) // self.rows_per_page

    def row_span(self, page_index: int) -> tuple[int, int]:
        if page_index < 0 or page_index >= self.page_count:
            raise IndexError(f"overlay topology page index out of range: {page_index}")
        start = page_index * self.rows_per_page
        return start, min(start + self.rows_per_page, self.row_count)
