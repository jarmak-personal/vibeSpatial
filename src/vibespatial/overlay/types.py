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


@dataclass(frozen=True)
class SplitEventDeviceState:
    source_segment_ids: DeviceArray
    packed_keys: DeviceArray
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
        self._source_segment_ids = np.asarray(
            runtime.copy_device_to_host(ds.source_segment_ids), dtype=np.int32,
        )
        self._t = np.asarray(
            runtime.copy_device_to_host(ds.t), dtype=np.float64,
        )
        self._x = np.asarray(
            runtime.copy_device_to_host(ds.x), dtype=np.float64,
        )
        self._y = np.asarray(
            runtime.copy_device_to_host(ds.y), dtype=np.float64,
        )
        if ds.source_side is not None:
            self._source_side = np.asarray(
                runtime.copy_device_to_host(ds.source_side), dtype=np.int8,
            )
        else:
            # Derive from source_segment_ids + left_segment_count
            ids = self._source_segment_ids
            self._source_side = np.where(ids < self.left_segment_count, 1, 2).astype(np.int8)
        if ds.row_indices is not None:
            self._row_indices = np.asarray(
                runtime.copy_device_to_host(ds.row_indices), dtype=np.int32,
            )
        else:
            self._row_indices = np.empty(0, dtype=np.int32)
        if ds.part_indices is not None:
            self._part_indices = np.asarray(
                runtime.copy_device_to_host(ds.part_indices), dtype=np.int32,
            )
        else:
            self._part_indices = np.empty(0, dtype=np.int32)
        if ds.ring_indices is not None:
            self._ring_indices = np.asarray(
                runtime.copy_device_to_host(ds.ring_indices), dtype=np.int32,
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
        self._source_segment_ids = np.asarray(
            runtime.copy_device_to_host(ds.source_segment_ids), dtype=np.int32,
        )
        self._direction = np.asarray(
            runtime.copy_device_to_host(ds.direction), dtype=np.int8,
        )
        self._src_x = np.asarray(runtime.copy_device_to_host(ds.src_x), dtype=np.float64)
        self._src_y = np.asarray(runtime.copy_device_to_host(ds.src_y), dtype=np.float64)
        self._dst_x = np.asarray(runtime.copy_device_to_host(ds.dst_x), dtype=np.float64)
        self._dst_y = np.asarray(runtime.copy_device_to_host(ds.dst_y), dtype=np.float64)

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
                self._source_side = np.asarray(
                    runtime.copy_device_to_host(ds.source_side), dtype=np.int8,
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
                self._row_indices = np.asarray(
                    runtime.copy_device_to_host(ds.row_indices), dtype=np.int32,
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
                self._part_indices = np.asarray(
                    runtime.copy_device_to_host(ds.part_indices), dtype=np.int32,
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
                self._ring_indices = np.asarray(
                    runtime.copy_device_to_host(ds.ring_indices), dtype=np.int32,
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
    node_x: DeviceArray
    node_y: DeviceArray
    src_node_ids: DeviceArray
    dst_node_ids: DeviceArray
    angle: DeviceArray
    sorted_edge_ids: DeviceArray
    edge_positions: DeviceArray
    next_edge_ids: DeviceArray
    src_x: DeviceArray
    src_y: DeviceArray
    # Metadata arrays — carried from AtomicEdgeTable to avoid D->H
    # round-trips when GPU consumers need per-edge source metadata.
    source_segment_ids: DeviceArray | None = None
    source_side: DeviceArray | None = None
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
        def _to_host(arr):
            return np.asarray(runtime.copy_device_to_host(arr), dtype=np.float64) if arr is not None else None

        def _to_host_i32(arr):
            return np.asarray(runtime.copy_device_to_host(arr), dtype=np.int32) if arr is not None else None
        self._src_x = _to_host(ds.src_x)
        self._src_y = _to_host(ds.src_y)
        self._dst_x = _to_host(getattr(ds, 'dst_x', None))
        self._dst_y = _to_host(getattr(ds, 'dst_y', None))
        self._node_x = _to_host(ds.node_x)
        self._node_y = _to_host(ds.node_y)
        self._src_node_ids = _to_host_i32(ds.src_node_ids)
        self._dst_node_ids = _to_host_i32(ds.dst_node_ids)
        self._angle = _to_host(ds.angle)
        self._sorted_edge_ids = _to_host_i32(ds.sorted_edge_ids)
        self._edge_positions = _to_host_i32(ds.edge_positions)
        self._next_edge_ids = _to_host_i32(ds.next_edge_ids)

    def _ensure_host_metadata(self) -> None:
        """Lazily copy per-edge metadata arrays from device to host on first access."""
        if self._source_segment_ids is not None:
            return
        ds = self.device_state
        if ds is None:
            return
        runtime = get_cuda_runtime()
        def _to_host_i32(arr):
            return np.asarray(runtime.copy_device_to_host(arr), dtype=np.int32) if arr is not None else None
        def _to_host_i8(arr):
            return np.asarray(runtime.copy_device_to_host(arr), dtype=np.int8) if arr is not None else None
        self._source_segment_ids = _to_host_i32(ds.source_segment_ids)
        self._source_side = _to_host_i8(ds.source_side)
        self._row_indices = _to_host_i32(ds.row_indices)
        self._part_indices = _to_host_i32(ds.part_indices)
        self._ring_indices = _to_host_i32(ds.ring_indices)
        self._direction = _to_host_i8(ds.direction)

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
        if self._node_x is not None:
            return int(self._node_x.size)
        if self.device_state is not None and self.device_state.node_x is not None:
            return int(self.device_state.node_x.size)
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
        def _h(arr, dt):
            return np.asarray(runtime.copy_device_to_host(arr), dtype=dt) if arr is not None else None
        self._face_offsets = _h(ds.face_offsets, np.int32)
        self._face_edge_ids = _h(ds.face_edge_ids, np.int32)
        self._bounded_mask = _h(ds.bounded_mask, np.int8)
        self._signed_area = _h(ds.signed_area, np.float64)
        self._centroid_x = _h(ds.centroid_x, np.float64)
        self._centroid_y = _h(ds.centroid_y, np.float64)
        self._left_covered = _h(ds.left_covered, np.int8)
        self._right_covered = _h(ds.right_covered, np.int8)

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
        if self._left_covered is None and self.device_state is not None and self.device_state.left_covered is not None:
            runtime = get_cuda_runtime()
            self._left_covered = np.asarray(runtime.copy_device_to_host(self.device_state.left_covered), dtype=np.int8)
        return self._left_covered  # type: ignore[return-value]

    @property
    def right_covered(self) -> np.ndarray:
        if self._right_covered is None:
            self._ensure_host()
        if self._right_covered is None and self.device_state is not None and self.device_state.right_covered is not None:
            runtime = get_cuda_runtime()
            self._right_covered = np.asarray(runtime.copy_device_to_host(self.device_state.right_covered), dtype=np.int8)
        return self._right_covered  # type: ignore[return-value]

    @property
    def face_count(self) -> int:
        return self._face_count
