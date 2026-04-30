"""GPU WKB decode kernel pipeline (P1a).

Implements a 5-stage GPU pipeline that reads WKB byte streams directly
on device and produces OwnedGeometryArray results without host round-trips.

Stages:
  1. Header scan kernel -- 1 thread per WKB record, reads endian + type tag
  2. Family partition -- CuPy boolean masks to bucket records by geometry type
  3. Size scan + offset computation -- sizing kernels + CCCL exclusive_sum
  4. Family decode kernels -- per-family coordinate extraction
  5. Assembly -- build OwnedGeometryArray via io_pylibcudf helpers

ADR-0033 dispatch tiers:
  Tier 1 (custom NVRTC) for geometry-specific decode
  Tier 3a (CCCL)        for prefix-sum offset computation
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
    make_kernel_cache_key,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.cuda.nvrtc_precompile import (
    request_nvrtc_warmup as _request_nvrtc_warmup,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.io.pylibcudf import (
    _build_device_mixed_owned,
    _build_device_single_family_owned,
)
from vibespatial.kernels.core.wkb_decode_source import (
    _WKB_DECODE_KERNEL_NAMES,
    _WKB_DECODE_KERNEL_SOURCE,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CCCL warmup (ADR-0034)
# ---------------------------------------------------------------------------
request_warmup(["exclusive_scan_i32", "exclusive_scan_i64"])

# ---------------------------------------------------------------------------
# Shared CUDA device helpers for WKB byte reading
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Kernel compilation (matches io_wkb._wkb_encode_kernels pattern)
# ---------------------------------------------------------------------------
_request_nvrtc_warmup([
    ("wkb-decode", _WKB_DECODE_KERNEL_SOURCE, _WKB_DECODE_KERNEL_NAMES),
])


def _wkb_decode_kernels() -> dict[str, Any]:
    runtime = get_cuda_runtime()
    return runtime.compile_kernels(
        cache_key=make_kernel_cache_key("wkb-decode", _WKB_DECODE_KERNEL_SOURCE),
        source=_WKB_DECODE_KERNEL_SOURCE,
        kernel_names=_WKB_DECODE_KERNEL_NAMES,
    )


# ---------------------------------------------------------------------------
# Family tag -> GeometryFamily mapping (matches FAMILY_TAGS ordering)
# ---------------------------------------------------------------------------

_TAG_TO_FAMILY = {
    0: GeometryFamily.POINT,
    1: GeometryFamily.LINESTRING,
    2: GeometryFamily.POLYGON,
    3: GeometryFamily.MULTIPOINT,
    4: GeometryFamily.MULTILINESTRING,
    5: GeometryFamily.MULTIPOLYGON,
}

# Reverse: GeometryFamily -> kernel family tag (int8)
_FAMILY_TO_TAG = {v: k for k, v in _TAG_TO_FAMILY.items()}


def _wkb_decode_device_to_host(device_array: object, *, reason: str) -> np.ndarray:
    """Copy WKB decode metadata through a named runtime D2H boundary."""
    return np.asarray(get_cuda_runtime().copy_device_to_host(device_array, reason=reason))


def _wkb_decode_size_summary(*values: object, reason: str) -> np.ndarray:
    """Return host allocation sizes from device scalar expressions."""
    import cupy as cp

    summary = cp.empty(len(values), dtype=cp.int64)
    for index, value in enumerate(values):
        summary[index] = value
    return _wkb_decode_device_to_host(summary, reason=reason).reshape(-1)


def _wkb_decode_bool_scalar(value: object, *, reason: str) -> bool:
    import cupy as cp

    host = _wkb_decode_device_to_host(cp.asarray(value, dtype=cp.bool_).reshape(1), reason=reason)
    return bool(host.reshape(-1)[0])


# ---------------------------------------------------------------------------
# Stage 1: Header scan
# ---------------------------------------------------------------------------

def _stage1_header_scan(
    payload_device,
    record_offsets_device,
    record_count: int,
) -> tuple[Any, Any, Any]:
    """Run the header scan kernel. Returns (family_tags, is_native, primary_counts)."""
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()
    kernel = kernels["wkb_header_scan"]

    family_tags = cp.full(record_count, -1, dtype=cp.int8)
    is_native = cp.zeros(record_count, dtype=cp.uint8)
    primary_counts = cp.zeros(record_count, dtype=cp.int32)

    if record_count == 0:
        return family_tags, is_native, primary_counts

    ptr = runtime.pointer
    grid, block = runtime.launch_config(kernel, record_count)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(
            (ptr(payload_device), ptr(record_offsets_device),
             ptr(family_tags), ptr(is_native), ptr(primary_counts),
             record_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_I32),
        ),
    )

    return family_tags, is_native, primary_counts


# ---------------------------------------------------------------------------
# Stage 2: Family partition
# ---------------------------------------------------------------------------

def _stage2_partition(
    family_tags,
    is_native,
    record_count: int,
) -> dict[GeometryFamily, Any]:
    """Partition row indices by geometry family using CuPy fancy indexing."""
    import cupy as cp

    partitions: dict[GeometryFamily, Any] = {}
    for tag, family in _TAG_TO_FAMILY.items():
        mask = (family_tags == cp.int8(tag)) & (is_native == cp.uint8(1))
        row_indexes = cp.flatnonzero(mask).astype(cp.int32, copy=False)
        if int(row_indexes.size) > 0:
            partitions[family] = row_indexes

    return partitions


# ---------------------------------------------------------------------------
# Stage 3 + 4: Per-family decode
# ---------------------------------------------------------------------------

def _decode_point_family(
    payload_device,
    record_offsets_device,
    row_indexes,
) -> DeviceFamilyGeometryBuffer:
    """Decode point records on GPU."""
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()
    kernel = kernels["decode_point_wkb"]

    n = int(row_indexes.size)
    x_out = cp.empty(n, dtype=cp.float64)
    y_out = cp.empty(n, dtype=cp.float64)
    empty_out = cp.zeros(n, dtype=cp.uint8)

    if n > 0:
        ptr = runtime.pointer
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(
            kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(x_out), ptr(y_out), ptr(empty_out), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )

    empty_mask = empty_out.astype(cp.bool_, copy=False)
    nonempty = ~empty_mask
    nonempty_counts = nonempty.astype(cp.int32, copy=False)
    geometry_offsets = cp.empty(n + 1, dtype=cp.int32)
    geometry_offsets[0] = 0
    if n > 0:
        offsets_excl = exclusive_sum(nonempty_counts, synchronize=False)
        geometry_offsets[1:] = offsets_excl + nonempty_counts

    x_valid = x_out[nonempty]
    y_valid = y_out[nonempty]

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        x=x_valid,
        y=y_valid,
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        bounds=None,
    )


def _decode_linestring_family(
    payload_device,
    record_offsets_device,
    row_indexes,
    primary_counts,
) -> DeviceFamilyGeometryBuffer:
    """Decode linestring records on GPU."""
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()

    n = int(row_indexes.size)
    point_counts = primary_counts[row_indexes].astype(cp.int32, copy=False)

    # Build geometry offsets via CCCL exclusive_sum (ADR-0033)
    coord_offsets = exclusive_sum(point_counts, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)
    geometry_offsets = cp.empty(n + 1, dtype=cp.int32)
    geometry_offsets[0] = 0
    if n > 0:
        geometry_offsets[1:] = coord_offsets + point_counts

    size_summary = (
        _wkb_decode_size_summary(
            geometry_offsets[-1],
            reason="WKB linestring decode coord-count scalar fence",
        )
        if n > 0
        else np.zeros(1, dtype=np.int64)
    )
    total_coords = int(size_summary[0])

    x_out = cp.empty(total_coords, dtype=cp.float64)
    y_out = cp.empty(total_coords, dtype=cp.float64)

    if n > 0 and total_coords > 0:
        kernel = kernels["decode_linestring_wkb"]
        ptr = runtime.pointer
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(
            kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(coord_offsets),
                 ptr(x_out), ptr(y_out), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.LINESTRING,
        x=x_out,
        y=y_out,
        geometry_offsets=geometry_offsets,
        empty_mask=point_counts == 0,
        bounds=None,
    )


def _decode_polygon_family(
    payload_device,
    record_offsets_device,
    row_indexes,
) -> DeviceFamilyGeometryBuffer:
    """Decode polygon records on GPU using sizing kernel + decode kernel."""
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()

    n = int(row_indexes.size)

    # Stage 3: size scan
    total_rings_per = cp.zeros(n, dtype=cp.int32)
    total_coords_per = cp.zeros(n, dtype=cp.int32)

    if n > 0:
        size_kernel = kernels["wkb_polygon_size_scan"]
        ptr = runtime.pointer
        grid, block = runtime.launch_config(size_kernel, n)
        runtime.launch(
            size_kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(total_rings_per), ptr(total_coords_per), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
            ),
        )

    # Geometry offsets = prefix sum of ring counts (rings per polygon, ADR-0033)
    ring_count_offsets = exclusive_sum(total_rings_per, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)
    geometry_offsets = cp.empty(n + 1, dtype=cp.int32)
    geometry_offsets[0] = 0
    if n > 0:
        geometry_offsets[1:] = ring_count_offsets + total_rings_per
    coord_offsets = exclusive_sum(total_coords_per, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)
    size_summary = (
        _wkb_decode_size_summary(
            geometry_offsets[-1],
            coord_offsets[-1] + total_coords_per[-1],
            reason="WKB polygon decode size summary scalar fence",
        )
        if n > 0
        else np.zeros(2, dtype=np.int64)
    )
    total_rings = int(size_summary[0])
    total_coords = int(size_summary[1])
    dense_single_ring_width = None
    if n > 0 and total_coords > 0 and total_coords % n == 0:
        candidate_width = total_coords // n
        dense = cp.all(total_rings_per == 1) & cp.all(total_coords_per == candidate_width)
        if _wkb_decode_bool_scalar(
            dense,
            reason="WKB polygon dense single-ring scalar fence",
        ):
            dense_single_ring_width = int(candidate_width)

    # Allocate output
    ring_offsets_out = cp.empty(total_rings + 1, dtype=cp.int32)
    x_out = cp.empty(total_coords, dtype=cp.float64)
    y_out = cp.empty(total_coords, dtype=cp.float64)

    if n > 0 and total_coords > 0:
        decode_kernel = kernels["decode_polygon_wkb"]
        ptr = runtime.pointer
        grid, block = runtime.launch_config(decode_kernel, n)
        runtime.launch(
            decode_kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(ring_count_offsets), ptr(coord_offsets),
                 ptr(ring_offsets_out), ptr(x_out), ptr(y_out), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )

    # Set the sentinel ring offset
    if total_rings > 0:
        ring_offsets_out[total_rings] = total_coords
    elif total_rings == 0 and n > 0:
        ring_offsets_out[0] = 0

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        x=x_out,
        y=y_out,
        geometry_offsets=geometry_offsets,
        empty_mask=total_rings_per == 0,
        ring_offsets=ring_offsets_out,
        bounds=None,
        dense_single_ring_width=dense_single_ring_width,
    )


def _decode_multipoint_family(
    payload_device,
    record_offsets_device,
    row_indexes,
    primary_counts,
) -> DeviceFamilyGeometryBuffer:
    """Decode multipoint records on GPU."""
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()

    n = int(row_indexes.size)
    part_counts = primary_counts[row_indexes].astype(cp.int32, copy=False)

    coord_offsets = exclusive_sum(part_counts, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)
    geometry_offsets = cp.empty(n + 1, dtype=cp.int32)
    geometry_offsets[0] = 0
    if n > 0:
        geometry_offsets[1:] = coord_offsets + part_counts
    size_summary = (
        _wkb_decode_size_summary(
            geometry_offsets[-1],
            reason="WKB multipoint decode coord-count scalar fence",
        )
        if n > 0
        else np.zeros(1, dtype=np.int64)
    )
    total_coords = int(size_summary[0])

    x_out = cp.empty(total_coords, dtype=cp.float64)
    y_out = cp.empty(total_coords, dtype=cp.float64)

    if n > 0 and total_coords > 0:
        kernel = kernels["decode_multipoint_wkb"]
        ptr = runtime.pointer
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(
            kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(coord_offsets),
                 ptr(x_out), ptr(y_out), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOINT,
        x=x_out,
        y=y_out,
        geometry_offsets=geometry_offsets,
        empty_mask=part_counts == 0,
        bounds=None,
    )


def _decode_multilinestring_family(
    payload_device,
    record_offsets_device,
    row_indexes,
) -> DeviceFamilyGeometryBuffer:
    """Decode multilinestring records on GPU using sizing kernel + decode kernel."""
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()

    n = int(row_indexes.size)

    # Stage 3: size scan
    total_parts_per = cp.zeros(n, dtype=cp.int32)
    total_coords_per = cp.zeros(n, dtype=cp.int32)

    if n > 0:
        size_kernel = kernels["wkb_multilinestring_size_scan"]
        ptr = runtime.pointer
        grid, block = runtime.launch_config(size_kernel, n)
        runtime.launch(
            size_kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(total_parts_per), ptr(total_coords_per), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
            ),
        )

    # Geometry offsets = prefix sum of part counts (ADR-0033)
    part_count_offsets = exclusive_sum(total_parts_per, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)
    geometry_offsets = cp.empty(n + 1, dtype=cp.int32)
    geometry_offsets[0] = 0
    if n > 0:
        geometry_offsets[1:] = part_count_offsets + total_parts_per
    coord_offsets = exclusive_sum(total_coords_per, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)
    size_summary = (
        _wkb_decode_size_summary(
            geometry_offsets[-1],
            coord_offsets[-1] + total_coords_per[-1],
            reason="WKB multilinestring decode size summary scalar fence",
        )
        if n > 0
        else np.zeros(2, dtype=np.int64)
    )
    total_parts = int(size_summary[0])
    total_coords = int(size_summary[1])

    part_offsets_out = cp.empty(total_parts + 1, dtype=cp.int32)
    x_out = cp.empty(total_coords, dtype=cp.float64)
    y_out = cp.empty(total_coords, dtype=cp.float64)

    if n > 0 and total_coords > 0:
        decode_kernel = kernels["decode_multilinestring_wkb"]
        ptr = runtime.pointer
        grid, block = runtime.launch_config(decode_kernel, n)
        runtime.launch(
            decode_kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(part_count_offsets), ptr(coord_offsets),
                 ptr(part_offsets_out), ptr(x_out), ptr(y_out), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )

    # Set sentinel
    if total_parts > 0:
        part_offsets_out[total_parts] = total_coords
    elif total_parts == 0 and n > 0:
        part_offsets_out[0] = 0

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTILINESTRING,
        x=x_out,
        y=y_out,
        geometry_offsets=geometry_offsets,
        empty_mask=total_parts_per == 0,
        part_offsets=part_offsets_out,
        bounds=None,
    )


def _decode_multipolygon_family(
    payload_device,
    record_offsets_device,
    row_indexes,
) -> DeviceFamilyGeometryBuffer:
    """Decode multipolygon records on GPU using sizing kernel + decode kernel."""
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()

    n = int(row_indexes.size)

    # Stage 3: size scan
    total_parts_per = cp.zeros(n, dtype=cp.int32)
    total_rings_per = cp.zeros(n, dtype=cp.int32)
    total_coords_per = cp.zeros(n, dtype=cp.int32)

    if n > 0:
        size_kernel = kernels["wkb_multipolygon_size_scan"]
        ptr = runtime.pointer
        grid, block = runtime.launch_config(size_kernel, n)
        runtime.launch(
            size_kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(total_parts_per),
                 ptr(total_rings_per), ptr(total_coords_per), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )

    # Geometry offsets = prefix sum of part counts (polygon parts per multipolygon, ADR-0033)
    poly_count_offsets = exclusive_sum(total_parts_per, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)
    geometry_offsets = cp.empty(n + 1, dtype=cp.int32)
    geometry_offsets[0] = 0
    if n > 0:
        geometry_offsets[1:] = poly_count_offsets + total_parts_per
    ring_count_offsets = exclusive_sum(total_rings_per, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)
    coord_offsets = exclusive_sum(total_coords_per, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)

    size_summary = (
        _wkb_decode_size_summary(
            geometry_offsets[-1],
            ring_count_offsets[-1] + total_rings_per[-1],
            coord_offsets[-1] + total_coords_per[-1],
            reason="WKB multipolygon decode size summary scalar fence",
        )
        if n > 0
        else np.zeros(3, dtype=np.int64)
    )
    total_parts = int(size_summary[0])
    total_rings = int(size_summary[1])
    total_coords = int(size_summary[2])

    part_offsets_out = cp.empty(total_parts + 1, dtype=cp.int32)
    ring_offsets_out = cp.empty(total_rings + 1, dtype=cp.int32)
    x_out = cp.empty(total_coords, dtype=cp.float64)
    y_out = cp.empty(total_coords, dtype=cp.float64)

    if n > 0 and total_coords > 0:
        decode_kernel = kernels["decode_multipolygon_wkb"]
        ptr = runtime.pointer
        grid, block = runtime.launch_config(decode_kernel, n)
        runtime.launch(
            decode_kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(poly_count_offsets),
                 ptr(ring_count_offsets), ptr(coord_offsets),
                 ptr(part_offsets_out), ptr(ring_offsets_out),
                 ptr(x_out), ptr(y_out), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )

    # Set sentinels
    if total_parts > 0:
        part_offsets_out[total_parts] = total_rings
    elif total_parts == 0 and n > 0:
        part_offsets_out[0] = 0

    if total_rings > 0:
        ring_offsets_out[total_rings] = total_coords
    elif total_rings == 0 and n > 0:
        ring_offsets_out[0] = 0

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOLYGON,
        x=x_out,
        y=y_out,
        geometry_offsets=geometry_offsets,
        empty_mask=total_parts_per == 0,
        part_offsets=part_offsets_out,
        ring_offsets=ring_offsets_out,
        bounds=None,
    )


# ---------------------------------------------------------------------------
# Stage 5: Assembly
# ---------------------------------------------------------------------------

def _assemble_single_family(
    family: GeometryFamily,
    family_buffer: DeviceFamilyGeometryBuffer,
    validity_device,
    record_count: int,
) -> OwnedGeometryArray:
    """Assemble single-family result using io_pylibcudf helper."""
    return _build_device_single_family_owned(
        family=family,
        validity_device=validity_device,
        x_device=family_buffer.x,
        y_device=family_buffer.y,
        geometry_offsets_device=family_buffer.geometry_offsets,
        empty_mask_device=family_buffer.empty_mask,
        part_offsets_device=family_buffer.part_offsets,
        ring_offsets_device=family_buffer.ring_offsets,
        dense_single_ring_width=family_buffer.dense_single_ring_width,
        detail="created device-resident owned geometry array from GPU WKB decode kernel pipeline",
    )


def _assemble_mixed(
    partitions: dict[GeometryFamily, Any],
    family_buffers: dict[GeometryFamily, DeviceFamilyGeometryBuffer],
    family_tags,
    is_native,
    record_count: int,
) -> OwnedGeometryArray:
    """Assemble mixed-family result using io_pylibcudf helper."""
    import cupy as cp

    valid_mask = is_native.astype(cp.bool_, copy=False)
    tags_device = cp.where(
        valid_mask,
        family_tags.astype(cp.int8, copy=False),
        cp.int8(-1),
    ).astype(cp.int8, copy=False)

    family_row_offsets_device = cp.full(record_count, -1, dtype=cp.int32)
    for family, row_indexes in partitions.items():
        n_rows = int(row_indexes.size)
        if n_rows > 0:
            family_row_offsets_device[row_indexes] = cp.arange(n_rows, dtype=cp.int32)

    family_devices: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}
    for family in family_buffers:
        family_devices[family] = family_buffers[family]

    return _build_device_mixed_owned(
        validity_device=valid_mask,
        tags_device=tags_device,
        family_row_offsets_device=family_row_offsets_device,
        family_devices=family_devices,
        detail="created device-resident owned geometry array from GPU WKB decode kernel pipeline (mixed)",
    )


# ---------------------------------------------------------------------------
# Public API: decode_wkb_device_pipeline
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "decode_wkb",
    "gpu-cuda-python",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=tuple(family.value for family in GeometryFamily),
    tags=("wkb", "decode", "nvrtc", "cccl"),
)
def decode_wkb_device_pipeline(
    payload_device,
    record_offsets_device,
    record_count: int,
) -> OwnedGeometryArray:
    """GPU WKB decode pipeline.

    Reads WKB byte streams directly on GPU using custom CUDA kernels.
    Returns an OwnedGeometryArray with device-resident geometry buffers.

    Parameters
    ----------
    payload_device : device array (uint8)
        Contiguous WKB byte payload on device.
    record_offsets_device : device array (int32)
        Byte offsets for each record (length = record_count + 1).
    record_count : int
        Number of WKB records.

    Returns
    -------
    OwnedGeometryArray
        Device-resident geometry array.
    """
    import cupy as cp

    if record_count == 0:
        validity = cp.zeros(0, dtype=cp.bool_)
        return _build_device_single_family_owned(
            family=GeometryFamily.POINT,
            validity_device=validity,
            x_device=cp.empty(0, dtype=cp.float64),
            y_device=cp.empty(0, dtype=cp.float64),
            geometry_offsets_device=cp.zeros(1, dtype=cp.int32),
            empty_mask_device=cp.zeros(0, dtype=cp.bool_),
            detail="created empty device-resident owned geometry array from GPU WKB decode pipeline",
        )

    # Stage 1: Header scan
    family_tags, is_native, primary_counts = _stage1_header_scan(
        payload_device, record_offsets_device, record_count,
    )

    # Stage 2: Family partition
    partitions = _stage2_partition(family_tags, is_native, record_count)

    if not partitions:
        # No supported records found -- return all-null result
        validity = cp.zeros(record_count, dtype=cp.bool_)
        return _build_device_single_family_owned(
            family=GeometryFamily.POINT,
            validity_device=validity,
            x_device=cp.empty(0, dtype=cp.float64),
            y_device=cp.empty(0, dtype=cp.float64),
            geometry_offsets_device=cp.zeros(1, dtype=cp.int32),
            empty_mask_device=cp.zeros(0, dtype=cp.bool_),
            detail="created all-null device-resident owned geometry array from GPU WKB decode pipeline",
        )

    # Stages 3+4: Per-family decode
    family_buffers: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}

    for family, row_indexes in partitions.items():
        if family is GeometryFamily.POINT:
            family_buffers[family] = _decode_point_family(
                payload_device, record_offsets_device, row_indexes,
            )
        elif family is GeometryFamily.LINESTRING:
            family_buffers[family] = _decode_linestring_family(
                payload_device, record_offsets_device, row_indexes, primary_counts,
            )
        elif family is GeometryFamily.POLYGON:
            family_buffers[family] = _decode_polygon_family(
                payload_device, record_offsets_device, row_indexes,
            )
        elif family is GeometryFamily.MULTIPOINT:
            family_buffers[family] = _decode_multipoint_family(
                payload_device, record_offsets_device, row_indexes, primary_counts,
            )
        elif family is GeometryFamily.MULTILINESTRING:
            family_buffers[family] = _decode_multilinestring_family(
                payload_device, record_offsets_device, row_indexes,
            )
        elif family is GeometryFamily.MULTIPOLYGON:
            family_buffers[family] = _decode_multipolygon_family(
                payload_device, record_offsets_device, row_indexes,
            )

    # Stage 5: Assembly
    families_present = list(family_buffers.keys())
    if len(families_present) == 1:
        single_family = families_present[0]
        single_row_indexes = partitions[single_family]
        n_supported = int(single_row_indexes.size)
        # Check if ALL records are this family (no unsupported/fallback)
        if n_supported == record_count:
            return _assemble_single_family(
                single_family,
                family_buffers[single_family],
                is_native.astype(cp.bool_, copy=False),
                record_count,
            )

    return _assemble_mixed(
        partitions, family_buffers, family_tags, is_native, record_count,
    )
