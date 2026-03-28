"""GPU-accelerated reader for OpenStreetMap PBF files.

Extracts DenseNodes from OSM PBF files using a hybrid CPU/GPU pipeline:

1. **Block index parsing** (CPU) -- parse BlobHeader/Blob pairs sequentially
   to build a block index with offsets and types.
2. **Zlib decompression** (CPU) -- decompress each OSMData block using
   Python's zlib module.
3. **Protobuf field extraction** (CPU) -- locate DenseNodes fields (id,
   lat, lon deltas) within each PrimitiveBlock, producing varint byte
   ranges.
4. **Varint decoding** (GPU, Tier 1 NVRTC) -- parallel decode of protobuf
   varints to int64 arrays.  Each thread decodes one varint at a known
   position, handling the 7-bit-per-byte continuation encoding.
5. **Delta decoding** (GPU, Tier 2 CuPy) -- cumulative sum over
   delta-encoded arrays to recover absolute node IDs, latitudes, and
   longitudes.
6. **Coordinate scaling** (GPU, Tier 2 CuPy) -- convert from nanodegree
   integer representation to fp64 degrees.
7. **Assembly** -- build device-resident Point OwnedGeometryArray.

Tier classification (ADR-0033):
    - Block parsing: CPU (sequential metadata, not parallelizable)
    - Decompression: CPU (zlib)
    - Protobuf field location: CPU (sequential proto traversal, small data)
    - Varint decoding: Tier 1 (custom NVRTC -- binary format-specific)
    - Delta decode (cumsum): Tier 2 (CuPy cumsum)
    - Coordinate scaling: Tier 2 (CuPy element-wise)
    - Offset construction: Tier 2 (CuPy)

Precision (ADR-0002):
    The varint decode kernel is integer-only -- no floating-point
    computation, so no PrecisionPlan is needed.  Coordinate storage
    is always fp64 per ADR-0002 (same rationale as csv_gpu.py,
    kml_gpu.py, and all other IO readers).

Way extraction (Phase 2) is deferred -- DenseNodes contain ~99% of all
OSM nodes and are the primary parallelizable target.
"""
from __future__ import annotations

import ctypes
import logging
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_I64,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    DiagnosticKind,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    OwnedGeometryDeviceState,
)
from vibespatial.runtime.residency import Residency

if TYPE_CHECKING:
    import cupy as cp

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NVRTC kernel source (Tier 1) -- integer-only varint decoding
# ---------------------------------------------------------------------------

_VARINT_DECODE_SOURCE = r"""
// Protobuf varint decoder -- each thread decodes one varint at a known
// byte offset, producing a signed int64 via ZigZag decoding.
//
// Protobuf varint encoding:
//   - 7 data bits per byte, MSB is continuation flag (1 = more bytes)
//   - Signed integers use ZigZag: (n << 1) ^ (n >> 63)
//   - Maximum 10 bytes for int64
//
// Input:
//   data[]       -- raw protobuf bytes (device-resident)
//   positions[]  -- byte offset where each varint starts
//   data_len     -- total byte count of data buffer
// Output:
//   values[]     -- decoded signed int64 values
//   byte_counts[] -- number of bytes consumed per varint (optional, may be NULL)

extern "C" __global__ void __launch_bounds__(256, 4)
decode_varints_zigzag(
    const unsigned char* __restrict__ data,
    const long long* __restrict__ positions,
    long long* __restrict__ values,
    int* __restrict__ byte_counts,
    long long data_len,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    long long pos = positions[idx];
    unsigned long long raw = 0;
    int shift = 0;
    int bytes_read = 0;

    // Decode up to 10 bytes (max for 64-bit varint)
    for (int b = 0; b < 10; ++b) {
        if (pos + b >= data_len) break;
        unsigned long long byte_val = (unsigned long long)data[pos + b];
        raw |= (byte_val & 0x7FULL) << shift;
        shift += 7;
        bytes_read = b + 1;
        if ((byte_val & 0x80ULL) == 0ULL) break;
    }

    // ZigZag decode: (raw >> 1) ^ -(raw & 1)
    long long decoded = (long long)((raw >> 1) ^ (-(raw & 1ULL)));
    values[idx] = decoded;
    if (byte_counts != 0) {
        byte_counts[idx] = bytes_read;
    }
}

// Unsigned varint decoder -- same as above but without ZigZag.
// Used for field tags and lengths in protobuf.
extern "C" __global__ void __launch_bounds__(256, 4)
decode_varints_unsigned(
    const unsigned char* __restrict__ data,
    const long long* __restrict__ positions,
    unsigned long long* __restrict__ values,
    int* __restrict__ byte_counts,
    long long data_len,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    long long pos = positions[idx];
    unsigned long long raw = 0;
    int shift = 0;
    int bytes_read = 0;

    for (int b = 0; b < 10; ++b) {
        if (pos + b >= data_len) break;
        unsigned long long byte_val = (unsigned long long)data[pos + b];
        raw |= (byte_val & 0x7FULL) << shift;
        shift += 7;
        bytes_read = b + 1;
        if ((byte_val & 0x80ULL) == 0ULL) break;
    }

    values[idx] = raw;
    if (byte_counts != 0) {
        byte_counts[idx] = bytes_read;
    }
}
"""

_VARINT_DECODE_NAMES: tuple[str, ...] = (
    "decode_varints_zigzag",
    "decode_varints_unsigned",
)

# Register for NVRTC precompilation (ADR-0034)
request_nvrtc_warmup([
    ("osm-varint-decode", _VARINT_DECODE_SOURCE, _VARINT_DECODE_NAMES),
])


# ---------------------------------------------------------------------------
# Protobuf wire format constants
# ---------------------------------------------------------------------------

# Wire types
_WIRE_VARINT = 0
_WIRE_64BIT = 1
_WIRE_LENGTH_DELIMITED = 2
_WIRE_32BIT = 5

# PBF top-level blob structure
_BLOBHEADER_TYPE_FIELD = 1       # string
_BLOBHEADER_DATASIZE_FIELD = 3   # int32

_BLOB_RAW_FIELD = 1              # bytes (uncompressed)
_BLOB_RAW_SIZE_FIELD = 2         # int32
_BLOB_ZLIB_DATA_FIELD = 3        # bytes (zlib compressed)

# PrimitiveBlock fields
_PRIMITIVEBLOCK_STRINGTABLE_FIELD = 1
_PRIMITIVEBLOCK_PRIMITIVEGROUP_FIELD = 2
_PRIMITIVEBLOCK_GRANULARITY_FIELD = 17    # int32, default 100
_PRIMITIVEBLOCK_LAT_OFFSET_FIELD = 19     # int64, default 0
_PRIMITIVEBLOCK_LON_OFFSET_FIELD = 20     # int64, default 0

# PrimitiveGroup fields
_PRIMITIVEGROUP_DENSE_FIELD = 2           # DenseNodes

# DenseNodes fields
_DENSENODES_ID_FIELD = 1         # packed sint64 (ZigZag + delta)
_DENSENODES_LAT_FIELD = 8        # packed sint64 (ZigZag + delta)
_DENSENODES_LON_FIELD = 9        # packed sint64 (ZigZag + delta)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockInfo:
    """Metadata for one BlobHeader + Blob pair in the PBF file."""

    offset: int             # byte offset of the BlobHeader size prefix
    blob_header_size: int   # size of the BlobHeader protobuf
    blob_size: int          # size of the Blob protobuf (from BlobHeader.datasize)
    block_type: str         # "OSMHeader" or "OSMData"


@dataclass(frozen=True)
class DenseNodesBlock:
    """Extracted varint byte ranges for one DenseNodes within a PrimitiveBlock."""

    id_bytes: bytes         # raw packed varint bytes for delta-encoded IDs
    lat_bytes: bytes        # raw packed varint bytes for delta-encoded lats
    lon_bytes: bytes        # raw packed varint bytes for delta-encoded lons
    granularity: int        # nanodegree granularity (default 100)
    lat_offset: int         # nanodegree lat offset (default 0)
    lon_offset: int         # nanodegree lon offset (default 0)


@dataclass(frozen=True)
class OsmGpuResult:
    """Result of GPU-accelerated OSM PBF reading.

    Attributes
    ----------
    nodes
        Point OwnedGeometryArray containing all extracted nodes.
        ``None`` if no nodes were found.
    node_ids
        Device-resident int64 array of OSM node IDs, parallel to the
        geometry array.  ``None`` if no nodes were found.
    n_nodes
        Total number of nodes extracted.
    """

    nodes: OwnedGeometryArray | None
    node_ids: cp.ndarray | None
    n_nodes: int


# ---------------------------------------------------------------------------
# CPU: Protobuf low-level helpers
# ---------------------------------------------------------------------------

def _decode_varint(data: bytes | memoryview, offset: int) -> tuple[int, int]:
    """Decode one unsigned protobuf varint at *offset*.

    Returns (value, bytes_consumed).
    """
    result = 0
    shift = 0
    pos = offset
    while True:
        if pos >= len(data):
            raise ValueError(f"Truncated varint at offset {offset}")
        b = data[pos]
        result |= (b & 0x7F) << shift
        shift += 7
        pos += 1
        if (b & 0x80) == 0:
            break
        if shift >= 64:
            raise ValueError(f"Varint too long at offset {offset}")
    return result, pos - offset


def _decode_sint64(data: bytes | memoryview, offset: int) -> tuple[int, int]:
    """Decode one ZigZag-encoded signed varint."""
    raw, consumed = _decode_varint(data, offset)
    return (raw >> 1) ^ -(raw & 1), consumed


def _skip_field(data: bytes | memoryview, offset: int, wire_type: int) -> int:
    """Skip a protobuf field and return the new offset."""
    if wire_type == _WIRE_VARINT:
        _, consumed = _decode_varint(data, offset)
        return offset + consumed
    elif wire_type == _WIRE_64BIT:
        return offset + 8
    elif wire_type == _WIRE_LENGTH_DELIMITED:
        length, consumed = _decode_varint(data, offset)
        return offset + consumed + length
    elif wire_type == _WIRE_32BIT:
        return offset + 4
    else:
        raise ValueError(f"Unknown wire type {wire_type}")


def _parse_field_tag(data: bytes | memoryview, offset: int) -> tuple[int, int, int]:
    """Parse a protobuf field tag.

    Returns (field_number, wire_type, bytes_consumed).
    """
    tag, consumed = _decode_varint(data, offset)
    return tag >> 3, tag & 0x07, consumed


# ---------------------------------------------------------------------------
# CPU: Block index parsing
# ---------------------------------------------------------------------------

def _parse_block_index(path: Path) -> list[BlockInfo]:
    """Parse a PBF file to build a block index.

    PBF files are sequences of:
        4-byte big-endian BlobHeader size
        BlobHeader protobuf (contains type + datasize)
        Blob protobuf
    """
    blocks: list[BlockInfo] = []
    file_size = path.stat().st_size

    with open(path, "rb") as f:
        while f.tell() < file_size:
            block_start = f.tell()

            # Read 4-byte big-endian BlobHeader size
            size_bytes = f.read(4)
            if len(size_bytes) < 4:
                break
            blob_header_size = struct.unpack(">I", size_bytes)[0]

            # Read BlobHeader
            blob_header_data = f.read(blob_header_size)
            if len(blob_header_data) < blob_header_size:
                break

            # Parse BlobHeader to get type and datasize
            block_type = ""
            blob_size = 0
            pos = 0
            while pos < len(blob_header_data):
                field_num, wire_type, consumed = _parse_field_tag(
                    blob_header_data, pos,
                )
                pos += consumed

                if field_num == _BLOBHEADER_TYPE_FIELD and wire_type == _WIRE_LENGTH_DELIMITED:
                    length, consumed = _decode_varint(blob_header_data, pos)
                    pos += consumed
                    block_type = blob_header_data[pos : pos + length].decode("utf-8")
                    pos += length
                elif field_num == _BLOBHEADER_DATASIZE_FIELD and wire_type == _WIRE_VARINT:
                    blob_size, consumed = _decode_varint(blob_header_data, pos)
                    pos += consumed
                else:
                    pos = _skip_field(blob_header_data, pos, wire_type)

            blocks.append(BlockInfo(
                offset=block_start,
                blob_header_size=blob_header_size,
                blob_size=blob_size,
                block_type=block_type,
            ))

            # Skip past the Blob data
            f.seek(block_start + 4 + blob_header_size + blob_size)

    return blocks


# ---------------------------------------------------------------------------
# CPU: Block decompression
# ---------------------------------------------------------------------------

def _decompress_blob(data: bytes) -> bytes:
    """Decompress a single Blob protobuf message.

    Returns the raw (uncompressed) PrimitiveBlock bytes.
    """
    raw_data: bytes | None = None
    zlib_data: bytes | None = None
    pos = 0

    while pos < len(data):
        field_num, wire_type, consumed = _parse_field_tag(data, pos)
        pos += consumed

        if wire_type == _WIRE_LENGTH_DELIMITED:
            length, consumed = _decode_varint(data, pos)
            pos += consumed
            payload = data[pos : pos + length]
            pos += length

            if field_num == _BLOB_RAW_FIELD:
                raw_data = bytes(payload)
            elif field_num == _BLOB_ZLIB_DATA_FIELD:
                zlib_data = bytes(payload)
        elif field_num == _BLOB_RAW_SIZE_FIELD and wire_type == _WIRE_VARINT:
            # raw_size -- we don't need it since zlib.decompress handles sizing
            _, consumed = _decode_varint(data, pos)
            pos += consumed
        else:
            pos = _skip_field(data, pos, wire_type)

    if raw_data is not None:
        return raw_data
    if zlib_data is not None:
        return zlib.decompress(zlib_data)
    raise ValueError("Blob contains neither raw nor zlib_data")


def _read_and_decompress_blocks(
    path: Path,
    block_index: list[BlockInfo],
) -> list[bytes]:
    """Read and decompress all OSMData blocks from a PBF file."""
    results: list[bytes] = []

    with open(path, "rb") as f:
        for info in block_index:
            if info.block_type != "OSMData":
                continue
            # Seek past the 4-byte size prefix and the BlobHeader
            blob_offset = info.offset + 4 + info.blob_header_size
            f.seek(blob_offset)
            blob_data = f.read(info.blob_size)
            results.append(_decompress_blob(blob_data))

    return results


# ---------------------------------------------------------------------------
# CPU: Protobuf field extraction -- locate DenseNodes within PrimitiveBlock
# ---------------------------------------------------------------------------

def _extract_dense_nodes_blocks(raw_blocks: list[bytes]) -> list[DenseNodesBlock]:
    """Extract DenseNodes field byte ranges from decompressed PrimitiveBlocks.

    For each PrimitiveBlock, locates the DenseNodes message and extracts
    the raw packed varint bytes for the id, lat, and lon delta arrays,
    plus the granularity and offset parameters.
    """
    results: list[DenseNodesBlock] = []

    for block_data in raw_blocks:
        granularity = 100       # default per OSM PBF spec
        lat_offset = 0          # default
        lon_offset = 0          # default
        primitive_groups: list[tuple[int, int]] = []  # (start, length) of each group

        # Parse PrimitiveBlock top-level fields
        pos = 0
        while pos < len(block_data):
            field_num, wire_type, consumed = _parse_field_tag(block_data, pos)
            pos += consumed

            if wire_type == _WIRE_LENGTH_DELIMITED:
                length, consumed = _decode_varint(block_data, pos)
                pos += consumed
                if field_num == _PRIMITIVEBLOCK_PRIMITIVEGROUP_FIELD:
                    primitive_groups.append((pos, length))
                pos += length
            elif wire_type == _WIRE_VARINT:
                value, consumed = _decode_varint(block_data, pos)
                pos += consumed
                if field_num == _PRIMITIVEBLOCK_GRANULARITY_FIELD:
                    granularity = value
                elif field_num == _PRIMITIVEBLOCK_LAT_OFFSET_FIELD:
                    # ZigZag decode
                    lat_offset = (value >> 1) ^ -(value & 1)
                elif field_num == _PRIMITIVEBLOCK_LON_OFFSET_FIELD:
                    lon_offset = (value >> 1) ^ -(value & 1)
            else:
                pos = _skip_field(block_data, pos, wire_type)

        # Parse each PrimitiveGroup looking for DenseNodes
        for group_start, group_length in primitive_groups:
            pos = group_start
            end = group_start + group_length

            while pos < end:
                field_num, wire_type, consumed = _parse_field_tag(block_data, pos)
                pos += consumed

                if field_num == _PRIMITIVEGROUP_DENSE_FIELD and wire_type == _WIRE_LENGTH_DELIMITED:
                    length, consumed = _decode_varint(block_data, pos)
                    pos += consumed
                    dense_data = block_data[pos : pos + length]
                    pos += length

                    # Parse DenseNodes to find id, lat, lon packed arrays
                    id_bytes = b""
                    lat_bytes = b""
                    lon_bytes = b""
                    dpos = 0
                    while dpos < len(dense_data):
                        d_field, d_wire, d_consumed = _parse_field_tag(
                            dense_data, dpos,
                        )
                        dpos += d_consumed

                        if d_wire == _WIRE_LENGTH_DELIMITED:
                            d_length, d_consumed = _decode_varint(dense_data, dpos)
                            dpos += d_consumed
                            payload = bytes(dense_data[dpos : dpos + d_length])
                            dpos += d_length
                            if d_field == _DENSENODES_ID_FIELD:
                                id_bytes = payload
                            elif d_field == _DENSENODES_LAT_FIELD:
                                lat_bytes = payload
                            elif d_field == _DENSENODES_LON_FIELD:
                                lon_bytes = payload
                        else:
                            dpos = _skip_field(dense_data, dpos, d_wire)

                    if id_bytes and lat_bytes and lon_bytes:
                        results.append(DenseNodesBlock(
                            id_bytes=id_bytes,
                            lat_bytes=lat_bytes,
                            lon_bytes=lon_bytes,
                            granularity=granularity,
                            lat_offset=lat_offset,
                            lon_offset=lon_offset,
                        ))
                else:
                    pos = _skip_field(block_data, pos, wire_type)

    return results


# ---------------------------------------------------------------------------
# CPU: Count varints in a packed byte array
# ---------------------------------------------------------------------------

def _count_varints(data: bytes) -> int:
    """Count the number of varints in a packed varint byte array.

    Each varint ends at a byte with MSB == 0.
    """
    count = 0
    for b in data:
        if (b & 0x80) == 0:
            count += 1
    return count


def _locate_varint_positions(data: bytes, count: int) -> np.ndarray:
    """Find the byte offset of each varint in a packed array.

    Returns an int64 numpy array of starting positions.
    """
    positions = np.empty(count, dtype=np.int64)
    idx = 0
    pos = 0
    while pos < len(data) and idx < count:
        positions[idx] = pos
        idx += 1
        # Skip to end of this varint
        while pos < len(data):
            if (data[pos] & 0x80) == 0:
                pos += 1
                break
            pos += 1
    return positions


# ---------------------------------------------------------------------------
# GPU: Varint decoding
# ---------------------------------------------------------------------------

def _compile_varint_kernels():
    """Compile the varint decode NVRTC kernels (cached)."""
    return compile_kernel_group(
        "osm-varint-decode",
        _VARINT_DECODE_SOURCE,
        _VARINT_DECODE_NAMES,
    )


def _gpu_decode_varints_zigzag(
    data_bytes: bytes,
    positions: np.ndarray,
) -> cp.ndarray:
    """Decode packed ZigZag varints on GPU.

    Parameters
    ----------
    data_bytes
        Raw protobuf varint bytes.
    positions
        Host int64 array of varint start offsets within data_bytes.

    Returns
    -------
    Device-resident int64 array of decoded signed values.
    """
    n = len(positions)
    if n == 0:
        return cp.empty(0, dtype=cp.int64)

    runtime = get_cuda_runtime()
    kernels = _compile_varint_kernels()

    # Upload data and positions to device
    d_data = cp.frombuffer(data_bytes, dtype=cp.uint8).copy()
    d_positions = cp.asarray(positions, dtype=cp.int64)
    d_values = cp.empty(n, dtype=cp.int64)

    kernel = kernels["decode_varints_zigzag"]
    grid, block = runtime.launch_config(kernel, n)

    ptr = runtime.pointer
    params = (
        (ptr(d_data), ptr(d_positions), ptr(d_values), ptr(None),
         ctypes.c_longlong(len(data_bytes)), ctypes.c_int(n)),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_I64, KERNEL_PARAM_I32),
    )

    runtime.launch(kernel, grid=grid, block=block, params=params)
    # No sync needed -- CuPy cumsum on same stream will observe writes
    return d_values


# ---------------------------------------------------------------------------
# GPU: Delta decode + coordinate scaling
# ---------------------------------------------------------------------------

def _gpu_delta_decode_and_scale(
    dense_blocks: list[DenseNodesBlock],
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray] | None:
    """Decode all DenseNodes blocks and produce absolute coordinates.

    Returns (d_node_ids, d_lat, d_lon) as device-resident arrays,
    or None if there are no nodes.
    """
    if not dense_blocks:
        return None

    all_id_deltas: list[cp.ndarray] = []
    all_lat_deltas: list[cp.ndarray] = []
    all_lon_deltas: list[cp.ndarray] = []
    # Track per-block metadata for coordinate scaling
    block_sizes: list[int] = []
    block_granularities: list[int] = []
    block_lat_offsets: list[int] = []
    block_lon_offsets: list[int] = []

    for dense in dense_blocks:
        # Count and locate varints for each field
        n_ids = _count_varints(dense.id_bytes)
        n_lats = _count_varints(dense.lat_bytes)
        n_lons = _count_varints(dense.lon_bytes)

        if n_ids == 0:
            continue

        # Verify consistent counts
        if not (n_ids == n_lats == n_lons):
            logger.warning(
                "DenseNodes field count mismatch: ids=%d, lats=%d, lons=%d; skipping block",
                n_ids, n_lats, n_lons,
            )
            continue

        # Locate varint start positions on CPU (cheap -- sequential scan
        # over small metadata, typically <1MB per block)
        id_positions = _locate_varint_positions(dense.id_bytes, n_ids)
        lat_positions = _locate_varint_positions(dense.lat_bytes, n_lats)
        lon_positions = _locate_varint_positions(dense.lon_bytes, n_lons)

        # GPU decode varints (ZigZag)
        d_id_deltas = _gpu_decode_varints_zigzag(dense.id_bytes, id_positions)
        d_lat_deltas = _gpu_decode_varints_zigzag(dense.lat_bytes, lat_positions)
        d_lon_deltas = _gpu_decode_varints_zigzag(dense.lon_bytes, lon_positions)

        all_id_deltas.append(d_id_deltas)
        all_lat_deltas.append(d_lat_deltas)
        all_lon_deltas.append(d_lon_deltas)
        block_sizes.append(n_ids)
        block_granularities.append(dense.granularity)
        block_lat_offsets.append(dense.lat_offset)
        block_lon_offsets.append(dense.lon_offset)

    if not all_id_deltas:
        return None

    # Check if all blocks share the same granularity/offsets (common case)
    uniform_granularity = all(g == block_granularities[0] for g in block_granularities)
    uniform_offsets = (
        all(o == block_lat_offsets[0] for o in block_lat_offsets)
        and all(o == block_lon_offsets[0] for o in block_lon_offsets)
    )

    # Delta encoding resets at each PBF block boundary -- cumsum must be
    # per-block, then results are concatenated.  When all blocks share the
    # same granularity/offsets (the common case), we apply the single
    # scale factor after concatenation to avoid redundant multiplies.

    result_ids: list[cp.ndarray] = []
    result_lat_nano: list[cp.ndarray] = []
    result_lon_nano: list[cp.ndarray] = []

    for d_ids, d_lats, d_lons in zip(
        all_id_deltas, all_lat_deltas, all_lon_deltas,
    ):
        # Per-block cumsum (Tier 2 CuPy) -- resets at block boundary
        result_ids.append(cp.cumsum(d_ids))
        result_lat_nano.append(cp.cumsum(d_lats))
        result_lon_nano.append(cp.cumsum(d_lons))

    d_node_ids = cp.concatenate(result_ids) if len(result_ids) > 1 else result_ids[0]

    if uniform_granularity and uniform_offsets:
        # Fast path: single scale factor for all blocks
        d_lat_nano_all = (
            cp.concatenate(result_lat_nano) if len(result_lat_nano) > 1
            else result_lat_nano[0]
        )
        d_lon_nano_all = (
            cp.concatenate(result_lon_nano) if len(result_lon_nano) > 1
            else result_lon_nano[0]
        )
        scale = block_granularities[0] * 1e-9
        d_lat = d_lat_nano_all.astype(cp.float64) * scale + (block_lat_offsets[0] * 1e-9)
        d_lon = d_lon_nano_all.astype(cp.float64) * scale + (block_lon_offsets[0] * 1e-9)

        return d_node_ids, d_lat, d_lon

    # Slow path: per-block scaling (different granularities/offsets)
    result_lats: list[cp.ndarray] = []
    result_lons: list[cp.ndarray] = []

    for i, (d_lat_nano, d_lon_nano) in enumerate(
        zip(result_lat_nano, result_lon_nano),
    ):
        scale = block_granularities[i] * 1e-9
        result_lats.append(
            d_lat_nano.astype(cp.float64) * scale + (block_lat_offsets[i] * 1e-9),
        )
        result_lons.append(
            d_lon_nano.astype(cp.float64) * scale + (block_lon_offsets[i] * 1e-9),
        )

    d_lat = cp.concatenate(result_lats) if len(result_lats) > 1 else result_lats[0]
    d_lon = cp.concatenate(result_lons) if len(result_lons) > 1 else result_lons[0]

    return d_node_ids, d_lat, d_lon


# ---------------------------------------------------------------------------
# GPU: OwnedGeometryArray assembly for Points
# ---------------------------------------------------------------------------

def _assemble_point_geometry(
    d_lon: cp.ndarray,
    d_lat: cp.ndarray,
    n_nodes: int,
) -> OwnedGeometryArray:
    """Build a device-resident Point OwnedGeometryArray from lon/lat arrays.

    x = longitude, y = latitude (GIS convention).
    """
    runtime = get_cuda_runtime()
    family = GeometryFamily.POINT

    # All nodes are valid points
    d_validity = cp.ones(n_nodes, dtype=cp.bool_)
    d_tags = cp.full(n_nodes, FAMILY_TAGS[family], dtype=cp.int8)
    d_family_row_offsets = cp.arange(n_nodes, dtype=cp.int32)

    # Point geometry offsets: [0, 1, 2, ..., n] (one coordinate per point)
    d_geometry_offsets = cp.arange(n_nodes + 1, dtype=cp.int32)
    d_empty_mask = cp.zeros(n_nodes, dtype=cp.bool_)

    # Host copies of structural metadata (small, KB-scale)
    validity = runtime.copy_device_to_host(d_validity).astype(np.bool_, copy=False)
    tags = runtime.copy_device_to_host(d_tags).astype(np.int8, copy=False)
    family_row_offsets = runtime.copy_device_to_host(d_family_row_offsets).astype(
        np.int32, copy=False,
    )
    host_geometry_offsets = np.ascontiguousarray(
        runtime.copy_device_to_host(d_geometry_offsets), dtype=np.int32,
    )
    host_empty_mask = np.ascontiguousarray(
        runtime.copy_device_to_host(d_empty_mask), dtype=np.bool_,
    )

    buffer = FamilyGeometryBuffer(
        family=family,
        schema=get_geometry_buffer_schema(family),
        row_count=n_nodes,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=host_geometry_offsets,
        empty_mask=host_empty_mask,
        part_offsets=None,
        ring_offsets=None,
        bounds=None,
        host_materialized=False,
    )

    owned = OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={family: buffer},
        residency=Residency.DEVICE,
        device_state=OwnedGeometryDeviceState(
            validity=d_validity,
            tags=d_tags,
            family_row_offsets=d_family_row_offsets,
            families={
                family: DeviceFamilyGeometryBuffer(
                    family=family,
                    x=d_lon,   # x = longitude
                    y=d_lat,   # y = latitude
                    geometry_offsets=d_geometry_offsets,
                    empty_mask=d_empty_mask,
                    part_offsets=None,
                    ring_offsets=None,
                    bounds=None,
                )
            },
        ),
    )
    owned._record(
        DiagnosticKind.CREATED,
        f"osm_gpu: {n_nodes} Point nodes from PBF",
        visible=True,
    )
    return owned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_osm_pbf_nodes(path: str | Path) -> OsmGpuResult:
    """Extract all nodes from an OSM PBF file as Point geometries.

    Uses a hybrid CPU/GPU pipeline:
    - CPU: parse block structure, decompress zlib, locate protobuf fields
    - GPU: decode varints, delta-decode via cumsum, scale coordinates

    Parameters
    ----------
    path
        Path to the ``.osm.pbf`` file.

    Returns
    -------
    OsmGpuResult
        Contains a device-resident Point OwnedGeometryArray (``nodes``),
        device-resident int64 node IDs (``node_ids``), and total count.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PBF file not found: {path}")

    # Phase 1: CPU -- parse block index
    block_index = _parse_block_index(path)
    logger.info(
        "OSM PBF: %d blocks (%d OSMData)",
        len(block_index),
        sum(1 for b in block_index if b.block_type == "OSMData"),
    )

    # Phase 2: CPU -- decompress all OSMData blocks
    raw_blocks = _read_and_decompress_blocks(path, block_index)
    logger.info("OSM PBF: decompressed %d data blocks", len(raw_blocks))

    # Phase 3: CPU -- extract DenseNodes field byte ranges
    dense_blocks = _extract_dense_nodes_blocks(raw_blocks)
    logger.info("OSM PBF: found %d DenseNodes blocks", len(dense_blocks))

    if not dense_blocks:
        return OsmGpuResult(nodes=None, node_ids=None, n_nodes=0)

    # Phase 4-5: GPU -- varint decode + delta decode + scale
    result = _gpu_delta_decode_and_scale(dense_blocks)
    if result is None:
        return OsmGpuResult(nodes=None, node_ids=None, n_nodes=0)

    d_node_ids, d_lat, d_lon = result
    n_nodes = int(d_node_ids.shape[0])
    logger.info("OSM PBF: extracted %d nodes", n_nodes)

    # Phase 6: GPU -- assemble Point OwnedGeometryArray
    owned = _assemble_point_geometry(d_lon, d_lat, n_nodes)

    return OsmGpuResult(
        nodes=owned,
        node_ids=d_node_ids,
        n_nodes=n_nodes,
    )
