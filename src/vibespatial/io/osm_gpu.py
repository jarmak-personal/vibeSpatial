"""GPU-accelerated reader for OpenStreetMap PBF files.

Extracts DenseNodes (Points), Ways (LineStrings/Polygons), and Relations
(MultiPolygons) from OSM PBF files using a hybrid CPU/GPU pipeline:

**Nodes pipeline:**

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

**Ways pipeline:**

1. **Way field extraction** (CPU) -- parse Way messages from each
   PrimitiveGroup to extract way IDs, delta-encoded node refs, and
   tag key/value pairs.  Delta decoding of refs happens on CPU since
   each Way has a small number of refs.
2. **Node lookup table** (GPU, Tier 2 CuPy) -- sort extracted node IDs
   by value to enable binary search.
3. **Coordinate gathering** (GPU, Tier 1 NVRTC) -- for each node ref
   in each Way, binary-search the sorted node table to resolve
   lon/lat coordinates.
4. **Way classification** (GPU, Tier 2 CuPy) -- classify closed Ways
   (first ref == last ref) as Polygon, open Ways as LineString.
5. **Assembly** -- build separate device-resident LineString and Polygon
   OwnedGeometryArrays, then combine into a mixed OwnedGeometryArray.

**Relations pipeline (MultiPolygon assembly):**

1. **Relation field extraction** (CPU) -- parse Relation messages from
   each PrimitiveGroup to extract relation IDs, member IDs/types/roles.
2. **Way lookup** (CPU) -- build a Way ID -> node refs dict from parsed
   Way data so Relations can resolve their Way members.
3. **Way chaining** (CPU, per-relation) -- for each MultiPolygon relation,
   chain outer and inner Way members into closed rings by endpoint matching.
4. **Coordinate gathering** (GPU, Tier 1 NVRTC) -- reuse the existing
   binary-search kernel to resolve all ring node refs to coordinates.
5. **MultiPolygon assembly** -- build device-resident MultiPolygon
   OwnedGeometryArray with geometry/part/ring offset hierarchy.

Tier classification (ADR-0033):
    - Block parsing: CPU (sequential metadata, not parallelizable)
    - Decompression: CPU (zlib)
    - Protobuf field location: CPU (sequential proto traversal, small data)
    - Varint decoding: Tier 1 (custom NVRTC -- binary format-specific)
    - Way coordinate gathering: Tier 1 (custom NVRTC -- binary search)
    - Delta decode (cumsum): Tier 2 (CuPy cumsum)
    - Coordinate scaling: Tier 2 (CuPy element-wise)
    - Node lookup sort: Tier 2 (CuPy argsort)
    - Way classification: Tier 2 (CuPy element-wise comparison)
    - Offset construction: Tier 2 (CuPy)
    - Way chaining: CPU (small graph traversal, <10 ways per ring)
    - Relation coordinate gathering: Tier 1 (reuse NVRTC binary search)

Precision (ADR-0002):
    The varint decode and coordinate gather kernels are integer-only
    or use fp64 storage directly -- no floating-point computation that
    benefits from precision dispatch.  Coordinate storage is always
    fp64 per ADR-0002 (same rationale as csv_gpu.py, kml_gpu.py, and
    all other IO readers).
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

_WAY_COORD_GATHER_SOURCE = r"""
// Way coordinate gatherer -- each thread resolves one node reference
// by binary-searching a sorted node ID table and writing the
// corresponding lon/lat to the output arrays.
//
// Input:
//   sorted_node_ids[] -- device-resident int64, sorted ascending
//   sorted_x[]        -- fp64 longitudes parallel to sorted_node_ids
//   sorted_y[]        -- fp64 latitudes parallel to sorted_node_ids
//   way_refs[]        -- flat int64 array of all node refs across all ways
//   n_refs            -- total number of refs
//   n_nodes           -- size of the sorted node table
// Output:
//   out_x[]           -- fp64 longitudes for each ref (NaN if not found)
//   out_y[]           -- fp64 latitudes for each ref (NaN if not found)

extern "C" __global__ void __launch_bounds__(256, 4)
osm_gather_way_coords(
    const long long* __restrict__ sorted_node_ids,
    const double* __restrict__ sorted_x,
    const double* __restrict__ sorted_y,
    const long long* __restrict__ way_refs,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int n_refs,
    long long n_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_refs) return;

    long long target = way_refs[idx];

    // Binary search for target in sorted_node_ids[0..n_nodes)
    long long lo = 0;
    long long hi = n_nodes;
    while (lo < hi) {
        long long mid = lo + ((hi - lo) >> 1);
        if (sorted_node_ids[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if (lo < n_nodes && sorted_node_ids[lo] == target) {
        out_x[idx] = sorted_x[lo];
        out_y[idx] = sorted_y[lo];
    } else {
        // Node not found -- write NaN so downstream can detect missing refs
        out_x[idx] = __longlong_as_double(0x7FF8000000000000LL);
        out_y[idx] = __longlong_as_double(0x7FF8000000000000LL);
    }
}
"""

_WAY_COORD_GATHER_NAMES: tuple[str, ...] = (
    "osm_gather_way_coords",
)

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
    ("osm-way-coord-gather", _WAY_COORD_GATHER_SOURCE, _WAY_COORD_GATHER_NAMES),
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
_PRIMITIVEGROUP_WAYS_FIELD = 3            # repeated Way
_PRIMITIVEGROUP_RELATIONS_FIELD = 4       # repeated Relation

# DenseNodes fields
_DENSENODES_ID_FIELD = 1         # packed sint64 (ZigZag + delta)
_DENSENODES_LAT_FIELD = 8        # packed sint64 (ZigZag + delta)
_DENSENODES_LON_FIELD = 9        # packed sint64 (ZigZag + delta)

# Way fields
_WAY_ID_FIELD = 1                # int64 (varint)
_WAY_KEYS_FIELD = 2              # packed uint32
_WAY_VALS_FIELD = 3              # packed uint32
_WAY_REFS_FIELD = 8              # packed sint64 (ZigZag + delta)

# Relation fields
_RELATION_ID_FIELD = 1           # int64 (varint)
_RELATION_KEYS_FIELD = 2         # packed uint32
_RELATION_VALS_FIELD = 3         # packed uint32
_RELATION_ROLES_SID_FIELD = 8    # packed int32 (stringtable indices for member roles)
_RELATION_MEMIDS_FIELD = 9       # packed sint64 (delta-encoded member IDs)
_RELATION_TYPES_FIELD = 10       # packed int32 (MemberType enum: 0=NODE, 1=WAY, 2=RELATION)

# Relation MemberType enum
_MEMBER_TYPE_NODE = 0
_MEMBER_TYPE_WAY = 1
_MEMBER_TYPE_RELATION = 2


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
class WayBlock:
    """Extracted data for Ways within one PrimitiveBlock."""

    way_ids: list[int]                    # absolute Way IDs
    refs_per_way: list[list[int]]         # absolute node refs per Way (delta-decoded on CPU)
    tag_keys_per_way: list[list[int]]     # stringtable indices per Way
    tag_vals_per_way: list[list[int]]     # stringtable indices per Way
    stringtable: list[bytes]              # block's string table


@dataclass(frozen=True)
class RelationMember:
    """A single member of an OSM Relation."""

    member_id: int      # absolute ID of the referenced element
    member_type: int    # 0=Node, 1=Way, 2=Relation
    role: str           # "outer", "inner", etc.


@dataclass(frozen=True)
class RelationBlock:
    """Extracted data for Relations within one PrimitiveBlock."""

    relation_ids: list[int]                           # absolute relation IDs
    members_per_relation: list[list[RelationMember]]   # parsed members
    stringtable: list[bytes]                          # block's string table
    granularity: int
    lat_offset: int
    lon_offset: int


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
    ways
        Mixed LineString/Polygon OwnedGeometryArray containing all
        extracted Ways.  ``None`` if no ways were found or no nodes
        available for coordinate resolution.
    way_ids
        Device-resident int64 array of OSM Way IDs, parallel to the
        ways geometry array.  ``None`` if no ways were found.
    n_ways
        Total number of ways extracted.
    relations
        MultiPolygon OwnedGeometryArray from Relation assembly.
        ``None`` if no multipolygon relations were found.
    relation_ids
        Device-resident int64 array of OSM Relation IDs, parallel to
        the relations geometry array.  ``None`` if no relations found.
    n_relations
        Total number of multipolygon relations extracted.
    """

    nodes: OwnedGeometryArray | None
    node_ids: cp.ndarray | None
    n_nodes: int
    ways: OwnedGeometryArray | None = None
    way_ids: cp.ndarray | None = None
    n_ways: int = 0
    relations: OwnedGeometryArray | None = None
    relation_ids: cp.ndarray | None = None
    n_relations: int = 0


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
# CPU: Protobuf field extraction -- locate Ways within PrimitiveBlock
# ---------------------------------------------------------------------------

def _parse_stringtable(block_data: bytes, st_start: int, st_length: int) -> list[bytes]:
    """Parse a StringTable message and return the list of byte strings."""
    strings: list[bytes] = []
    pos = st_start
    end = st_start + st_length
    while pos < end:
        field_num, wire_type, consumed = _parse_field_tag(block_data, pos)
        pos += consumed
        if field_num == 1 and wire_type == _WIRE_LENGTH_DELIMITED:
            length, consumed = _decode_varint(block_data, pos)
            pos += consumed
            strings.append(bytes(block_data[pos : pos + length]))
            pos += length
        else:
            pos = _skip_field(block_data, pos, wire_type)
    return strings


def _decode_packed_uint32(data: bytes) -> list[int]:
    """Decode a packed repeated uint32 field from raw bytes."""
    result: list[int] = []
    pos = 0
    while pos < len(data):
        val, consumed = _decode_varint(data, pos)
        result.append(val)
        pos += consumed
    return result


def _decode_packed_sint64_cpu(data: bytes) -> list[int]:
    """Decode a packed repeated sint64 (ZigZag) field, returning absolute values after delta decode."""
    deltas: list[int] = []
    pos = 0
    while pos < len(data):
        raw, consumed = _decode_varint(data, pos)
        decoded = (raw >> 1) ^ -(raw & 1)
        deltas.append(decoded)
        pos += consumed
    # Delta decode (cumulative sum)
    absolute: list[int] = []
    acc = 0
    for d in deltas:
        acc += d
        absolute.append(acc)
    return absolute


def _parse_single_way(way_data: bytes) -> tuple[int, list[int], list[int], list[int]]:
    """Parse a single Way message.

    Returns (way_id, refs, tag_keys, tag_vals).
    """
    way_id = 0
    refs_bytes = b""
    keys_bytes = b""
    vals_bytes = b""

    pos = 0
    while pos < len(way_data):
        field_num, wire_type, consumed = _parse_field_tag(way_data, pos)
        pos += consumed

        if field_num == _WAY_ID_FIELD and wire_type == _WIRE_VARINT:
            way_id, consumed = _decode_varint(way_data, pos)
            pos += consumed
        elif wire_type == _WIRE_LENGTH_DELIMITED:
            length, consumed = _decode_varint(way_data, pos)
            pos += consumed
            payload = bytes(way_data[pos : pos + length])
            pos += length
            if field_num == _WAY_KEYS_FIELD:
                keys_bytes = payload
            elif field_num == _WAY_VALS_FIELD:
                vals_bytes = payload
            elif field_num == _WAY_REFS_FIELD:
                refs_bytes = payload
        else:
            pos = _skip_field(way_data, pos, wire_type)

    refs = _decode_packed_sint64_cpu(refs_bytes) if refs_bytes else []
    tag_keys = _decode_packed_uint32(keys_bytes) if keys_bytes else []
    tag_vals = _decode_packed_uint32(vals_bytes) if vals_bytes else []

    return way_id, refs, tag_keys, tag_vals


def _extract_way_blocks(raw_blocks: list[bytes]) -> list[WayBlock]:
    """Extract Way data from decompressed PrimitiveBlocks.

    For each PrimitiveBlock, locates Way messages in PrimitiveGroups
    and extracts way IDs, node refs (delta-decoded on CPU), and tags.
    """
    results: list[WayBlock] = []

    for block_data in raw_blocks:
        stringtable: list[bytes] = []
        primitive_groups: list[tuple[int, int]] = []
        st_start = -1
        st_length = 0

        # Parse PrimitiveBlock top-level fields
        pos = 0
        while pos < len(block_data):
            field_num, wire_type, consumed = _parse_field_tag(block_data, pos)
            pos += consumed

            if wire_type == _WIRE_LENGTH_DELIMITED:
                length, consumed = _decode_varint(block_data, pos)
                pos += consumed
                if field_num == _PRIMITIVEBLOCK_STRINGTABLE_FIELD:
                    st_start = pos
                    st_length = length
                elif field_num == _PRIMITIVEBLOCK_PRIMITIVEGROUP_FIELD:
                    primitive_groups.append((pos, length))
                pos += length
            elif wire_type == _WIRE_VARINT:
                _, consumed = _decode_varint(block_data, pos)
                pos += consumed
            else:
                pos = _skip_field(block_data, pos, wire_type)

        if st_start >= 0:
            stringtable = _parse_stringtable(block_data, st_start, st_length)

        # Parse each PrimitiveGroup looking for Ways (field 3)
        block_way_ids: list[int] = []
        block_refs: list[list[int]] = []
        block_tag_keys: list[list[int]] = []
        block_tag_vals: list[list[int]] = []

        for group_start, group_length in primitive_groups:
            pos = group_start
            end = group_start + group_length

            while pos < end:
                field_num, wire_type, consumed = _parse_field_tag(block_data, pos)
                pos += consumed

                if field_num == _PRIMITIVEGROUP_WAYS_FIELD and wire_type == _WIRE_LENGTH_DELIMITED:
                    length, consumed = _decode_varint(block_data, pos)
                    pos += consumed
                    way_data = block_data[pos : pos + length]
                    pos += length

                    way_id, refs, tag_keys, tag_vals = _parse_single_way(way_data)
                    if refs:  # Skip ways with no refs
                        block_way_ids.append(way_id)
                        block_refs.append(refs)
                        block_tag_keys.append(tag_keys)
                        block_tag_vals.append(tag_vals)
                else:
                    pos = _skip_field(block_data, pos, wire_type)

        if block_way_ids:
            results.append(WayBlock(
                way_ids=block_way_ids,
                refs_per_way=block_refs,
                tag_keys_per_way=block_tag_keys,
                tag_vals_per_way=block_tag_vals,
                stringtable=stringtable,
            ))

    return results


# ---------------------------------------------------------------------------
# CPU: Protobuf field extraction -- locate Relations within PrimitiveBlock
# ---------------------------------------------------------------------------

def _parse_single_relation(
    relation_data: bytes,
    stringtable: list[bytes],
) -> tuple[int, list[RelationMember]]:
    """Parse a single Relation message.

    Returns (relation_id, members).
    """
    relation_id = 0
    roles_sid_bytes = b""
    memids_bytes = b""
    types_bytes = b""

    pos = 0
    while pos < len(relation_data):
        field_num, wire_type, consumed = _parse_field_tag(relation_data, pos)
        pos += consumed

        if field_num == _RELATION_ID_FIELD and wire_type == _WIRE_VARINT:
            relation_id, consumed = _decode_varint(relation_data, pos)
            pos += consumed
        elif wire_type == _WIRE_LENGTH_DELIMITED:
            length, consumed = _decode_varint(relation_data, pos)
            pos += consumed
            payload = bytes(relation_data[pos : pos + length])
            pos += length
            if field_num == _RELATION_ROLES_SID_FIELD:
                roles_sid_bytes = payload
            elif field_num == _RELATION_MEMIDS_FIELD:
                memids_bytes = payload
            elif field_num == _RELATION_TYPES_FIELD:
                types_bytes = payload
        else:
            pos = _skip_field(relation_data, pos, wire_type)

    # Decode member fields
    # memids: packed sint64, delta-encoded -> absolute member IDs
    member_ids = _decode_packed_sint64_cpu(memids_bytes) if memids_bytes else []
    # types: packed uint32 -> MemberType enum values
    member_types = _decode_packed_uint32(types_bytes) if types_bytes else []
    # roles_sid: packed uint32 -> stringtable indices
    roles_sid = _decode_packed_uint32(roles_sid_bytes) if roles_sid_bytes else []

    # Build member list -- all three arrays must be parallel
    n_members = min(len(member_ids), len(member_types), len(roles_sid))
    members: list[RelationMember] = []
    for i in range(n_members):
        sid = roles_sid[i]
        role = ""
        if sid < len(stringtable):
            role = stringtable[sid].decode("utf-8", errors="replace")
        members.append(RelationMember(
            member_id=member_ids[i],
            member_type=member_types[i],
            role=role,
        ))

    return relation_id, members


def _extract_relation_blocks(raw_blocks: list[bytes]) -> list[RelationBlock]:
    """Extract Relation data from decompressed PrimitiveBlocks.

    For each PrimitiveBlock, locates Relation messages in PrimitiveGroups
    and extracts relation IDs and member lists (IDs, types, roles).
    """
    results: list[RelationBlock] = []

    for block_data in raw_blocks:
        stringtable: list[bytes] = []
        primitive_groups: list[tuple[int, int]] = []
        st_start = -1
        st_length = 0
        granularity = 100
        lat_offset = 0
        lon_offset = 0

        # Parse PrimitiveBlock top-level fields
        pos = 0
        while pos < len(block_data):
            field_num, wire_type, consumed = _parse_field_tag(block_data, pos)
            pos += consumed

            if wire_type == _WIRE_LENGTH_DELIMITED:
                length, consumed = _decode_varint(block_data, pos)
                pos += consumed
                if field_num == _PRIMITIVEBLOCK_STRINGTABLE_FIELD:
                    st_start = pos
                    st_length = length
                elif field_num == _PRIMITIVEBLOCK_PRIMITIVEGROUP_FIELD:
                    primitive_groups.append((pos, length))
                pos += length
            elif wire_type == _WIRE_VARINT:
                value, consumed = _decode_varint(block_data, pos)
                pos += consumed
                if field_num == _PRIMITIVEBLOCK_GRANULARITY_FIELD:
                    granularity = value
                elif field_num == _PRIMITIVEBLOCK_LAT_OFFSET_FIELD:
                    lat_offset = (value >> 1) ^ -(value & 1)
                elif field_num == _PRIMITIVEBLOCK_LON_OFFSET_FIELD:
                    lon_offset = (value >> 1) ^ -(value & 1)
            else:
                pos = _skip_field(block_data, pos, wire_type)

        if st_start >= 0:
            stringtable = _parse_stringtable(block_data, st_start, st_length)

        # Parse each PrimitiveGroup looking for Relations (field 4)
        block_relation_ids: list[int] = []
        block_members: list[list[RelationMember]] = []

        for group_start, group_length in primitive_groups:
            pos = group_start
            end = group_start + group_length

            while pos < end:
                field_num, wire_type, consumed = _parse_field_tag(block_data, pos)
                pos += consumed

                if field_num == _PRIMITIVEGROUP_RELATIONS_FIELD and wire_type == _WIRE_LENGTH_DELIMITED:
                    length, consumed = _decode_varint(block_data, pos)
                    pos += consumed
                    relation_data = block_data[pos : pos + length]
                    pos += length

                    rel_id, members = _parse_single_relation(
                        relation_data, stringtable,
                    )
                    if members:  # Skip relations with no members
                        block_relation_ids.append(rel_id)
                        block_members.append(members)
                else:
                    pos = _skip_field(block_data, pos, wire_type)

        if block_relation_ids:
            results.append(RelationBlock(
                relation_ids=block_relation_ids,
                members_per_relation=block_members,
                stringtable=stringtable,
                granularity=granularity,
                lat_offset=lat_offset,
                lon_offset=lon_offset,
            ))

    return results


# ---------------------------------------------------------------------------
# CPU: Way chaining for MultiPolygon assembly
# ---------------------------------------------------------------------------

def _build_way_id_to_refs(
    way_blocks: list[WayBlock],
) -> dict[int, list[int]]:
    """Build a Way ID -> node refs lookup from parsed WayBlocks.

    This is CPU-side since it builds a dict from already-parsed Way data.
    """
    lookup: dict[int, list[int]] = {}
    for wb in way_blocks:
        for way_id, refs in zip(wb.way_ids, wb.refs_per_way):
            lookup[way_id] = refs  # noqa: PERF403 – nested loop across blocks
    return lookup


def _chain_ways_to_rings(way_ref_lists: list[list[int]]) -> list[list[int]]:
    """Chain multiple open Ways into closed rings by matching endpoints.

    Algorithm:
    1. Start with any unused Way.  current_ring = way_refs.copy()
    2. While ring is not closed (first != last):
       a. Find a Way whose first_ref == current_ring[-1] -> append (same direction)
       b. Or whose last_ref == current_ring[-1] -> append reversed
       c. If no match found, close the ring forcibly and start a new one
    3. Return list of closed rings.

    This is O(ways_per_ring^2) worst case but typically < 10 Ways per ring.
    """
    if not way_ref_lists:
        return []

    # Filter out single-ref ways (degenerate)
    ways = [refs[:] for refs in way_ref_lists if len(refs) >= 2]
    if not ways:
        return []

    rings: list[list[int]] = []
    used = [False] * len(ways)

    while True:
        # Find the first unused Way to start a new ring
        start_idx = -1
        for i, u in enumerate(used):
            if not u:
                start_idx = i
                break
        if start_idx == -1:
            break  # All ways consumed

        used[start_idx] = True
        ring = ways[start_idx][:]

        # Check if this single Way is already closed
        if ring[0] == ring[-1] and len(ring) >= 4:
            rings.append(ring)
            continue

        # Try to extend the ring by matching endpoints
        changed = True
        while changed and ring[0] != ring[-1]:
            changed = False
            for i, w in enumerate(ways):
                if used[i]:
                    continue
                if w[0] == ring[-1]:
                    # Same direction: append without duplicating the junction node
                    ring.extend(w[1:])
                    used[i] = True
                    changed = True
                    break
                elif w[-1] == ring[-1]:
                    # Reversed direction: append reversed without duplicating
                    ring.extend(reversed(w[:-1]))
                    used[i] = True
                    changed = True
                    break

        # Force-close if still open (degenerate data)
        if ring[0] != ring[-1]:
            ring.append(ring[0])

        if len(ring) >= 4:
            rings.append(ring)

    return rings


def _collect_way_refs(
    rel_id: int,
    members: list[RelationMember],
    way_lookup: dict[int, list[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Collect outer/inner Way ref lists for a relation's Way-type members."""
    outer_way_refs: list[list[int]] = []
    inner_way_refs: list[list[int]] = []

    for m in members:
        if m.member_type != _MEMBER_TYPE_WAY:
            continue

        refs = way_lookup.get(m.member_id)
        if refs is None:
            logger.debug(
                "Relation %d: Way member %d not found in dataset, skipping",
                rel_id, m.member_id,
            )
            continue

        role = m.role.strip().lower()
        if role == "outer" or role == "":
            outer_way_refs.append(refs)
        elif role == "inner":
            inner_way_refs.append(refs)

    return outer_way_refs, inner_way_refs


# ---------------------------------------------------------------------------
# GPU: Relation (MultiPolygon) processing pipeline
# ---------------------------------------------------------------------------

def _process_relations_gpu(
    relation_blocks: list[RelationBlock],
    way_blocks: list[WayBlock],
    d_node_ids: cp.ndarray,
    d_lon: cp.ndarray,
    d_lat: cp.ndarray,
) -> tuple[OwnedGeometryArray | None, cp.ndarray | None, int]:
    """Process all extracted Relations into device-resident MultiPolygon geometries.

    1. Build Way ID -> node refs lookup from WayBlocks
    2. For each multipolygon Relation, chain Way members into closed rings
    3. Flatten all rings' node refs, resolve coordinates via GPU binary search
    4. Build MultiPolygon OwnedGeometryArray with geometry/part/ring offsets

    Returns (relations_owned, d_relation_ids, n_relations).
    """
    # Collect all relations across blocks
    all_relation_ids: list[int] = []
    all_members: list[list[RelationMember]] = []

    for rb in relation_blocks:
        all_relation_ids.extend(rb.relation_ids)
        all_members.extend(rb.members_per_relation)

    if not all_relation_ids:
        return None, None, 0

    # Build Way ID -> node refs lookup
    way_lookup = _build_way_id_to_refs(way_blocks)

    # Resolve relations in two passes:
    #
    # Pass 1: Resolve relations whose members are all Ways (no recursion).
    #   This handles ~99.5% of multipolygon relations.
    #
    # Pass 2: Resolve relations that reference other relations (type=2).
    #   Look up the already-resolved child relation's rings from pass 1
    #   and merge them into the parent.  This handles administrative
    #   boundaries and other "super-relations" (single level of recursion).
    #
    # For each resolved relation, produce:
    #   - List of outer rings (each a list of node refs)
    #   - List of inner rings (each a list of node refs)

    # Index for quick lookup: relation_id -> index in all_relation_ids
    rel_id_to_idx = {rid: i for i, rid in enumerate(all_relation_ids)}

    # Resolved ring data keyed by relation ID (populated in pass 1, read in pass 2)
    resolved_rings: dict[int, tuple[list[list[int]], list[list[int]]]] = {}

    # Track which relations have sub-relation members (need pass 2)
    has_sub_relations: list[int] = []  # relation IDs needing pass 2

    # ------------------------------------------------------------------
    # Pass 1: Resolve Way-only relations
    # ------------------------------------------------------------------
    for rel_id, members in zip(all_relation_ids, all_members):
        has_type2 = any(m.member_type == _MEMBER_TYPE_RELATION for m in members)
        if has_type2:
            has_sub_relations.append(rel_id)
            continue  # defer to pass 2

        outer_way_refs, inner_way_refs = _collect_way_refs(
            rel_id, members, way_lookup,
        )

        if not outer_way_refs:
            continue

        outer_rings = _chain_ways_to_rings(outer_way_refs)
        inner_rings = _chain_ways_to_rings(inner_way_refs) if inner_way_refs else []

        if outer_rings:
            resolved_rings[rel_id] = (outer_rings, inner_rings)

    # ------------------------------------------------------------------
    # Pass 2: Resolve relations with sub-relation members
    # ------------------------------------------------------------------
    for rel_id in has_sub_relations:
        idx = rel_id_to_idx[rel_id]
        members = all_members[idx]

        # Start with Way members (same as pass 1)
        outer_way_refs, inner_way_refs = _collect_way_refs(
            rel_id, members, way_lookup,
        )

        # Merge rings from resolved child relations
        for m in members:
            if m.member_type != _MEMBER_TYPE_RELATION:
                continue

            child_rings = resolved_rings.get(m.member_id)
            if child_rings is None:
                logger.debug(
                    "Relation %d: child relation %d not resolved "
                    "(missing or deeper recursion), skipping",
                    rel_id, m.member_id,
                )
                continue

            child_outer, child_inner = child_rings
            role = m.role.strip().lower()
            if role == "outer" or role == "":
                # Child's outer rings become outer rings of the parent
                for ring in child_outer:
                    outer_way_refs.append(ring)  # already closed rings, no chaining needed
                for ring in child_inner:
                    inner_way_refs.append(ring)
            elif role == "inner":
                # Child's outer rings become inner rings of the parent
                for ring in child_outer:
                    inner_way_refs.append(ring)

        if not outer_way_refs:
            continue

        # Chain any unchained Way refs; already-closed rings from children
        # pass through _chain_ways_to_rings unchanged (they're already closed).
        outer_rings = _chain_ways_to_rings(outer_way_refs)
        inner_rings = _chain_ways_to_rings(inner_way_refs) if inner_way_refs else []

        if outer_rings:
            resolved_rings[rel_id] = (outer_rings, inner_rings)

    # Build the final output from all resolved relations
    valid_relation_ids: list[int] = []
    relation_ring_data: list[tuple[list[list[int]], list[list[int]]]] = []

    for rel_id in all_relation_ids:
        if rel_id in resolved_rings:
            valid_relation_ids.append(rel_id)
            relation_ring_data.append(resolved_rings[rel_id])

    n_relations = len(valid_relation_ids)
    if n_relations == 0:
        return None, None, 0

    # Build the MultiPolygon offset hierarchy and flatten all node refs.
    #
    # MultiPolygon layout: row -> polygon parts -> rings -> coordinates
    # - geometry_offsets[i] = start index into part_offsets for relation i
    # - part_offsets[j] = start index into ring_offsets for polygon j
    # - ring_offsets[k] = start index into x/y for ring k
    #
    # For each relation:
    #   - Each outer ring starts a new "part" (polygon)
    #   - Inner rings are assigned to the most recent outer ring
    #     (correct for well-formed OSM data where inners follow their outer)

    geom_offsets_list = [0]  # row -> polygon part count
    part_offsets_list = [0]  # polygon -> ring count
    ring_offsets_list = [0]  # ring -> coordinate count
    all_ring_refs: list[list[int]] = []  # flat list of all rings' node refs

    for outer_rings, inner_rings in relation_ring_data:
        n_parts = len(outer_rings)
        geom_offsets_list.append(geom_offsets_list[-1] + n_parts)

        # Simple assignment: each outer ring is a part, all inner rings
        # go to the last outer ring.  This is correct for the vast majority
        # of OSM multipolygon data where a single outer + multiple inners
        # is the dominant pattern.
        for oi, outer_ring in enumerate(outer_rings):
            if oi == len(outer_rings) - 1:
                # Last outer ring gets all inner rings
                n_rings = 1 + len(inner_rings)
            else:
                n_rings = 1
            part_offsets_list.append(part_offsets_list[-1] + n_rings)

            # Outer ring
            ring_offsets_list.append(ring_offsets_list[-1] + len(outer_ring))
            all_ring_refs.append(outer_ring)

            # Inner rings (attached to last outer only)
            if oi == len(outer_rings) - 1:
                for inner_ring in inner_rings:
                    ring_offsets_list.append(ring_offsets_list[-1] + len(inner_ring))
                    all_ring_refs.append(inner_ring)

    # Flatten all ring refs into a single array
    flat_refs: list[int] = []
    for ring_refs in all_ring_refs:
        flat_refs.extend(ring_refs)

    total_coords = len(flat_refs)
    if total_coords == 0:
        return None, None, 0

    # Build node lookup table on GPU (Tier 2 CuPy)
    d_sorted_ids, d_sorted_x, d_sorted_y = _build_node_lookup(
        d_node_ids, d_lon, d_lat,
    )

    # Upload flat refs to GPU and resolve coordinates via binary search
    h_flat_refs = np.array(flat_refs, dtype=np.int64)
    d_refs = cp.asarray(h_flat_refs)

    d_out_x, d_out_y = _gpu_gather_way_coords(
        d_sorted_ids, d_sorted_x, d_sorted_y, d_refs,
    )

    # Build offset arrays on GPU (Tier 2 CuPy)
    h_geom_offsets = np.array(geom_offsets_list, dtype=np.int32)
    h_part_offsets = np.array(part_offsets_list, dtype=np.int32)
    h_ring_offsets = np.array(ring_offsets_list, dtype=np.int32)

    d_geom_offsets = cp.asarray(h_geom_offsets)
    d_part_offsets = cp.asarray(h_part_offsets)
    d_ring_offsets = cp.asarray(h_ring_offsets)

    # Build relation IDs on device
    h_relation_ids = np.array(valid_relation_ids, dtype=np.int64)
    d_relation_ids = cp.asarray(h_relation_ids)

    # Assemble MultiPolygon OwnedGeometryArray
    from vibespatial.io.pylibcudf import _build_device_single_family_owned

    d_validity = cp.ones(n_relations, dtype=cp.bool_)
    d_empty_mask = cp.zeros(n_relations, dtype=cp.bool_)

    owned = _build_device_single_family_owned(
        family=GeometryFamily.MULTIPOLYGON,
        validity_device=d_validity,
        x_device=d_out_x,
        y_device=d_out_y,
        geometry_offsets_device=d_geom_offsets,
        empty_mask_device=d_empty_mask,
        part_offsets_device=d_part_offsets,
        ring_offsets_device=d_ring_offsets,
        detail=f"osm_gpu: {n_relations} MultiPolygon Relations from PBF",
    )

    logger.info(
        "OSM PBF: assembled %d MultiPolygon relations (%d total rings, %d coordinates)",
        n_relations, len(all_ring_refs), total_coords,
    )

    return owned, d_relation_ids, n_relations


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
# GPU: Node lookup table for Way coordinate resolution
# ---------------------------------------------------------------------------

def _build_node_lookup(
    d_node_ids: cp.ndarray,
    d_lon: cp.ndarray,
    d_lat: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Build a sorted node ID -> coordinate lookup table on GPU.

    Returns (d_sorted_ids, d_sorted_x, d_sorted_y) sorted by node ID
    for binary search lookup by the Way coordinate gather kernel.

    All operations are device-side (Tier 2 CuPy argsort + fancy indexing).
    """
    d_order = cp.argsort(d_node_ids)
    d_sorted_ids = d_node_ids[d_order]
    d_sorted_x = d_lon[d_order]
    d_sorted_y = d_lat[d_order]
    return d_sorted_ids, d_sorted_x, d_sorted_y


# ---------------------------------------------------------------------------
# GPU: Way coordinate gathering via NVRTC binary search kernel
# ---------------------------------------------------------------------------

def _compile_way_coord_gather_kernel():
    """Compile the Way coordinate gather NVRTC kernel (cached)."""
    return compile_kernel_group(
        "osm-way-coord-gather",
        _WAY_COORD_GATHER_SOURCE,
        _WAY_COORD_GATHER_NAMES,
    )


def _gpu_gather_way_coords(
    d_sorted_ids: cp.ndarray,
    d_sorted_x: cp.ndarray,
    d_sorted_y: cp.ndarray,
    d_way_refs: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Resolve Way node references to coordinates using GPU binary search.

    Parameters
    ----------
    d_sorted_ids
        Device int64 array of node IDs sorted ascending.
    d_sorted_x, d_sorted_y
        Device fp64 coordinate arrays parallel to d_sorted_ids.
    d_way_refs
        Device int64 flat array of all node refs across all Ways.

    Returns
    -------
    (d_out_x, d_out_y) -- device fp64 arrays of resolved coordinates.
    Unresolved refs get NaN.
    """
    n_refs = int(d_way_refs.shape[0])
    if n_refs == 0:
        return cp.empty(0, dtype=cp.float64), cp.empty(0, dtype=cp.float64)

    n_nodes = int(d_sorted_ids.shape[0])
    runtime = get_cuda_runtime()
    kernels = _compile_way_coord_gather_kernel()

    d_out_x = cp.empty(n_refs, dtype=cp.float64)
    d_out_y = cp.empty(n_refs, dtype=cp.float64)

    kernel = kernels["osm_gather_way_coords"]
    grid, block = runtime.launch_config(kernel, n_refs)

    ptr = runtime.pointer
    params = (
        (ptr(d_sorted_ids), ptr(d_sorted_x), ptr(d_sorted_y),
         ptr(d_way_refs), ptr(d_out_x), ptr(d_out_y),
         ctypes.c_int(n_refs), ctypes.c_longlong(n_nodes)),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_I32, KERNEL_PARAM_I64),
    )

    runtime.launch(kernel, grid=grid, block=block, params=params)
    # No sync needed -- subsequent CuPy ops on same stream will observe writes
    return d_out_x, d_out_y


# ---------------------------------------------------------------------------
# GPU: Way processing pipeline
# ---------------------------------------------------------------------------

def _process_ways_gpu(
    way_blocks: list[WayBlock],
    d_node_ids: cp.ndarray,
    d_lon: cp.ndarray,
    d_lat: cp.ndarray,
) -> tuple[OwnedGeometryArray | None, cp.ndarray | None, int]:
    """Process all extracted Ways into device-resident geometries.

    1. Build sorted node lookup table on GPU
    2. Flatten all Way refs and upload to GPU
    3. Binary search to resolve coordinates
    4. Classify Ways as LineString (open) or Polygon (closed)
    5. Assemble OwnedGeometryArray(s)

    Returns (ways_owned, d_way_ids, n_ways).
    """
    # Collect all ways across blocks
    all_way_ids: list[int] = []
    all_refs: list[list[int]] = []

    for wb in way_blocks:
        all_way_ids.extend(wb.way_ids)
        all_refs.extend(wb.refs_per_way)

    n_ways = len(all_way_ids)
    if n_ways == 0:
        return None, None, 0

    # Build node lookup table on GPU (Tier 2 CuPy)
    d_sorted_ids, d_sorted_x, d_sorted_y = _build_node_lookup(
        d_node_ids, d_lon, d_lat,
    )

    # Flatten refs into a single array and compute per-way offsets (CPU,
    # small data -- list of ints from protobuf parsing)
    ref_counts = [len(refs) for refs in all_refs]
    flat_refs: list[int] = []
    for refs in all_refs:
        flat_refs.extend(refs)

    total_refs = len(flat_refs)
    if total_refs == 0:
        return None, None, 0

    # Upload flat refs to GPU
    h_flat_refs = np.array(flat_refs, dtype=np.int64)
    d_way_refs = cp.asarray(h_flat_refs)

    # Resolve coordinates on GPU via binary search (Tier 1 NVRTC)
    d_out_x, d_out_y = _gpu_gather_way_coords(
        d_sorted_ids, d_sorted_x, d_sorted_y, d_way_refs,
    )

    # Compute per-way offsets on GPU (Tier 2 CuPy)
    h_ref_counts = np.array(ref_counts, dtype=np.int32)
    d_ref_counts = cp.asarray(h_ref_counts)
    d_way_offsets = cp.zeros(n_ways + 1, dtype=cp.int32)
    cp.cumsum(d_ref_counts, out=d_way_offsets[1:])

    # Classify: closed Way (first_ref == last_ref) -> Polygon, else LineString
    # Compare first and last ref per way on device (Tier 2 CuPy)
    d_first_ref = d_way_refs[d_way_offsets[:-1].astype(cp.int64)]
    d_last_ref = d_way_refs[(d_way_offsets[1:] - 1).astype(cp.int64)]
    d_is_polygon = d_first_ref == d_last_ref

    # Way IDs to device
    h_way_ids = np.array(all_way_ids, dtype=np.int64)
    d_way_ids = cp.asarray(h_way_ids)

    # Split into LineString and Polygon groups
    d_poly_mask = d_is_polygon
    d_line_mask = ~d_is_polygon

    n_poly = int(cp.sum(d_poly_mask))
    n_line = int(cp.sum(d_line_mask))

    if n_poly == 0 and n_line == 0:
        return None, None, 0

    # Build the OwnedGeometryArray -- use _build_device_single_family_owned
    # for single-family results or _build_device_mixed_owned for mixed.
    from vibespatial.io.pylibcudf import _build_device_mixed_owned

    if n_poly == 0:
        # All LineStrings
        owned = _assemble_linestring_ways(
            d_out_x, d_out_y, d_way_offsets, n_line,
        )
        return owned, d_way_ids, n_ways

    if n_line == 0:
        # All Polygons
        owned = _assemble_polygon_ways(
            d_out_x, d_out_y, d_way_offsets, n_poly,
        )
        return owned, d_way_ids, n_ways

    # Mixed: split coordinates by classification and build mixed owned
    # Reorder ways so LineStrings come first, then Polygons
    d_line_indices = cp.flatnonzero(d_line_mask).astype(cp.int32)
    d_poly_indices = cp.flatnonzero(d_poly_mask).astype(cp.int32)
    d_reorder = cp.concatenate([d_line_indices, d_poly_indices])

    # Reorder way IDs
    d_way_ids = d_way_ids[d_reorder]

    # Extract per-family coordinate slices
    d_line_offsets_raw = d_way_offsets[:-1][d_line_indices]
    d_line_counts = d_ref_counts[d_line_indices]
    d_poly_offsets_raw = d_way_offsets[:-1][d_poly_indices]
    d_poly_counts = d_ref_counts[d_poly_indices]

    # Build LineString family buffers
    d_line_geom_offsets = cp.zeros(n_line + 1, dtype=cp.int32)
    cp.cumsum(d_line_counts, out=d_line_geom_offsets[1:])
    total_line_coords = int(d_line_geom_offsets[-1])

    # Gather LineString coordinates
    d_line_coord_indices = _expand_offsets_to_indices(
        d_line_offsets_raw, d_line_counts, total_line_coords,
    )
    d_line_x = d_out_x[d_line_coord_indices]
    d_line_y = d_out_y[d_line_coord_indices]
    d_line_empty = cp.zeros(n_line, dtype=cp.bool_)

    line_buf = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.LINESTRING,
        x=d_line_x,
        y=d_line_y,
        geometry_offsets=d_line_geom_offsets,
        empty_mask=d_line_empty,
        part_offsets=None,
        ring_offsets=None,
        bounds=None,
    )

    # Build Polygon family buffers
    # Polygon: geometry_offsets (row -> ring), ring_offsets (ring -> coord)
    # Each OSM Way polygon is a single exterior ring, so 1 ring per polygon
    d_poly_geom_offsets = cp.arange(n_poly + 1, dtype=cp.int32)  # each polygon has 1 ring
    d_poly_ring_offsets = cp.zeros(n_poly + 1, dtype=cp.int32)
    cp.cumsum(d_poly_counts, out=d_poly_ring_offsets[1:])
    total_poly_coords = int(d_poly_ring_offsets[-1])

    # Gather Polygon coordinates
    d_poly_coord_indices = _expand_offsets_to_indices(
        d_poly_offsets_raw, d_poly_counts, total_poly_coords,
    )
    d_poly_x = d_out_x[d_poly_coord_indices]
    d_poly_y = d_out_y[d_poly_coord_indices]
    d_poly_empty = cp.zeros(n_poly, dtype=cp.bool_)

    poly_buf = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        x=d_poly_x,
        y=d_poly_y,
        geometry_offsets=d_poly_geom_offsets,
        empty_mask=d_poly_empty,
        part_offsets=None,
        ring_offsets=d_poly_ring_offsets,
        bounds=None,
    )

    # Build mixed validity/tags/family_row_offsets
    total_rows = n_line + n_poly
    d_validity = cp.ones(total_rows, dtype=cp.bool_)
    d_tags = cp.empty(total_rows, dtype=cp.int8)
    d_tags[:n_line] = FAMILY_TAGS[GeometryFamily.LINESTRING]
    d_tags[n_line:] = FAMILY_TAGS[GeometryFamily.POLYGON]

    # family_row_offsets: for each row, the index within its family
    d_family_row_offsets = cp.empty(total_rows, dtype=cp.int32)
    d_family_row_offsets[:n_line] = cp.arange(n_line, dtype=cp.int32)
    d_family_row_offsets[n_line:] = cp.arange(n_poly, dtype=cp.int32)

    owned = _build_device_mixed_owned(
        validity_device=d_validity,
        tags_device=d_tags,
        family_row_offsets_device=d_family_row_offsets,
        family_devices={
            GeometryFamily.LINESTRING: line_buf,
            GeometryFamily.POLYGON: poly_buf,
        },
        detail=f"osm_gpu: {n_line} LineString + {n_poly} Polygon Ways from PBF",
    )
    return owned, d_way_ids, n_ways


def _expand_offsets_to_indices(
    d_starts: cp.ndarray,
    d_counts: cp.ndarray,
    total: int,
) -> cp.ndarray:
    """Expand (start, count) pairs into a flat index array on GPU.

    For starts=[5, 10] counts=[3, 2], produces [5, 6, 7, 10, 11].
    All operations are Tier 2 CuPy -- no host round-trips.
    """
    if total == 0:
        return cp.empty(0, dtype=cp.int64)

    # Build offsets within the output array
    d_out_offsets = cp.zeros(d_counts.shape[0] + 1, dtype=cp.int32)
    cp.cumsum(d_counts, out=d_out_offsets[1:])

    # For each output position, determine which segment it belongs to
    d_segment_ids = cp.searchsorted(d_out_offsets[1:], cp.arange(total, dtype=cp.int32), side="right")
    # Local offset within segment
    d_local = cp.arange(total, dtype=cp.int64) - d_out_offsets[:-1][d_segment_ids].astype(cp.int64)
    # Global index
    d_indices = d_starts[d_segment_ids].astype(cp.int64) + d_local
    return d_indices


def _assemble_linestring_ways(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_way_offsets: cp.ndarray,
    n_ways: int,
) -> OwnedGeometryArray:
    """Build a device-resident LineString OwnedGeometryArray from Way data."""
    from vibespatial.io.pylibcudf import _build_device_single_family_owned

    d_validity = cp.ones(n_ways, dtype=cp.bool_)
    d_empty_mask = cp.zeros(n_ways, dtype=cp.bool_)

    return _build_device_single_family_owned(
        family=GeometryFamily.LINESTRING,
        validity_device=d_validity,
        x_device=d_x,
        y_device=d_y,
        geometry_offsets_device=d_way_offsets,
        empty_mask_device=d_empty_mask,
        detail=f"osm_gpu: {n_ways} LineString Ways from PBF",
    )


def _assemble_polygon_ways(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_way_offsets: cp.ndarray,
    n_ways: int,
) -> OwnedGeometryArray:
    """Build a device-resident Polygon OwnedGeometryArray from Way data.

    Each OSM Way polygon is a single closed ring (no holes), so:
    - geometry_offsets: [0, 1, 2, ..., n] (1 ring per polygon)
    - ring_offsets: way_offsets (ring -> coordinate)
    """
    from vibespatial.io.pylibcudf import _build_device_single_family_owned

    d_validity = cp.ones(n_ways, dtype=cp.bool_)
    d_empty_mask = cp.zeros(n_ways, dtype=cp.bool_)
    d_geom_offsets = cp.arange(n_ways + 1, dtype=cp.int32)  # 1 ring per polygon

    return _build_device_single_family_owned(
        family=GeometryFamily.POLYGON,
        validity_device=d_validity,
        x_device=d_x,
        y_device=d_y,
        geometry_offsets_device=d_geom_offsets,
        empty_mask_device=d_empty_mask,
        ring_offsets_device=d_way_offsets,
        detail=f"osm_gpu: {n_ways} Polygon Ways from PBF",
    )


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
        Way fields are ``None`` / 0.
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


def read_osm_pbf(path: str | Path) -> OsmGpuResult:
    """Extract nodes, ways, and relations from an OSM PBF file.

    Uses a hybrid CPU/GPU pipeline to extract:
    - **Nodes** as Point geometries (same pipeline as ``read_osm_pbf_nodes``)
    - **Ways** as LineString (open) or Polygon (closed ring) geometries
    - **Relations** as MultiPolygon geometries (assembled from Way members)

    Way and relation coordinate resolution uses a GPU binary-search kernel
    against a sorted node lookup table built from the extracted DenseNodes.

    Parameters
    ----------
    path
        Path to the ``.osm.pbf`` file.

    Returns
    -------
    OsmGpuResult
        Contains device-resident Point, LineString, Polygon, and
        MultiPolygon OwnedGeometryArrays with corresponding OSM IDs.
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

    # Phase 3a: CPU -- extract DenseNodes field byte ranges
    dense_blocks = _extract_dense_nodes_blocks(raw_blocks)
    logger.info("OSM PBF: found %d DenseNodes blocks", len(dense_blocks))

    # Phase 3b: CPU -- extract Way data
    way_blocks = _extract_way_blocks(raw_blocks)
    total_ways = sum(len(wb.way_ids) for wb in way_blocks)
    logger.info("OSM PBF: found %d Way blocks (%d ways)", len(way_blocks), total_ways)

    # Phase 3c: CPU -- extract Relation data
    relation_blocks = _extract_relation_blocks(raw_blocks)
    total_relations = sum(len(rb.relation_ids) for rb in relation_blocks)
    logger.info(
        "OSM PBF: found %d Relation blocks (%d relations)",
        len(relation_blocks), total_relations,
    )

    # Phase 4-5: GPU -- node varint decode + delta decode + scale
    nodes_result = _gpu_delta_decode_and_scale(dense_blocks) if dense_blocks else None

    d_node_ids: cp.ndarray | None = None
    d_lat: cp.ndarray | None = None
    d_lon: cp.ndarray | None = None
    n_nodes = 0
    nodes_owned: OwnedGeometryArray | None = None

    if nodes_result is not None:
        d_node_ids, d_lat, d_lon = nodes_result
        n_nodes = int(d_node_ids.shape[0])
        logger.info("OSM PBF: extracted %d nodes", n_nodes)
        nodes_owned = _assemble_point_geometry(d_lon, d_lat, n_nodes)

    # Phase 6: GPU -- resolve Way coordinates and assemble
    ways_owned: OwnedGeometryArray | None = None
    d_way_ids: cp.ndarray | None = None
    n_ways = 0

    if way_blocks and d_node_ids is not None and d_lon is not None and d_lat is not None:
        ways_owned, d_way_ids, n_ways = _process_ways_gpu(
            way_blocks, d_node_ids, d_lon, d_lat,
        )
        logger.info(
            "OSM PBF: assembled %d ways (%s)",
            n_ways,
            "with geometry" if ways_owned is not None else "no resolved geometry",
        )

    # Phase 7: GPU -- resolve Relation Way members and assemble MultiPolygons
    relations_owned: OwnedGeometryArray | None = None
    d_relation_ids: cp.ndarray | None = None
    n_relations = 0

    if (
        relation_blocks
        and way_blocks
        and d_node_ids is not None
        and d_lon is not None
        and d_lat is not None
    ):
        relations_owned, d_relation_ids, n_relations = _process_relations_gpu(
            relation_blocks, way_blocks, d_node_ids, d_lon, d_lat,
        )
        logger.info(
            "OSM PBF: assembled %d relations (%s)",
            n_relations,
            "with geometry" if relations_owned is not None else "no resolved geometry",
        )

    return OsmGpuResult(
        nodes=nodes_owned,
        node_ids=d_node_ids,
        n_nodes=n_nodes,
        ways=ways_owned,
        way_ids=d_way_ids,
        n_ways=n_ways,
        relations=relations_owned,
        relation_ids=d_relation_ids,
        n_relations=n_relations,
    )
