"""Tests for GPU OSM PBF reader: protobuf + geometry construction on GPU.

Tests cover:
- Protobuf varint encoding/decoding (CPU helpers + GPU kernel)
- ZigZag encoding/decoding
- Delta decoding via cumulative sum
- Coordinate scaling from nanodegrees to degrees
- End-to-end: minimal PBF file -> Point OwnedGeometryArray
- Block index parsing
- Blob decompression (raw and zlib)

Test data is constructed programmatically as raw protobuf bytes -- no
external dependencies (osmium, etc.) required.
"""
from __future__ import annotations

import struct
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pytest

from vibespatial.geometry.buffers import GeometryFamily

try:
    import cupy as cp

    HAS_GPU = True
except (ImportError, ModuleNotFoundError):
    HAS_GPU = False

needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="GPU not available")


# ---------------------------------------------------------------------------
# Protobuf encoding helpers -- build raw protobuf bytes for test data
# ---------------------------------------------------------------------------


def _encode_varint(value: int) -> bytes:
    """Encode an unsigned integer as a protobuf varint."""
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)


def _encode_zigzag(value: int) -> int:
    """ZigZag-encode a signed integer."""
    return (value << 1) ^ (value >> 63) if value >= 0 else ((-value - 1) << 1) | 1


def _encode_sint64(value: int) -> bytes:
    """Encode a signed int64 as a ZigZag varint."""
    return _encode_varint(_encode_zigzag(value))


def _encode_field_tag(field_number: int, wire_type: int) -> bytes:
    """Encode a protobuf field tag."""
    return _encode_varint((field_number << 3) | wire_type)


def _encode_length_delimited(field_number: int, data: bytes) -> bytes:
    """Encode a length-delimited protobuf field."""
    return _encode_field_tag(field_number, 2) + _encode_varint(len(data)) + data


def _encode_varint_field(field_number: int, value: int) -> bytes:
    """Encode a varint protobuf field."""
    return _encode_field_tag(field_number, 0) + _encode_varint(value)


def _encode_packed_sint64(values: list[int]) -> bytes:
    """Encode a packed repeated sint64 field (just the packed bytes, no tag)."""
    result = bytearray()
    for v in values:
        result.extend(_encode_sint64(v))
    return bytes(result)


# ---------------------------------------------------------------------------
# Build a minimal valid PBF file
# ---------------------------------------------------------------------------


def _build_dense_nodes(
    id_deltas: list[int],
    lat_deltas: list[int],
    lon_deltas: list[int],
) -> bytes:
    """Build a DenseNodes protobuf message."""
    id_packed = _encode_packed_sint64(id_deltas)
    lat_packed = _encode_packed_sint64(lat_deltas)
    lon_packed = _encode_packed_sint64(lon_deltas)
    return (
        _encode_length_delimited(1, id_packed)    # field 1: id
        + _encode_length_delimited(8, lat_packed)  # field 8: lat
        + _encode_length_delimited(9, lon_packed)  # field 9: lon
    )


def _build_primitive_group(dense_nodes: bytes) -> bytes:
    """Build a PrimitiveGroup containing DenseNodes."""
    return _encode_length_delimited(2, dense_nodes)  # field 2: dense


def _build_primitive_block(
    primitive_group: bytes,
    granularity: int = 100,
    lat_offset: int = 0,
    lon_offset: int = 0,
) -> bytes:
    """Build a PrimitiveBlock protobuf message."""
    # Stringtable (required, can be empty)
    stringtable = _encode_length_delimited(1, b"")  # field 1: stringtable

    result = stringtable
    # field 2: primitivegroup
    result += _encode_length_delimited(2, primitive_group)
    # field 17: granularity
    result += _encode_varint_field(17, granularity)
    # field 19: lat_offset (ZigZag-encoded)
    if lat_offset != 0:
        result += _encode_varint_field(19, _encode_zigzag(lat_offset))
    # field 20: lon_offset (ZigZag-encoded)
    if lon_offset != 0:
        result += _encode_varint_field(20, _encode_zigzag(lon_offset))

    return result


def _build_blob(raw_data: bytes, compress: bool = True) -> bytes:
    """Build a Blob protobuf message."""
    if compress:
        compressed = zlib.compress(raw_data)
        return (
            _encode_varint_field(2, len(raw_data))              # field 2: raw_size
            + _encode_length_delimited(3, compressed)           # field 3: zlib_data
        )
    else:
        return _encode_length_delimited(1, raw_data)            # field 1: raw


def _build_blob_header(block_type: str, blob_size: int) -> bytes:
    """Build a BlobHeader protobuf message."""
    return (
        _encode_length_delimited(1, block_type.encode("utf-8"))  # field 1: type
        + _encode_varint_field(3, blob_size)                     # field 3: datasize
    )


def _build_pbf_block(block_type: str, payload: bytes, compress: bool = True) -> bytes:
    """Build a complete (BlobHeader size prefix + BlobHeader + Blob) block."""
    blob = _build_blob(payload, compress=compress)
    blob_header = _build_blob_header(block_type, len(blob))
    return struct.pack(">I", len(blob_header)) + blob_header + blob


def _build_osm_header() -> bytes:
    """Build a minimal OSMHeader block."""
    # Minimal HeaderBlock: just required_features
    header_block = _encode_length_delimited(
        4, b"OsmSchema-V0.6",  # field 4: required_features
    )
    return _build_pbf_block("OSMHeader", header_block)


def _build_test_pbf(
    id_deltas: list[int],
    lat_deltas: list[int],
    lon_deltas: list[int],
    granularity: int = 100,
    lat_offset: int = 0,
    lon_offset: int = 0,
    compress: bool = True,
) -> bytes:
    """Build a complete minimal PBF file with one OSMData block."""
    # OSM Header block
    header = _build_osm_header()

    # Build the data block
    dense = _build_dense_nodes(id_deltas, lat_deltas, lon_deltas)
    group = _build_primitive_group(dense)
    primitive_block = _build_primitive_block(
        group,
        granularity=granularity,
        lat_offset=lat_offset,
        lon_offset=lon_offset,
    )
    data_block = _build_pbf_block("OSMData", primitive_block, compress=compress)

    return header + data_block


def _write_temp_pbf(content: bytes) -> Path:
    """Write bytes to a temporary .osm.pbf file and return the path."""
    f = tempfile.NamedTemporaryFile(suffix=".osm.pbf", delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


# ---------------------------------------------------------------------------
# Tests: CPU protobuf helpers
# ---------------------------------------------------------------------------


class TestVarintEncoding:
    """Test varint encode/decode round-trip on CPU."""

    def test_small_values(self):
        from vibespatial.io.osm_gpu import _decode_varint

        for val in [0, 1, 42, 127]:
            encoded = _encode_varint(val)
            decoded, consumed = _decode_varint(encoded, 0)
            assert decoded == val
            assert consumed == len(encoded)

    def test_multibyte_values(self):
        from vibespatial.io.osm_gpu import _decode_varint

        for val in [128, 255, 300, 16384, 2**20, 2**32, 2**50]:
            encoded = _encode_varint(val)
            decoded, consumed = _decode_varint(encoded, 0)
            assert decoded == val
            assert consumed == len(encoded)

    def test_zigzag_round_trip(self):
        """ZigZag encoding: 0->0, -1->1, 1->2, -2->3, 2->4, etc."""
        values = [0, -1, 1, -2, 2, -100, 100, -2**30, 2**30]
        for val in values:
            zz = _encode_zigzag(val)
            # Manual ZigZag decode
            decoded = (zz >> 1) ^ -(zz & 1)
            assert decoded == val, f"ZigZag round-trip failed for {val}"


class TestBlockIndexParsing:
    """Test CPU-side PBF block index parsing."""

    def test_header_and_data_blocks(self):
        from vibespatial.io.osm_gpu import _parse_block_index

        pbf = _build_test_pbf(
            id_deltas=[1, 1, 1],
            lat_deltas=[10, 20, 30],
            lon_deltas=[40, 50, 60],
        )
        path = _write_temp_pbf(pbf)
        try:
            blocks = _parse_block_index(path)
            assert len(blocks) == 2
            assert blocks[0].block_type == "OSMHeader"
            assert blocks[1].block_type == "OSMData"
            assert blocks[1].blob_size > 0
        finally:
            path.unlink()

    def test_empty_file(self):
        from vibespatial.io.osm_gpu import _parse_block_index

        path = _write_temp_pbf(b"")
        try:
            blocks = _parse_block_index(path)
            assert blocks == []
        finally:
            path.unlink()


class TestBlobDecompression:
    """Test CPU-side blob decompression."""

    def test_zlib_compressed(self):
        from vibespatial.io.osm_gpu import _decompress_blob

        payload = b"Hello, OSM world! " * 100
        blob = _build_blob(payload, compress=True)
        result = _decompress_blob(blob)
        assert result == payload

    def test_raw_uncompressed(self):
        from vibespatial.io.osm_gpu import _decompress_blob

        payload = b"Raw data"
        blob = _build_blob(payload, compress=False)
        result = _decompress_blob(blob)
        assert result == payload


class TestDenseNodesExtraction:
    """Test CPU-side extraction of DenseNodes fields from PrimitiveBlock."""

    def test_single_block(self):
        from vibespatial.io.osm_gpu import _extract_dense_nodes_blocks

        id_deltas = [10, 1, 1]
        lat_deltas = [407000000, 100, 200]
        lon_deltas = [-740000000, -50, -100]

        dense = _build_dense_nodes(id_deltas, lat_deltas, lon_deltas)
        group = _build_primitive_group(dense)
        pblock = _build_primitive_block(group, granularity=100)

        results = _extract_dense_nodes_blocks([pblock])
        assert len(results) == 1
        assert results[0].granularity == 100
        assert results[0].lat_offset == 0
        assert results[0].lon_offset == 0
        # Verify varint byte content is non-empty
        assert len(results[0].id_bytes) > 0
        assert len(results[0].lat_bytes) > 0
        assert len(results[0].lon_bytes) > 0

    def test_custom_granularity_and_offsets(self):
        from vibespatial.io.osm_gpu import _extract_dense_nodes_blocks

        dense = _build_dense_nodes([1], [100], [200])
        group = _build_primitive_group(dense)
        pblock = _build_primitive_block(
            group, granularity=1000, lat_offset=500000000, lon_offset=-1000000000,
        )

        results = _extract_dense_nodes_blocks([pblock])
        assert len(results) == 1
        assert results[0].granularity == 1000
        assert results[0].lat_offset == 500000000
        assert results[0].lon_offset == -1000000000


# ---------------------------------------------------------------------------
# Tests: GPU varint decoding
# ---------------------------------------------------------------------------


class TestGpuVarintDecode:
    """Test GPU-side varint decoding kernel."""

    @needs_gpu
    def test_simple_zigzag_values(self):
        """Decode a sequence of known ZigZag-encoded varints."""
        from vibespatial.io.osm_gpu import (
            _count_varints,
            _gpu_decode_varints_zigzag,
            _locate_varint_positions,
        )

        # Encode known values
        values = [0, 1, -1, 42, -42, 1000, -1000, 2**20, -(2**20)]
        packed = _encode_packed_sint64(values)

        n = _count_varints(packed)
        assert n == len(values)

        positions = _locate_varint_positions(packed, n)
        d_result = _gpu_decode_varints_zigzag(packed, positions)
        result = cp.asnumpy(d_result)

        np.testing.assert_array_equal(result, values)

    @needs_gpu
    def test_large_values(self):
        """Decode large int64 values near the boundaries."""
        from vibespatial.io.osm_gpu import (
            _count_varints,
            _gpu_decode_varints_zigzag,
            _locate_varint_positions,
        )

        values = [2**50, -(2**50), 2**62, -(2**62), 0]
        packed = _encode_packed_sint64(values)
        n = _count_varints(packed)
        positions = _locate_varint_positions(packed, n)
        d_result = _gpu_decode_varints_zigzag(packed, positions)
        result = cp.asnumpy(d_result)

        np.testing.assert_array_equal(result, values)

    @needs_gpu
    def test_empty_input(self):
        """Empty byte array produces empty output."""
        from vibespatial.io.osm_gpu import _gpu_decode_varints_zigzag

        positions = np.empty(0, dtype=np.int64)
        d_result = _gpu_decode_varints_zigzag(b"", positions)
        assert d_result.shape[0] == 0

    @needs_gpu
    def test_single_zero(self):
        """Single zero varint."""
        from vibespatial.io.osm_gpu import (
            _count_varints,
            _gpu_decode_varints_zigzag,
            _locate_varint_positions,
        )

        packed = _encode_packed_sint64([0])
        n = _count_varints(packed)
        positions = _locate_varint_positions(packed, n)
        d_result = _gpu_decode_varints_zigzag(packed, positions)
        assert cp.asnumpy(d_result)[0] == 0


# ---------------------------------------------------------------------------
# Tests: GPU delta decoding and coordinate scaling
# ---------------------------------------------------------------------------


class TestGpuDeltaDecode:
    """Test GPU-side delta decode + coordinate scaling."""

    @needs_gpu
    def test_cumsum_produces_absolute_ids(self):
        """Delta-encoded IDs: cumsum of [10, 1, 2, 3] -> [10, 11, 13, 16]."""
        from vibespatial.io.osm_gpu import DenseNodesBlock, _gpu_delta_decode_and_scale

        # Each block's deltas are cumsum'd independently
        # IDs: 10, 11, 13, 16
        id_deltas = [10, 1, 2, 3]
        lat_deltas = [100, 200, 300, 400]
        lon_deltas = [500, 600, 700, 800]

        block = DenseNodesBlock(
            id_bytes=_encode_packed_sint64(id_deltas),
            lat_bytes=_encode_packed_sint64(lat_deltas),
            lon_bytes=_encode_packed_sint64(lon_deltas),
            granularity=100,
            lat_offset=0,
            lon_offset=0,
        )

        result = _gpu_delta_decode_and_scale([block])
        assert result is not None
        d_ids, d_lat, d_lon = result

        ids = cp.asnumpy(d_ids)
        np.testing.assert_array_equal(ids, [10, 11, 13, 16])

    @needs_gpu
    def test_coordinate_scaling(self):
        """Verify nanodegree -> degree conversion with default granularity."""
        from vibespatial.io.osm_gpu import DenseNodesBlock, _gpu_delta_decode_and_scale

        # With granularity=100, lat in nanodegrees = cumsum * 100 * 1e-9
        # If we want lat = 40.7 degrees:
        #   raw nanodegree = 40.7 / (100 * 1e-9) = 40.7 / 1e-7 = 407000000
        # So delta = 407000000 for the first node
        lat_nano_value = 407000000   # -> 40.7 degrees
        lon_nano_value = -740000000  # -> -74.0 degrees

        block = DenseNodesBlock(
            id_bytes=_encode_packed_sint64([1]),
            lat_bytes=_encode_packed_sint64([lat_nano_value]),
            lon_bytes=_encode_packed_sint64([lon_nano_value]),
            granularity=100,
            lat_offset=0,
            lon_offset=0,
        )

        result = _gpu_delta_decode_and_scale([block])
        assert result is not None
        _, d_lat, d_lon = result

        lat = cp.asnumpy(d_lat)
        lon = cp.asnumpy(d_lon)

        np.testing.assert_allclose(lat, [40.7], rtol=1e-10)
        np.testing.assert_allclose(lon, [-74.0], rtol=1e-10)

    @needs_gpu
    def test_coordinate_scaling_with_offsets(self):
        """Verify coordinate scaling with non-zero lat/lon offsets."""
        from vibespatial.io.osm_gpu import DenseNodesBlock, _gpu_delta_decode_and_scale

        # lat = delta * granularity * 1e-9 + lat_offset * 1e-9
        # With granularity=100, lat_offset=1000000000 (0.1 degree offset)
        # delta=407000000: lat = 407000000 * 100 * 1e-9 + 1000000000 * 1e-9
        #                     = 40.7 + 1.0 = 41.7
        block = DenseNodesBlock(
            id_bytes=_encode_packed_sint64([1]),
            lat_bytes=_encode_packed_sint64([407000000]),
            lon_bytes=_encode_packed_sint64([-740000000]),
            granularity=100,
            lat_offset=10000000000,   # +10.0 degrees
            lon_offset=-5000000000,   # -5.0 degrees
        )

        result = _gpu_delta_decode_and_scale([block])
        assert result is not None
        _, d_lat, d_lon = result

        lat = cp.asnumpy(d_lat)
        lon = cp.asnumpy(d_lon)

        np.testing.assert_allclose(lat, [40.7 + 10.0], rtol=1e-10)
        np.testing.assert_allclose(lon, [-74.0 + (-5.0)], rtol=1e-10)

    @needs_gpu
    def test_multiple_nodes_delta(self):
        """Multiple nodes with delta encoding: verify absolute coordinates."""
        from vibespatial.io.osm_gpu import DenseNodesBlock, _gpu_delta_decode_and_scale

        # Three nodes at lat = 40.0, 40.1, 40.3
        # nanodegree values: 400000000, 401000000, 403000000
        # deltas (at granularity=100): 4000000, 10000, 20000
        # (since nanodeg = cumsum(delta) * granularity = cumsum(delta) * 100)
        # Actually: raw = value / (gran * 1e-9) = value / 1e-7
        # 40.0 / 1e-7 = 400000000
        # 40.1 / 1e-7 = 401000000 -> delta = 1000000
        # 40.3 / 1e-7 = 403000000 -> delta = 2000000

        block = DenseNodesBlock(
            id_bytes=_encode_packed_sint64([100, 1, 1]),
            lat_bytes=_encode_packed_sint64([400000000, 1000000, 2000000]),
            lon_bytes=_encode_packed_sint64([-740000000, -100000, -200000]),
            granularity=100,
            lat_offset=0,
            lon_offset=0,
        )

        result = _gpu_delta_decode_and_scale([block])
        assert result is not None
        d_ids, d_lat, d_lon = result

        ids = cp.asnumpy(d_ids)
        lat = cp.asnumpy(d_lat)
        lon = cp.asnumpy(d_lon)

        np.testing.assert_array_equal(ids, [100, 101, 102])
        np.testing.assert_allclose(lat, [40.0, 40.1, 40.3], rtol=1e-10)
        np.testing.assert_allclose(
            lon, [-74.0, -74.01, -74.03], rtol=1e-10,
        )

    @needs_gpu
    def test_multi_block_delta_reset(self):
        """Delta encoding resets at block boundaries -- cumsum is per-block."""
        from vibespatial.io.osm_gpu import DenseNodesBlock, _gpu_delta_decode_and_scale

        # Block 1: ids 100, 101 (deltas: 100, 1)
        # Block 2: ids 200, 201 (deltas: 200, 1)  -- NOT cumsum from block 1
        block1 = DenseNodesBlock(
            id_bytes=_encode_packed_sint64([100, 1]),
            lat_bytes=_encode_packed_sint64([400000000, 1000000]),
            lon_bytes=_encode_packed_sint64([-740000000, -100000]),
            granularity=100,
            lat_offset=0,
            lon_offset=0,
        )
        block2 = DenseNodesBlock(
            id_bytes=_encode_packed_sint64([200, 1]),
            lat_bytes=_encode_packed_sint64([500000000, 2000000]),
            lon_bytes=_encode_packed_sint64([-600000000, 300000]),
            granularity=100,
            lat_offset=0,
            lon_offset=0,
        )

        result = _gpu_delta_decode_and_scale([block1, block2])
        assert result is not None
        d_ids, d_lat, d_lon = result

        ids = cp.asnumpy(d_ids)
        lat = cp.asnumpy(d_lat)
        lon = cp.asnumpy(d_lon)

        # Block 1: ids [100, 101], lat [40.0, 40.1], lon [-74.0, -74.01]
        # Block 2: ids [200, 201], lat [50.0, 50.2], lon [-60.0, -59.97]
        np.testing.assert_array_equal(ids, [100, 101, 200, 201])
        np.testing.assert_allclose(lat, [40.0, 40.1, 50.0, 50.2], rtol=1e-10)
        np.testing.assert_allclose(
            lon, [-74.0, -74.01, -60.0, -59.97], rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# Tests: End-to-end PBF -> Point OwnedGeometryArray
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end tests: PBF file -> OsmGpuResult -> coordinates."""

    @needs_gpu
    def test_single_node(self):
        """One node at known coordinates."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        # Node at (lat=51.5, lon=-0.1) -- London
        # nanodeg = value / (100 * 1e-9) = value / 1e-7
        lat_nano = 515000000     # 51.5 degrees
        lon_nano = -1000000      # -0.1 degrees

        pbf = _build_test_pbf(
            id_deltas=[42],
            lat_deltas=[lat_nano],
            lon_deltas=[lon_nano],
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf_nodes(path)

            assert result.n_nodes == 1
            assert result.nodes is not None
            assert result.node_ids is not None

            node_id = cp.asnumpy(result.node_ids)
            assert node_id[0] == 42

            # Check coordinates from device state
            ds = result.nodes.device_state
            assert ds is not None
            assert GeometryFamily.POINT in ds.families
            pt = ds.families[GeometryFamily.POINT]

            x = cp.asnumpy(pt.x)  # longitude
            y = cp.asnumpy(pt.y)  # latitude

            np.testing.assert_allclose(x, [-0.1], rtol=1e-9)
            np.testing.assert_allclose(y, [51.5], rtol=1e-9)
        finally:
            path.unlink()

    @needs_gpu
    def test_three_nodes(self):
        """Three nodes with delta encoding."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        # Three nodes:
        # Node 1: id=100, lat=40.0, lon=-74.0
        # Node 2: id=101, lat=40.1, lon=-74.01
        # Node 3: id=102, lat=40.3, lon=-74.03
        pbf = _build_test_pbf(
            id_deltas=[100, 1, 1],
            lat_deltas=[400000000, 1000000, 2000000],
            lon_deltas=[-740000000, -100000, -200000],
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf_nodes(path)

            assert result.n_nodes == 3
            ids = cp.asnumpy(result.node_ids)
            np.testing.assert_array_equal(ids, [100, 101, 102])

            ds = result.nodes.device_state
            pt = ds.families[GeometryFamily.POINT]
            x = cp.asnumpy(pt.x)
            y = cp.asnumpy(pt.y)

            np.testing.assert_allclose(y, [40.0, 40.1, 40.3], rtol=1e-10)
            np.testing.assert_allclose(x, [-74.0, -74.01, -74.03], rtol=1e-10)
        finally:
            path.unlink()

    @needs_gpu
    def test_uncompressed_blob(self):
        """PBF with uncompressed (raw) blob data."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        pbf = _build_test_pbf(
            id_deltas=[1],
            lat_deltas=[0],
            lon_deltas=[0],
            compress=False,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf_nodes(path)
            assert result.n_nodes == 1
        finally:
            path.unlink()

    @needs_gpu
    def test_file_not_found(self):
        """Non-existent file raises FileNotFoundError."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        with pytest.raises(FileNotFoundError):
            read_osm_pbf_nodes("/nonexistent/path.osm.pbf")

    @needs_gpu
    def test_geometry_array_properties(self):
        """OwnedGeometryArray has correct residency and family structure."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes
        from vibespatial.runtime.residency import Residency

        pbf = _build_test_pbf(
            id_deltas=[1, 2, 3],
            lat_deltas=[100, 200, 300],
            lon_deltas=[400, 500, 600],
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf_nodes(path)
            owned = result.nodes

            assert owned.residency == Residency.DEVICE
            assert GeometryFamily.POINT in owned.families
            assert owned._row_count == 3

            # Tags should all be POINT
            tags = owned.tags
            from vibespatial.geometry.owned import FAMILY_TAGS

            assert all(t == FAMILY_TAGS[GeometryFamily.POINT] for t in tags)

            # Validity should all be True
            assert all(owned.validity)
        finally:
            path.unlink()

    @needs_gpu
    def test_negative_deltas(self):
        """Nodes with negative delta values (ZigZag encoding)."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        # Nodes moving south and east
        # Node 1: id=1000, lat=50.0, lon=10.0
        # Node 2: id=999 (delta=-1), lat=49.0 (delta=-10000000), lon=11.0 (delta=10000000)
        pbf = _build_test_pbf(
            id_deltas=[1000, -1],
            lat_deltas=[500000000, -10000000],
            lon_deltas=[100000000, 10000000],
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf_nodes(path)
            assert result.n_nodes == 2

            ids = cp.asnumpy(result.node_ids)
            np.testing.assert_array_equal(ids, [1000, 999])

            ds = result.nodes.device_state
            pt = ds.families[GeometryFamily.POINT]
            y = cp.asnumpy(pt.y)
            x = cp.asnumpy(pt.x)

            np.testing.assert_allclose(y, [50.0, 49.0], rtol=1e-10)
            np.testing.assert_allclose(x, [10.0, 11.0], rtol=1e-10)
        finally:
            path.unlink()

    @needs_gpu
    def test_custom_granularity(self):
        """Non-default granularity scales coordinates differently."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        # granularity=1 means 1 nanodegree per unit
        # To get lat=1.0: delta = 1.0 / (1 * 1e-9) = 1000000000
        pbf = _build_test_pbf(
            id_deltas=[1],
            lat_deltas=[1000000000],
            lon_deltas=[2000000000],
            granularity=1,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf_nodes(path)
            assert result.n_nodes == 1

            ds = result.nodes.device_state
            pt = ds.families[GeometryFamily.POINT]
            y = cp.asnumpy(pt.y)
            x = cp.asnumpy(pt.x)

            np.testing.assert_allclose(y, [1.0], rtol=1e-10)
            np.testing.assert_allclose(x, [2.0], rtol=1e-10)
        finally:
            path.unlink()


# ---------------------------------------------------------------------------
# Tests: Varint position locator
# ---------------------------------------------------------------------------


class TestVarintPositionLocator:
    """Test the CPU-side varint position locator."""

    def test_simple_positions(self):
        from vibespatial.io.osm_gpu import _count_varints, _locate_varint_positions

        # Three single-byte varints: 0x00, 0x01, 0x7F
        data = bytes([0x00, 0x01, 0x7F])
        n = _count_varints(data)
        assert n == 3
        positions = _locate_varint_positions(data, n)
        np.testing.assert_array_equal(positions, [0, 1, 2])

    def test_multibyte_varints(self):
        from vibespatial.io.osm_gpu import _count_varints, _locate_varint_positions

        # First varint: 300 = 0xAC 0x02 (2 bytes)
        # Second varint: 1 = 0x01 (1 byte)
        data = bytes([0xAC, 0x02, 0x01])
        n = _count_varints(data)
        assert n == 2
        positions = _locate_varint_positions(data, n)
        np.testing.assert_array_equal(positions, [0, 2])

    def test_empty_data(self):
        from vibespatial.io.osm_gpu import _count_varints, _locate_varint_positions

        n = _count_varints(b"")
        assert n == 0
        positions = _locate_varint_positions(b"", 0)
        assert len(positions) == 0
