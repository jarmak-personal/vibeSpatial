"""Bounded PBF container reads and batched nvCOMP Deflate decompression.

Only Blob framing is inspected on the host. Zlib payloads are validated by
Adler-32 on the device after decompression. Missing nvCOMP explicitly declines
native admission; only the bounded file header is decompressed on the host.
"""
from __future__ import annotations

import ctypes
import mmap
import os
import zlib
from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path

import cupy as cp


def _varint(data, pos, end):
    value = 0
    for shift in range(0, 70, 7):
        if pos >= end:
            raise ValueError("Truncated OSM PBF varint")
        byte = data[pos]
        pos += 1
        if shift == 63 and byte > 1:
            raise ValueError("Overflowing OSM PBF varint")
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, pos
    raise ValueError("Overflowing OSM PBF varint")


def _fields(data, begin, end):
    """Bound every container field by its enclosing message, before slicing."""
    pos = begin
    while pos < end:
        tag, pos = _varint(data, pos, end)
        field, wire = tag >> 3, tag & 7
        if not 0 < field < 1 << 29:
            raise ValueError("Invalid OSM PBF protobuf field")
        if wire == 0:
            value, pos = _varint(data, pos, end)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, pos = _varint(data, pos, end)
            else:
                size = 8 if wire == 1 else 4
            if size > end - pos:
                raise ValueError("Truncated OSM PBF protobuf field")
            value = (pos, pos + size)
            pos += size
        else:
            raise ValueError("Unsupported OSM PBF protobuf wire type")
        yield field, wire, value


def _identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class _DeflateOptions(ctypes.Structure):
    _fields_ = [("backend", ctypes.c_int), ("sort", ctypes.c_int), ("reserved", ctypes.c_char * 56)]


@lru_cache(maxsize=1)
def _nvcomp():
    try:
        spec = find_spec("nvidia.libnvcomp")
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        raise NotImplementedError("OSM native inflate requires nvidia-libnvcomp (included with pylibcudf)")
    roots = spec.submodule_search_locations
    try:
        library = ctypes.CDLL(str(Path(next(iter(roots))) / "lib64" / "libnvcomp.so.5"))
    except OSError as exc:
        raise NotImplementedError("OSM native inflate requires the nvCOMP 5 ABI") from exc
    library.nvcompBatchedDeflateDecompressGetTempSizeAsync.restype = ctypes.c_int
    library.nvcompBatchedDeflateDecompressAsync.restype = ctypes.c_int
    library.nvcompBatchedDeflateDecompressGetTempSizeAsync.argtypes = [
        ctypes.c_size_t, ctypes.c_size_t, _DeflateOptions,
        ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t,
    ]
    library.nvcompBatchedDeflateDecompressAsync.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        _DeflateOptions, ctypes.c_void_p, ctypes.c_void_p,
    ]
    return library


@dataclass(frozen=True)
class _Blob:
    start: int
    end: int
    raw_size: int
    compressed: bool
    checksum: int


class PbfSource:
    """Read-only source mapping plus byte-authoritative Blob metadata."""

    def __init__(self, path):
        self.path = Path(path)
        self.file = None
        self.mapping = None
        self.blobs = []

    def __enter__(self):
        self.file = self.path.open("rb")
        try:
            self.identity = _identity(os.fstat(self.file.fileno()))
            if not self.identity[2]:
                raise ValueError("Empty PBF source")
            self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            self._index()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        if self.mapping is not None:
            self.mapping.close()
        if self.file is not None:
            self.file.close()

    def validate_identity(self):
        if _identity(os.fstat(self.file.fileno())) != self.identity or _identity(self.path.stat()) != self.identity:
            raise ValueError("OSM PBF source changed during read")

    def _blob(self, begin, end):
        raw_size, payload = None, None
        for field, wire, value in _fields(self.mapping, begin, end):
            if field == 2 and wire == 0:
                raw_size = value
            elif field in (1, 3, 4, 5, 6, 7) and wire == 2:
                if field not in (1, 3):
                    raise NotImplementedError("Unsupported OSM PBF compression")
                if payload is not None:
                    raise ValueError("OSM PBF Blob has multiple data payloads")
                payload = (*value, field == 3)
            elif field in range(1, 8):
                raise ValueError("Invalid OSM PBF Blob wire type")
        if payload is None:
            raise ValueError("OSM PBF Blob contains no payload")
        start, stop, compressed = payload
        checksum = 0
        if compressed:
            if stop - start < 6 or raw_size is None:
                raise ValueError("Invalid zlib OSM PBF Blob")
            cmf, flg = self.mapping[start:start+2]
            if cmf & 15 != 8 or cmf >> 4 > 7 or ((cmf << 8) | flg) % 31:
                raise ValueError("Invalid OSM PBF zlib header")
            if flg & 32:
                raise NotImplementedError("OSM PBF preset zlib dictionary is unsupported")
            checksum = int.from_bytes(self.mapping[stop-4:stop], "big")
            start += 2
            stop -= 4
        else:
            if raw_size is not None and raw_size != stop - start:
                raise ValueError("OSM PBF raw_size disagrees with raw data")
            raw_size = stop - start
        if not 0 < raw_size < 32 << 20:
            raise ValueError("OSM PBF raw Blob must be smaller than 32 MiB")
        return _Blob(start, stop, raw_size, compressed, checksum)

    def _header(self, blob):
        if blob.compressed:
            codec = zlib.decompressobj()
            header = codec.decompress(self.mapping[blob.start-2:blob.end+4], blob.raw_size+1)
            if len(header) != blob.raw_size or not codec.eof or codec.unused_data:
                raise ValueError("Invalid OSM PBF header compression")
        else:
            header = self.mapping[blob.start:blob.end]
        for field, wire, value in _fields(header, 0, len(header)):
            if field == 4:
                if wire != 2:
                    raise ValueError("Invalid OSM PBF required feature")
                if header[value[0]:value[1]] not in (b"OsmSchema-V0.6", b"DenseNodes"):
                    raise NotImplementedError("Unsupported OSM PBF required feature")

    def _index(self):
        pos, header_seen = 0, False
        while pos < len(self.mapping):
            if len(self.mapping) - pos < 4:
                raise ValueError("Truncated OSM PBF BlobHeader length")
            size = int.from_bytes(self.mapping[pos:pos+4], "big")
            pos += 4
            if not 0 < size < 65536 or size > len(self.mapping)-pos:
                raise ValueError("Truncated or oversized OSM PBF BlobHeader")
            kind, blob_size = None, None
            for field, wire, value in _fields(self.mapping, pos, pos+size):
                if field == 1 and wire == 2:
                    kind = self.mapping[value[0]:value[1]]
                elif field == 3 and wire == 0:
                    blob_size = value
            pos += size
            if blob_size is None or not 0 < blob_size <= (32 << 20)+(64 << 10) or blob_size > len(self.mapping)-pos:
                raise ValueError("Truncated or oversized OSM PBF Blob")
            blob = self._blob(pos, pos+blob_size)
            pos += blob_size
            if kind == b"OSMHeader":
                if header_seen or self.blobs:
                    raise NotImplementedError("Native PBF requires one leading OSMHeader")
                self._header(blob)
                header_seen = True
            elif kind == b"OSMData":
                if not header_seen:
                    raise ValueError("OSM PBF data precedes its required header")
                self.blobs.append(blob)
            else:
                raise NotImplementedError("Unsupported OSM PBF block type")
        self.validate_identity()

    def batches(self, indices=None, *, raw_budget=32 << 20):
        batch, size = [], 0
        for index in range(len(self.blobs)) if indices is None else indices:
            blob = self.blobs[index]
            if batch and size + blob.raw_size > raw_budget:
                yield tuple(batch)
                batch, size = [], 0
            batch.append(index)
            size += blob.raw_size
        if batch:
            yield tuple(batch)

    def inflate(self, indices):
        self.validate_identity()
        blobs = [self.blobs[i] for i in indices]
        total = sum(blob.raw_size for blob in blobs)
        output = cp.empty(total, dtype=cp.uint8)
        starts, offset = [], 0
        input_parts, input_starts, input_sizes, input_offset = [], [], [], 0
        compressed_rows = []
        for row, blob in enumerate(blobs):
            starts.append(offset)
            payload = self.mapping[blob.start:blob.end]
            if blob.compressed:
                compressed_rows.append(row)
                input_starts.append(input_offset)
                input_sizes.append(len(payload))
                pad = -len(payload) % 4
                input_parts.extend((payload, b"\x00" * pad))
                input_offset += len(payload) + pad
            else:
                output[offset:offset+blob.raw_size] = cp.frombuffer(payload, dtype=cp.uint8)
            offset += blob.raw_size
        starts.append(offset)
        offsets = cp.asarray(starts, dtype=cp.int64)
        checksums = cp.asarray([b.checksum if b.compressed else -1 for b in blobs], dtype=cp.int64)
        if not compressed_rows:
            return output, offsets, checksums, None
        compressed = cp.frombuffer(b"".join(input_parts), dtype=cp.uint8)
        iptrs = cp.asarray(input_starts, dtype=cp.uint64) + compressed.data.ptr
        isizes = cp.asarray(input_sizes, dtype=cp.uint64)
        osizes = cp.asarray([blobs[r].raw_size for r in compressed_rows], dtype=cp.uint64)
        optrs = cp.asarray([starts[r] for r in compressed_rows], dtype=cp.uint64) + output.data.ptr
        actual = cp.empty(len(compressed_rows), dtype=cp.uint64)
        status = cp.empty(len(compressed_rows), dtype=cp.int32)
        library = _nvcomp()
        temp_size = ctypes.c_size_t()
        options = _DeflateOptions()
        code = library.nvcompBatchedDeflateDecompressGetTempSizeAsync(
            len(compressed_rows), max(b.raw_size for b in blobs), options,
            ctypes.byref(temp_size), total,
        )
        if code:
            raise RuntimeError(f"nvCOMP Deflate workspace query failed ({code})")
        scratch = cp.empty(temp_size.value, dtype=cp.uint8)
        code = library.nvcompBatchedDeflateDecompressAsync(
            iptrs.data.ptr, isizes.data.ptr, osizes.data.ptr, actual.data.ptr,
            len(compressed_rows), scratch.data.ptr, scratch.nbytes, optrs.data.ptr,
            options, status.data.ptr, cp.cuda.get_current_stream().ptr,
        )
        if code:
            raise RuntimeError(f"nvCOMP Deflate decompression failed ({code})")
        # Same-stream consumers see completed decode; retain all borrowed memory
        # until their completion event, including when asynchronous pools are used.
        from vibespatial.cuda._runtime import get_cuda_completion_retainer

        get_cuda_completion_retainer().defer(
            cp.cuda.get_current_stream(),
            (compressed, iptrs, isizes, osizes, optrs, actual, status, scratch),
            lambda _resources: None,
        )
        validation = cp.any((status != 0) | (actual != osizes)).astype(cp.int32).reshape(1)
        return output, offsets, checksums, validation
