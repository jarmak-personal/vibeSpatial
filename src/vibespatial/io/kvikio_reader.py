"""Accelerated file-to-device reader using kvikio.

When kvikio is installed, reads files directly to GPU device memory through
parallel POSIX threads with pinned bounce buffers.  No GDS (GPU Direct Storage)
is required — kvikio falls back to buffered IO automatically.

When kvikio is not installed, falls back to np.fromfile + cp.asarray with a
manual >2 GiB chunking workaround for CuPy limitations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import kvikio

    _HAS_KVIKIO = True
except ImportError:
    _HAS_KVIKIO = False

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None


@dataclass
class FileReadResult:
    """Result of reading a file to device memory.

    Attributes
    ----------
    device_bytes
        Device-resident uint8 array containing the file contents.
    host_bytes
        Host-resident uint8 array, or ``None`` when kvikio was used
        (the caller should read separately if needed — the OS page
        cache will be warm from the kvikio buffered read).
    """

    device_bytes: cp.ndarray
    host_bytes: np.ndarray | None


def read_file_to_device(path: Path, file_size: int) -> FileReadResult:
    """Read a file directly into a CuPy device array.

    When kvikio is available, uses parallel POSIX reads through pinned
    bounce buffers (configurable via ``KVIKIO_NTHREADS``).  The read uses
    buffered IO by default, which populates the OS page cache as a side
    effect — a subsequent ``np.fromfile`` for the same path will hit warm
    cache.

    When kvikio is not installed, falls back to ``np.fromfile`` followed by
    ``cp.asarray`` with chunking for files larger than 2 GiB.  The host
    array is returned alongside the device array so callers can reuse it
    without a redundant second file read.

    Parameters
    ----------
    path
        Path to the file to read.
    file_size
        Size of the file in bytes.  Passed explicitly to avoid a redundant
        ``stat`` call when the caller already knows it.

    Returns
    -------
    FileReadResult
        ``.device_bytes`` is the device-resident uint8 array.
        ``.host_bytes`` is the host numpy array when the fallback path was
        used, or ``None`` when kvikio handled the transfer.
    """
    if cp is None:
        raise ImportError(
            "cupy is required for read_file_to_device but is not installed"
        )

    if file_size == 0:
        return FileReadResult(
            device_bytes=cp.empty(0, dtype=cp.uint8),
            host_bytes=np.empty(0, dtype=np.uint8),
        )

    if _HAS_KVIKIO:
        d_buf = cp.empty(file_size, dtype=cp.uint8)
        with kvikio.CuFile(str(path), "r") as f:
            nbytes = f.read(d_buf)
        if nbytes != file_size:
            raise OSError(
                f"kvikio short read: got {nbytes} bytes, expected {file_size}"
            )
        return FileReadResult(device_bytes=d_buf, host_bytes=None)

    # Fallback: np.fromfile + cp.asarray with >2 GiB chunking.
    # Return the host array so the caller can reuse it without a
    # redundant second read.
    host_bytes = np.fromfile(str(path), dtype=np.uint8)
    _2GIB = 2 * 1024 * 1024 * 1024
    if len(host_bytes) > _2GIB:
        d_buf = cp.empty(len(host_bytes), dtype=cp.uint8)
        offset = 0
        while offset < len(host_bytes):
            end = min(offset + _2GIB, len(host_bytes))
            d_buf[offset:end] = cp.asarray(host_bytes[offset:end])
            offset = end
        return FileReadResult(device_bytes=d_buf, host_bytes=host_bytes)
    return FileReadResult(
        device_bytes=cp.asarray(host_bytes), host_bytes=host_bytes
    )


def has_kvikio() -> bool:
    """Return True if kvikio is available for accelerated file reads."""
    return _HAS_KVIKIO
