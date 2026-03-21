"""Tests for kvikio-accelerated file-to-device reader."""
from __future__ import annotations

import numpy as np
import pytest

from vibespatial.io.kvikio_reader import has_kvikio, read_file_to_device

try:
    import cupy as cp

    HAS_GPU = True
except (ImportError, ModuleNotFoundError):
    HAS_GPU = False

needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="GPU not available")


@needs_gpu
def test_read_file_to_device_basic(tmp_path):
    """Round-trip: write known bytes, read to device, verify contents."""
    data = np.arange(1024, dtype=np.uint8)
    path = tmp_path / "test_basic.bin"
    data.tofile(str(path))

    result = read_file_to_device(path, file_size=len(data))

    assert isinstance(result.device_bytes, cp.ndarray)
    assert result.device_bytes.dtype == np.uint8
    assert len(result.device_bytes) == len(data)
    np.testing.assert_array_equal(cp.asnumpy(result.device_bytes), data)


@needs_gpu
def test_read_file_to_device_empty(tmp_path):
    """Zero-byte file returns empty device array."""
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    result = read_file_to_device(path, file_size=0)

    assert isinstance(result.device_bytes, cp.ndarray)
    assert len(result.device_bytes) == 0
    assert result.device_bytes.dtype == np.uint8
    assert result.host_bytes is not None
    assert len(result.host_bytes) == 0


@needs_gpu
def test_read_file_to_device_larger_than_bounce(tmp_path):
    """File larger than kvikio's default 16 MiB bounce buffer."""
    size = 20 * 1024 * 1024  # 20 MiB
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, size=size, dtype=np.uint8)
    path = tmp_path / "large.bin"
    data.tofile(str(path))

    result = read_file_to_device(path, file_size=size)
    d_buf = result.device_bytes

    assert len(d_buf) == size
    # Spot-check: first 1024, last 1024, and a random middle slice
    np.testing.assert_array_equal(cp.asnumpy(d_buf[:1024]), data[:1024])
    np.testing.assert_array_equal(cp.asnumpy(d_buf[-1024:]), data[-1024:])
    mid = size // 2
    np.testing.assert_array_equal(
        cp.asnumpy(d_buf[mid : mid + 1024]), data[mid : mid + 1024]
    )


@needs_gpu
def test_fallback_without_kvikio(tmp_path, monkeypatch):
    """When kvikio is unavailable, falls back to cp.asarray and returns host_bytes."""
    import vibespatial.io.kvikio_reader as mod

    monkeypatch.setattr(mod, "_HAS_KVIKIO", False)

    data = np.arange(256, dtype=np.uint8)
    path = tmp_path / "fallback.bin"
    data.tofile(str(path))

    result = read_file_to_device(path, file_size=len(data))

    assert isinstance(result.device_bytes, cp.ndarray)
    np.testing.assert_array_equal(cp.asnumpy(result.device_bytes), data)
    # Fallback path returns host_bytes for reuse
    assert result.host_bytes is not None
    np.testing.assert_array_equal(result.host_bytes, data)


@needs_gpu
def test_kvikio_path_returns_none_host_bytes(tmp_path):
    """When kvikio is used, host_bytes is None (caller reads separately)."""
    if not has_kvikio():
        pytest.skip("kvikio not installed")

    data = np.arange(512, dtype=np.uint8)
    path = tmp_path / "kvikio_path.bin"
    data.tofile(str(path))

    result = read_file_to_device(path, file_size=len(data))

    assert result.host_bytes is None
    np.testing.assert_array_equal(cp.asnumpy(result.device_bytes), data)


def test_has_kvikio_returns_bool():
    """has_kvikio() returns a bool reflecting import availability."""
    result = has_kvikio()
    assert isinstance(result, bool)
