"""Tests for NVRTC disk cache in cuda_runtime."""

from __future__ import annotations

import pytest

from vibespatial.cuda_runtime import (
    _delete_cached_cubin,
    _disk_cache_key,
    _get_cache_dir,
    _nvrtc_cached_key_set,
    _read_cached_cubin,
    _write_cached_cubin,
    clear_nvrtc_cache,
    nvrtc_cache_stats,
    nvrtc_is_cached,
)

# ---------------------------------------------------------------------------
# Cache key composition
# ---------------------------------------------------------------------------


def test_disk_cache_key_includes_all_components():
    key = _disk_cache_key("polygon-area-abc123", (8, 9), (), (12, 6))
    assert key == "v2-sm89-nvrtc12.6-polygon-area-abc123"


def test_disk_cache_key_changes_with_compute_capability():
    k1 = _disk_cache_key("k-abc", (8, 0), (), (12, 6))
    k2 = _disk_cache_key("k-abc", (8, 9), (), (12, 6))
    assert k1 != k2
    assert "sm80" in k1
    assert "sm89" in k2


def test_disk_cache_key_changes_with_nvrtc_version():
    k1 = _disk_cache_key("k-abc", (8, 9), (), (12, 6))
    k2 = _disk_cache_key("k-abc", (8, 9), (), (13, 0))
    assert k1 != k2
    assert "nvrtc12.6" in k1
    assert "nvrtc13.0" in k2


def test_disk_cache_key_changes_with_options():
    k1 = _disk_cache_key("k-abc", (8, 9), (), (12, 6))
    k2 = _disk_cache_key("k-abc", (8, 9), ("--use_fast_math",), (12, 6))
    assert k1 != k2
    assert "opts-" not in k1
    assert "opts-" in k2


def test_disk_cache_key_stable_without_options():
    k1 = _disk_cache_key("k-abc", (8, 9), (), (12, 6))
    k2 = _disk_cache_key("k-abc", (8, 9), (), (12, 6))
    assert k1 == k2


def test_disk_cache_key_options_order_independent():
    k1 = _disk_cache_key("k", (8, 9), ("--a", "--b"), (12, 6))
    k2 = _disk_cache_key("k", (8, 9), ("--b", "--a"), (12, 6))
    assert k1 == k2


# ---------------------------------------------------------------------------
# Read / write roundtrip
# ---------------------------------------------------------------------------


def test_write_and_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    _get_cache_dir.cache_clear()
    try:
        cubin = b"\x7fELF" + b"\x00" * 120  # ELF-like stub
        _write_cached_cubin("test-key", cubin)
        result = _read_cached_cubin("test-key")
        assert result == cubin
    finally:
        _get_cache_dir.cache_clear()


def test_read_nonexistent_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    _get_cache_dir.cache_clear()
    try:
        result = _read_cached_cubin("nonexistent-key")
        assert result is None
    finally:
        _get_cache_dir.cache_clear()


def test_read_corrupt_file_returns_none(tmp_path, monkeypatch):
    """Truncated / too-small file is treated as corrupt and deleted."""
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    _get_cache_dir.cache_clear()
    try:
        path = tmp_path / "test-corrupt.cubin"
        path.write_bytes(b"tiny")  # < 16 bytes
        result = _read_cached_cubin("test-corrupt")
        assert result is None
        assert not path.exists()  # corrupt file cleaned up
    finally:
        _get_cache_dir.cache_clear()


def test_delete_cached_cubin(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    _get_cache_dir.cache_clear()
    try:
        path = tmp_path / "test-delete.cubin"
        path.write_bytes(b"x" * 100)
        assert path.exists()
        _delete_cached_cubin("test-delete")
        assert not path.exists()
    finally:
        _get_cache_dir.cache_clear()


def test_delete_nonexistent_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    _get_cache_dir.cache_clear()
    try:
        _delete_cached_cubin("does-not-exist")  # should not raise
    finally:
        _get_cache_dir.cache_clear()


# ---------------------------------------------------------------------------
# Write failure handling
# ---------------------------------------------------------------------------


def test_write_failure_does_not_crash(tmp_path, monkeypatch):
    """Write to a read-only dir degrades gracefully."""
    import vibespatial.cuda_runtime as mod

    read_only = tmp_path / "readonly"
    read_only.mkdir()
    no_write = read_only / "subdir"
    # Don't create subdir — make parent read-only so mkdir fails
    read_only.chmod(0o444)
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(no_write))
    _get_cache_dir.cache_clear()
    old_flag = mod._disk_cache_writes_disabled
    mod._disk_cache_writes_disabled = False
    try:
        cubin = b"\x7fELF" + b"\x00" * 120
        _write_cached_cubin("test-readonly", cubin)  # should not raise
        assert mod._disk_cache_writes_disabled is True
    finally:
        read_only.chmod(0o755)
        mod._disk_cache_writes_disabled = old_flag
        _get_cache_dir.cache_clear()


# ---------------------------------------------------------------------------
# Atomic write — no partial files
# ---------------------------------------------------------------------------


def test_atomic_write_no_partial_files(tmp_path, monkeypatch):
    """After write, only the final file exists (no .tmp leftovers)."""
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    _get_cache_dir.cache_clear()
    try:
        cubin = b"\x7fELF" + b"\x00" * 120
        _write_cached_cubin("atomic-test", cubin)
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "atomic-test.cubin"
        assert ".tmp" not in files[0].name
    finally:
        _get_cache_dir.cache_clear()


# ---------------------------------------------------------------------------
# Environment variable configuration
# ---------------------------------------------------------------------------


def test_cache_disabled_via_env_var(monkeypatch):
    from vibespatial.cuda_runtime import _disk_cache_enabled
    _disk_cache_enabled.cache_clear()
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE", "0")
    try:
        assert _disk_cache_enabled() is False
    finally:
        _disk_cache_enabled.cache_clear()


def test_cache_enabled_by_default(monkeypatch):
    from vibespatial.cuda_runtime import _disk_cache_enabled
    _disk_cache_enabled.cache_clear()
    monkeypatch.delenv("VIBESPATIAL_NVRTC_CACHE", raising=False)
    try:
        assert _disk_cache_enabled() is True
    finally:
        _disk_cache_enabled.cache_clear()


def test_custom_cache_dir_via_env_var(tmp_path, monkeypatch):
    custom = tmp_path / "my_cache"
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(custom))
    _get_cache_dir.cache_clear()
    try:
        assert _get_cache_dir() == custom
    finally:
        _get_cache_dir.cache_clear()


def test_xdg_cache_home_respected(tmp_path, monkeypatch):
    monkeypatch.delenv("VIBESPATIAL_NVRTC_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _get_cache_dir.cache_clear()
    try:
        assert _get_cache_dir() == tmp_path / "vibespatial" / "nvrtc"
    finally:
        _get_cache_dir.cache_clear()


def test_default_cache_dir(monkeypatch):
    monkeypatch.delenv("VIBESPATIAL_NVRTC_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    _get_cache_dir.cache_clear()
    try:
        import pathlib
        expected = pathlib.Path.home() / ".cache" / "vibespatial" / "nvrtc"
        assert _get_cache_dir() == expected
    finally:
        _get_cache_dir.cache_clear()


# ---------------------------------------------------------------------------
# Integration: disk cache used by compile_kernels
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_compile_kernels_populates_disk_cache(tmp_path, monkeypatch):
    """First compile writes to disk cache; file exists after."""
    from vibespatial.cuda_runtime import (
        _disk_cache_enabled,
        compile_kernel_group,
    )
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE", "1")
    _get_cache_dir.cache_clear()
    _disk_cache_enabled.cache_clear()
    try:
        # Compile a trivial kernel
        source = r"""
        extern "C" __global__ void test_cache_kernel(int* out, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) out[idx] = idx;
        }
        """
        compile_kernel_group("test-disk-cache", source, ("test_cache_kernel",))

        # Verify our CUBIN file was written to disk
        matching = [p for p in tmp_path.glob("*.cubin") if "test-disk-cache" in p.name]
        assert len(matching) == 1
        assert matching[0].stat().st_size > 16
    finally:
        _get_cache_dir.cache_clear()
        _disk_cache_enabled.cache_clear()


# ---------------------------------------------------------------------------
# Public API: clear and stats
# ---------------------------------------------------------------------------


def test_clear_nvrtc_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    _get_cache_dir.cache_clear()
    try:
        # Write some files (mix of cubin and legacy ptx)
        for name in ("a", "b"):
            (tmp_path / f"{name}.cubin").write_bytes(b"x" * 100)
        (tmp_path / "legacy.ptx").write_bytes(b"x" * 100)
        removed = clear_nvrtc_cache()
        assert removed == 3
        assert len(list(tmp_path.glob("*.cubin"))) == 0
        assert len(list(tmp_path.glob("*.ptx"))) == 0
    finally:
        _get_cache_dir.cache_clear()


def test_clear_empty_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    _get_cache_dir.cache_clear()
    try:
        removed = clear_nvrtc_cache()
        assert removed == 0
    finally:
        _get_cache_dir.cache_clear()


def test_nvrtc_cache_stats(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    _get_cache_dir.cache_clear()
    try:
        (tmp_path / "a.cubin").write_bytes(b"x" * 200)
        (tmp_path / "b.cubin").write_bytes(b"y" * 300)
        stats = nvrtc_cache_stats()
        assert stats["directory"] == str(tmp_path)
        assert stats["file_count"] == 2
        assert stats["total_bytes"] == 500
        assert stats["enabled"] is True
    finally:
        _get_cache_dir.cache_clear()


# ---------------------------------------------------------------------------
# Disk cache probe (nvrtc_is_cached / _nvrtc_cached_key_set)
# ---------------------------------------------------------------------------


def test_nvrtc_cached_key_set(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    _get_cache_dir.cache_clear()
    try:
        (tmp_path / "v2-sm89-nvrtc12.6-test-abc123.cubin").write_bytes(b"x" * 100)
        (tmp_path / "v2-sm89-nvrtc12.6-other-def456.cubin").write_bytes(b"x" * 100)
        (tmp_path / "not-a-cubin.txt").write_bytes(b"noise")
        keys = _nvrtc_cached_key_set()
        assert "v2-sm89-nvrtc12.6-test-abc123" in keys
        assert "v2-sm89-nvrtc12.6-other-def456" in keys
        assert len(keys) == 2
    finally:
        _get_cache_dir.cache_clear()


def test_nvrtc_is_cached_true(tmp_path, monkeypatch):
    from unittest.mock import patch

    from vibespatial.cuda_runtime import _nvrtc_version
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    _get_cache_dir.cache_clear()
    _nvrtc_version.cache_clear()
    try:
        with patch("vibespatial.cuda_runtime._nvrtc_version", return_value=(12, 6)):
            disk_key = _disk_cache_key("test-abc123", (8, 9), (), (12, 6))
            (tmp_path / f"{disk_key}.cubin").write_bytes(b"x" * 100)
            assert nvrtc_is_cached("test-abc123", (8, 9), ()) is True
    finally:
        _get_cache_dir.cache_clear()
        _nvrtc_version.cache_clear()


def test_nvrtc_is_cached_false(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    _get_cache_dir.cache_clear()
    try:
        assert nvrtc_is_cached("nonexistent-key", (8, 9), ()) is False
    finally:
        _get_cache_dir.cache_clear()


def test_nvrtc_is_cached_false_when_disabled(monkeypatch):
    from vibespatial.cuda_runtime import _disk_cache_enabled
    _disk_cache_enabled.cache_clear()
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE", "0")
    try:
        assert nvrtc_is_cached("anything", (8, 9), ()) is False
    finally:
        _disk_cache_enabled.cache_clear()


def test_nvrtc_cached_key_set_empty_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    _get_cache_dir.cache_clear()
    try:
        assert _nvrtc_cached_key_set() == frozenset()
    finally:
        _get_cache_dir.cache_clear()
