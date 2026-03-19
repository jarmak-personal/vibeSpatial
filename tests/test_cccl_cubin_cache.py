"""Tests for CCCL CUBIN on-disk cache (cccl_cubin_cache.py).

These tests exercise the cache infrastructure without requiring a GPU:
cache key composition, CUBIN normalization, ELF parsing, disk I/O,
and environment-variable control.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from vibespatial.cccl_cubin_cache import (
    CacheEntry,
    CcclBinarySearchBuild,
    CcclReduceBuild,
    CcclScanBuild,
    CcclTypeInfo,
    _cache_key,
    _cccl_cache_enabled,
    _deserialize_cache_entry,
    _extract_kernel_names,
    _normalize_cubin,
    _read_cache_entry,
    _serialize_cache_entry,
    _write_cache_entry,
    cache_stats,
    clear_cache,
)

# ---------------------------------------------------------------------------
# Environment-variable control
# ---------------------------------------------------------------------------


class TestCacheEnabled:
    def test_enabled_by_default(self):
        _cccl_cache_enabled.cache_clear()
        with patch.dict(os.environ, {}, clear=True):
            _cccl_cache_enabled.cache_clear()
            assert _cccl_cache_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "off", "no"])
    def test_disabled_values(self, value: str):
        _cccl_cache_enabled.cache_clear()
        with patch.dict(os.environ, {"VIBESPATIAL_CCCL_CACHE": value}):
            _cccl_cache_enabled.cache_clear()
            assert _cccl_cache_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes"])
    def test_enabled_values(self, value: str):
        _cccl_cache_enabled.cache_clear()
        with patch.dict(os.environ, {"VIBESPATIAL_CCCL_CACHE": value}):
            _cccl_cache_enabled.cache_clear()
            assert _cccl_cache_enabled() is True


# ---------------------------------------------------------------------------
# ctypes struct sizes
# ---------------------------------------------------------------------------


class TestStructSizes:
    def test_type_info_size(self):
        import ctypes
        # size_t(8) + size_t(8) + int(4) = 20, padded to 24
        assert ctypes.sizeof(CcclTypeInfo) == 24

    def test_scan_build_has_expected_fields(self):
        fields = {f[0] for f in CcclScanBuild._fields_}
        assert "cc" in fields
        assert "cubin" in fields
        assert "cubin_size" in fields
        assert "library" in fields
        assert "init_kernel" in fields
        assert "scan_kernel" in fields
        assert "runtime_policy" in fields

    def test_reduce_build_has_expected_fields(self):
        fields = {f[0] for f in CcclReduceBuild._fields_}
        assert "single_tile_kernel" in fields
        assert "reduction_kernel" in fields
        assert "determinism" in fields

    def test_binary_search_build_is_smallest(self):
        import ctypes
        # Should be the simplest struct (no runtime_policy)
        assert ctypes.sizeof(CcclBinarySearchBuild) < ctypes.sizeof(CcclScanBuild)


# ---------------------------------------------------------------------------
# CUBIN normalization
# ---------------------------------------------------------------------------


class TestCubinNormalization:
    def test_normalizes_session_hash(self):
        cubin = b"_INTERNAL_abc_deadbeef_123 _INTERNAL_def_deadbeef_456"
        normalized = _normalize_cubin(cubin)
        assert b"deadbeef" not in normalized
        assert b"00000000" in normalized

    def test_multiple_hashes_returns_as_is(self):
        cubin = b"_INTERNAL_abc_11111111_1 _INTERNAL_def_22222222_2"
        normalized = _normalize_cubin(cubin)
        # Two different hashes — can't normalize
        assert normalized == cubin

    def test_no_hash_returns_as_is(self):
        cubin = b"no internal symbols here"
        assert _normalize_cubin(cubin) == cubin

    def test_normalized_is_idempotent(self):
        cubin = b"prefix _INTERNAL_foo_aabbccdd_99 suffix"
        first = _normalize_cubin(cubin)
        second = _normalize_cubin(first)
        assert first == second


# ---------------------------------------------------------------------------
# ELF kernel name extraction
# ---------------------------------------------------------------------------


class TestElfParsing:
    def test_non_elf_returns_empty(self):
        assert _extract_kernel_names(b"not an ELF") == []

    def test_too_short_returns_empty(self):
        assert _extract_kernel_names(b"\x7fELF") == []

    def test_32bit_elf_returns_empty(self):
        # Class byte = 1 (32-bit) — we only handle 64-bit
        header = b"\x7fELF\x01" + b"\x00" * 59
        assert _extract_kernel_names(header) == []


# ---------------------------------------------------------------------------
# Cache entry serialization
# ---------------------------------------------------------------------------


class TestCacheEntry:
    def _make_entry(self, name: str = "exclusive_scan_i32") -> CacheEntry:
        return CacheEntry(
            spec_name=name,
            family="exclusive_scan",
            cubin_bytes=b"\x7fELF" + b"\x00" * 100,
            kernel_names={"init_kernel": "kern_init", "scan_kernel": "kern_scan"},
            runtime_policy_bytes=b"\x01\x02\x03\x04" * 16,
            metadata={"cc": 89, "force_inclusive": False, "init_kind": 2},
        )

    def test_serialize_roundtrip(self):
        entry = self._make_entry()
        data = _serialize_cache_entry(entry)
        restored = _deserialize_cache_entry(data)
        assert restored is not None
        assert restored.spec_name == entry.spec_name
        assert restored.cubin_bytes == entry.cubin_bytes
        assert restored.kernel_names == entry.kernel_names
        assert restored.runtime_policy_bytes == entry.runtime_policy_bytes
        assert restored.metadata["cc"] == 89

    def test_deserialize_bad_magic_returns_none(self):
        assert _deserialize_cache_entry(b"BADMAGIC" + b"\x00" * 100) is None

    def test_deserialize_truncated_returns_none(self):
        entry = self._make_entry()
        data = _serialize_cache_entry(entry)
        assert _deserialize_cache_entry(data[:20]) is None

    def test_deserialize_corrupt_json_returns_none(self):
        # Valid magic + header_len pointing to garbage
        data = b"CCCLCCH\x00" + b"\x05\x00\x00\x00" + b"xxxxx"
        assert _deserialize_cache_entry(data) is None

    def test_file_starts_with_magic(self):
        entry = self._make_entry()
        data = _serialize_cache_entry(entry)
        assert data[:8] == b"CCCLCCH\x00"


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------


class TestDiskIO:
    def test_write_and_read_roundtrip(self, tmp_path: Path):
        with patch("vibespatial.cccl_cubin_cache._get_cache_dir", return_value=tmp_path), \
             patch("vibespatial.cccl_cubin_cache._compute_capability", return_value=(8, 9)), \
             patch("vibespatial.cccl_cubin_cache._cccl_version", return_value="0.5.1"), \
             patch("vibespatial.cccl_cubin_cache._disk_cache_writes_disabled", False):
            entry = CacheEntry(
                spec_name="exclusive_scan_i32",
                family="exclusive_scan",
                cubin_bytes=b"\x7fELF" + b"\x00" * 100,
                kernel_names={"init_kernel": "k1", "scan_kernel": "k2"},
                runtime_policy_bytes=b"\x00" * 32,
                metadata={"cc": 89},
            )
            _write_cache_entry(entry)

            # Should find it on read
            loaded = _read_cache_entry("exclusive_scan_i32")
            assert loaded is not None
            assert loaded.spec_name == "exclusive_scan_i32"
            assert loaded.cubin_bytes == entry.cubin_bytes

    def test_read_nonexistent_returns_none(self, tmp_path: Path):
        with patch("vibespatial.cccl_cubin_cache._get_cache_dir", return_value=tmp_path):
            assert _read_cache_entry("no_such_spec") is None

    def test_clear_cache(self, tmp_path: Path):
        (tmp_path / "test.cache").write_bytes(b"data")
        (tmp_path / "test2.cache").write_bytes(b"data2")
        with patch("vibespatial.cccl_cubin_cache._get_cache_dir", return_value=tmp_path):
            count = clear_cache()
        assert count == 2

    def test_cache_stats(self, tmp_path: Path):
        (tmp_path / "a.cache").write_bytes(b"x" * 100)
        with patch("vibespatial.cccl_cubin_cache._get_cache_dir", return_value=tmp_path):
            stats = cache_stats()
        assert stats["file_count"] == 1
        assert stats["total_bytes"] == 100


# ---------------------------------------------------------------------------
# Cache key composition
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_includes_all_components(self):
        with patch("vibespatial.cccl_cubin_cache._compute_capability", return_value=(8, 9)), \
             patch("vibespatial.cccl_cubin_cache._cccl_version", return_value="0.5.1"):
            key = _cache_key("reduce_sum_f64", b"\x7fELF" + b"\x00" * 100)
            assert "v1" in key
            assert "sm89" in key
            assert "cccl0.5.1" in key
            assert "reduce_sum_f64" in key

    def test_different_cubin_different_key(self):
        with patch("vibespatial.cccl_cubin_cache._compute_capability", return_value=(8, 9)), \
             patch("vibespatial.cccl_cubin_cache._cccl_version", return_value="0.5.1"):
            k1 = _cache_key("scan", b"cubin_a")
            k2 = _cache_key("scan", b"cubin_b")
            assert k1 != k2

    def test_different_cc_different_key(self):
        cubin = b"same_cubin"
        with patch("vibespatial.cccl_cubin_cache._cccl_version", return_value="0.5.1"):
            with patch("vibespatial.cccl_cubin_cache._compute_capability", return_value=(8, 0)):
                k1 = _cache_key("scan", cubin)
            with patch("vibespatial.cccl_cubin_cache._compute_capability", return_value=(9, 0)):
                k2 = _cache_key("scan", cubin)
        assert k1 != k2


# ---------------------------------------------------------------------------
# Family struct registry
# ---------------------------------------------------------------------------


class TestFamilyRegistry:
    def test_all_expected_families_present(self):
        from vibespatial.cccl_cubin_cache import _FAMILY_STRUCTS
        expected = {
            "exclusive_scan", "reduce_into", "segmented_reduce",
            "radix_sort", "merge_sort", "unique_by_key",
            "lower_bound", "upper_bound",
        }
        assert expected.issubset(set(_FAMILY_STRUCTS.keys()))

    def test_each_family_has_kernel_fields(self):
        from vibespatial.cccl_cubin_cache import _FAMILY_STRUCTS
        for name, (struct_type, kernel_fields) in _FAMILY_STRUCTS.items():
            assert len(kernel_fields) >= 1, f"{name} has no kernel fields"
