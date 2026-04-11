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

from vibespatial.cuda.cccl_cubin_cache import (
    CacheEntry,
    CcclBinarySearchBuild,
    CcclReduceBuild,
    CcclScanBuild,
    CcclScanBuild060,
    CcclTypeInfo,
    _cache_key,
    _cached_spec_name_set,
    _cccl_cache_enabled,
    _deserialize_cache_entry,
    _extract_kernel_names,
    _family_struct_spec,
    _known_families,
    _normalize_cubin,
    _read_cache_entry,
    _refresh_cached_op_state,
    _scan_struct_type_for_version,
    _serialize_cache_entry,
    _write_cache_entry,
    cache_stats,
    clear_cache,
    is_cached,
    save_after_build,
    try_load_cached,
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

    def test_scan_build_060_has_expected_fields(self):
        import ctypes

        fields = {f[0] for f in CcclScanBuild060._fields_}
        assert "input_type" in fields
        assert "output_type" in fields
        assert "use_warpspeed" in fields
        assert "runtime_policy" in fields
        assert ctypes.sizeof(CcclScanBuild060) == 160

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
        with patch("vibespatial.cuda.cccl_cubin_cache._get_cache_dir", return_value=tmp_path), \
             patch("vibespatial.cuda.cccl_cubin_cache._compute_capability", return_value=(8, 9)), \
             patch("vibespatial.cuda.cccl_cubin_cache._cccl_version", return_value="0.5.1"), \
             patch("vibespatial.cuda.cccl_cubin_cache._disk_cache_writes_disabled", False):
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
        with patch("vibespatial.cuda.cccl_cubin_cache._get_cache_dir", return_value=tmp_path):
            assert _read_cache_entry("no_such_spec") is None

    def test_clear_cache(self, tmp_path: Path):
        (tmp_path / "test.cache").write_bytes(b"data")
        (tmp_path / "test2.cache").write_bytes(b"data2")
        with patch("vibespatial.cuda.cccl_cubin_cache._get_cache_dir", return_value=tmp_path):
            count = clear_cache()
        assert count == 2

    def test_cache_stats(self, tmp_path: Path):
        (tmp_path / "a.cache").write_bytes(b"x" * 100)
        with patch("vibespatial.cuda.cccl_cubin_cache._get_cache_dir", return_value=tmp_path):
            stats = cache_stats()
        assert stats["file_count"] == 1
        assert stats["total_bytes"] == 100


# ---------------------------------------------------------------------------
# Cache key composition
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_includes_all_components(self):
        with patch("vibespatial.cuda.cccl_cubin_cache._compute_capability", return_value=(8, 9)), \
             patch("vibespatial.cuda.cccl_cubin_cache._cccl_version", return_value="0.5.1"):
            key = _cache_key("reduce_sum_f64", b"\x7fELF" + b"\x00" * 100)
            assert "v1" in key
            assert "sm89" in key
            assert "cccl0.5.1" in key
            assert "reduce_sum_f64" in key

    def test_different_cubin_different_key(self):
        with patch("vibespatial.cuda.cccl_cubin_cache._compute_capability", return_value=(8, 9)), \
             patch("vibespatial.cuda.cccl_cubin_cache._cccl_version", return_value="0.5.1"):
            k1 = _cache_key("scan", b"cubin_a")
            k2 = _cache_key("scan", b"cubin_b")
            assert k1 != k2

    def test_different_cc_different_key(self):
        cubin = b"same_cubin"
        with patch("vibespatial.cuda.cccl_cubin_cache._cccl_version", return_value="0.5.1"):
            with patch("vibespatial.cuda.cccl_cubin_cache._compute_capability", return_value=(8, 0)):
                k1 = _cache_key("scan", cubin)
            with patch("vibespatial.cuda.cccl_cubin_cache._compute_capability", return_value=(9, 0)):
                k2 = _cache_key("scan", cubin)
        assert k1 != k2


# ---------------------------------------------------------------------------
# Family struct registry
# ---------------------------------------------------------------------------


class TestFamilyRegistry:
    def test_all_expected_families_present(self):
        expected = {
            "exclusive_scan", "reduce_into", "segmented_reduce",
            "radix_sort", "merge_sort", "unique_by_key",
            "lower_bound", "upper_bound",
        }
        assert expected.issubset(_known_families())

    def test_each_family_has_kernel_fields(self):
        for name in _known_families():
            struct_type, kernel_fields = _family_struct_spec(name)
            assert len(kernel_fields) >= 1, f"{name} has no kernel fields"

    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ("0.5.1", CcclScanBuild),
            ("0.6.0", CcclScanBuild060),
            ("0.6.2", CcclScanBuild060),
            ("unknown", CcclScanBuild),
        ],
    )
    def test_scan_struct_switches_with_version(self, version: str, expected: type):
        assert _scan_struct_type_for_version(version) is expected

    def test_exclusive_scan_family_uses_runtime_version(self):
        with patch("vibespatial.cuda.cccl_cubin_cache._cccl_version", return_value="0.6.0"):
            struct_type, kernel_fields = _family_struct_spec("exclusive_scan")
        assert struct_type is CcclScanBuild060
        assert kernel_fields == ["init_kernel", "scan_kernel"]


class TestOpStateCompatibility:
    def test_refresh_prefers_update_op_state(self):
        calls: list[str] = []

        class _Adapter:
            def update_op_state(self, cccl_op) -> None:
                calls.append("update")
                cccl_op.state = b"patched"

        class _Op:
            state = b""

        op = _Op()
        _refresh_cached_op_state(_Adapter(), op)
        assert calls == ["update"]
        assert op.state == b"patched"

    def test_refresh_uses_get_state_when_update_missing(self):
        class _Adapter:
            def get_state(self) -> bytes:
                return b"stateful"

        class _Op:
            state = b""

        op = _Op()
        _refresh_cached_op_state(_Adapter(), op)
        assert op.state == b"stateful"

    def test_refresh_noops_for_stateless_adapter(self):
        class _Adapter:
            pass

        class _Op:
            state = b"original"

        op = _Op()
        _refresh_cached_op_state(_Adapter(), op)
        assert op.state == b"original"


# ---------------------------------------------------------------------------
# Disk cache probe (is_cached / _cached_spec_name_set)
# ---------------------------------------------------------------------------


class TestIsCached:
    def test_returns_true_for_cached_spec(self, tmp_path: Path):
        with patch("vibespatial.cuda.cccl_cubin_cache._get_cache_dir", return_value=tmp_path), \
             patch("vibespatial.cuda.cccl_cubin_cache._compute_capability", return_value=(8, 9)), \
             patch("vibespatial.cuda.cccl_cubin_cache._cccl_version", return_value="0.5.1"):
            # Create a cache file matching the expected pattern
            fname = "v1-sm89-cccl0.5.1-exclusive_scan_i32-dd7dbbd47276.cache"
            (tmp_path / fname).write_bytes(b"data")
            assert is_cached("exclusive_scan_i32") is True

    def test_returns_false_for_missing_spec(self, tmp_path: Path):
        with patch("vibespatial.cuda.cccl_cubin_cache._get_cache_dir", return_value=tmp_path), \
             patch("vibespatial.cuda.cccl_cubin_cache._compute_capability", return_value=(8, 9)), \
             patch("vibespatial.cuda.cccl_cubin_cache._cccl_version", return_value="0.5.1"):
            assert is_cached("nonexistent_spec") is False

    def test_returns_false_when_cache_disabled(self):
        with patch("vibespatial.cuda.cccl_cubin_cache._cccl_cache_enabled", return_value=False):
            assert is_cached("anything") is False

    def test_returns_false_when_cache_dir_missing(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist"
        with patch("vibespatial.cuda.cccl_cubin_cache._get_cache_dir", return_value=missing):
            assert is_cached("anything") is False

    def test_batch_probe_returns_all_cached_names(self, tmp_path: Path):
        with patch("vibespatial.cuda.cccl_cubin_cache._get_cache_dir", return_value=tmp_path), \
             patch("vibespatial.cuda.cccl_cubin_cache._compute_capability", return_value=(8, 9)), \
             patch("vibespatial.cuda.cccl_cubin_cache._cccl_version", return_value="0.5.1"):
            (tmp_path / "v1-sm89-cccl0.5.1-exclusive_scan_i32-abcd12345678.cache").write_bytes(b"data")
            (tmp_path / "v1-sm89-cccl0.5.1-reduce_sum_f64-efgh56789012.cache").write_bytes(b"data")
            # Wrong CC — should not match
            (tmp_path / "v1-sm80-cccl0.5.1-select_i32-1234567890ab.cache").write_bytes(b"data")
            names = _cached_spec_name_set()
            assert "exclusive_scan_i32" in names
            assert "reduce_sum_f64" in names
            assert "select_i32" not in names  # wrong CC

    def test_ignores_non_cache_files(self, tmp_path: Path):
        with patch("vibespatial.cuda.cccl_cubin_cache._get_cache_dir", return_value=tmp_path), \
             patch("vibespatial.cuda.cccl_cubin_cache._compute_capability", return_value=(8, 9)), \
             patch("vibespatial.cuda.cccl_cubin_cache._cccl_version", return_value="0.5.1"):
            (tmp_path / "v1-sm89-cccl0.5.1-scan_i32-abcdef123456.cache").write_bytes(b"data")
            (tmp_path / "random_file.txt").write_bytes(b"noise")
            (tmp_path / "v1-sm89-cccl0.5.1-scan_i32-abcdef123456.cache.tmp.12345").write_bytes(b"tmp")
            names = _cached_spec_name_set()
            assert "scan_i32" in names
            assert len(names) == 1


@pytest.mark.gpu
def test_exclusive_scan_disk_cache_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A saved exclusive-scan cache entry reloads and replays correctly."""
    import cupy as cp
    import numpy as np
    from cuda.compute import algorithms

    from vibespatial.cuda import cccl_cubin_cache as cache_mod
    from vibespatial.cuda.cccl_precompile import (
        SPEC_REGISTRY,
        _build_cached_callable,
        _query_cached_temp,
    )
    from vibespatial.cuda.cccl_primitives import _sum_op
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    monkeypatch.setenv("VIBESPATIAL_CCCL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VIBESPATIAL_CCCL_CACHE", "1")
    cache_mod._cccl_cache_enabled.cache_clear()
    cache_mod._get_cache_dir.cache_clear()
    try:
        d_in = cp.arange(16, dtype=cp.int32)
        d_out = cp.empty_like(d_in)
        init = np.asarray(0, dtype=np.int32)
        raw_callable = algorithms.make_exclusive_scan(d_in, d_out, _sum_op, init)
        raw_callable(None, d_in, d_out, _sum_op, int(d_in.size), init)

        save_after_build("exclusive_scan_i32", "exclusive_scan", raw_callable)
        matching = [path for path in tmp_path.glob("*.cache") if "exclusive_scan_i32" in path.name]
        assert len(matching) >= 1

        entry = try_load_cached("exclusive_scan_i32", "exclusive_scan")
        assert entry is not None

        spec = SPEC_REGISTRY["exclusive_scan_i32"]
        cached_callable = _build_cached_callable(entry, spec, cp, algorithms)
        assert cached_callable is not None

        temp_bytes = int(_query_cached_temp(cached_callable, spec, cp) or 0)
        d_temp = cp.empty(max(temp_bytes, 1), dtype=cp.uint8)
        d_cached_out = cp.empty_like(d_in)
        cached_callable(d_temp, d_in, d_cached_out, _sum_op, int(d_in.size), init)

        got = cp.asnumpy(d_cached_out)
        expected = np.array(
            [0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105],
            dtype=np.int32,
        )
        assert np.array_equal(got, expected)
    finally:
        cache_mod._cccl_cache_enabled.cache_clear()
        cache_mod._get_cache_dir.cache_clear()
