"""CCCL CUBIN on-disk cache for near-zero CCCL readiness on process restart.

After CCCL's make_* builds normally on first run, we extract the compiled
CUBIN and build metadata, cache them to disk.  On subsequent runs, we load
the cached CUBIN via cuLibraryLoadData, reconstruct the build result struct,
and call CCCL's own C compute functions directly via ctypes — same
libcccl.c.parallel.so, same dispatch logic, just without the NVRTC/nvJitLink
compilation cost (~1,300ms → ~2ms per spec).

Toggle: VIBESPATIAL_CCCL_CACHE env var (default: enabled).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import logging
import os
import pathlib
import re
import struct as pystruct
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_FORMAT_VERSION = "v1"
_CACHE_ENV_VAR = "VIBESPATIAL_CCCL_CACHE"
_CACHE_MAGIC = b"CCCLCCH\x00"  # 8-byte file magic
_disk_cache_writes_disabled = False

# ---------------------------------------------------------------------------
# Environment / path helpers (same pattern as NVRTC disk cache)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _cccl_cache_enabled() -> bool:
    value = os.environ.get(_CACHE_ENV_VAR, "")
    if not value:
        return True
    return value.lower() not in {"0", "false", "off", "no"}


@lru_cache(maxsize=1)
def _get_cache_dir() -> pathlib.Path:
    env_dir = os.environ.get("VIBESPATIAL_CCCL_CACHE_DIR")
    if env_dir:
        return pathlib.Path(env_dir)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return pathlib.Path(xdg) / "vibespatial" / "cccl"
    return pathlib.Path.home() / ".cache" / "vibespatial" / "cccl"


@lru_cache(maxsize=1)
def _cccl_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("cuda-cccl")
    except Exception:
        return "unknown"


@lru_cache(maxsize=1)
def _compute_capability() -> tuple[int, int]:
    try:
        import cupy as cp
        dev = cp.cuda.Device()
        return dev.compute_capability
    except Exception:
        return (0, 0)


# ---------------------------------------------------------------------------
# CUDA driver API via ctypes
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _libcuda():
    """Load libcuda.so.1 for cuLibraryLoadData / cuLibraryGetKernel."""
    lib = ctypes.CDLL("libcuda.so.1")

    # CUresult cuLibraryLoadData(CUlibrary*, const void*, CUjit_option*,
    #     void**, unsigned int, CUlibraryOption*, void**, unsigned int)
    lib.cuLibraryLoadData.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # library
        ctypes.c_void_p,                   # code
        ctypes.c_void_p,                   # jitOptions (NULL)
        ctypes.c_void_p,                   # jitOptionsValues (NULL)
        ctypes.c_uint,                     # numJitOptions (0)
        ctypes.c_void_p,                   # libraryOptions (NULL)
        ctypes.c_void_p,                   # libraryOptionValues (NULL)
        ctypes.c_uint,                     # numLibraryOptions (0)
    ]
    lib.cuLibraryLoadData.restype = ctypes.c_int

    # CUresult cuLibraryGetKernel(CUkernel*, CUlibrary, const char*)
    lib.cuLibraryGetKernel.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.cuLibraryGetKernel.restype = ctypes.c_int

    # CUresult cuLibraryUnload(CUlibrary)
    lib.cuLibraryUnload.argtypes = [ctypes.c_void_p]
    lib.cuLibraryUnload.restype = ctypes.c_int

    return lib


@lru_cache(maxsize=1)
def _libc():
    """Load libc for malloc_usable_size."""
    lib = ctypes.CDLL("libc.so.6")
    lib.malloc_usable_size.argtypes = [ctypes.c_void_p]
    lib.malloc_usable_size.restype = ctypes.c_size_t
    return lib


def _cu_library_load_data(cubin_bytes: bytes) -> tuple[ctypes.c_void_p, Any]:
    """Load a CUBIN via cuLibraryLoadData.  Returns (library_handle, cubin_buffer).

    The cubin_buffer must be kept alive as long as the library is in use.
    """
    cubin_buf = ctypes.create_string_buffer(cubin_bytes)
    library = ctypes.c_void_p()
    err = _libcuda().cuLibraryLoadData(
        ctypes.byref(library), cubin_buf,
        None, None, 0, None, None, 0,
    )
    if err != 0:
        raise RuntimeError(f"cuLibraryLoadData failed with error {err}")
    return library, cubin_buf


def _cu_library_get_kernel(library: ctypes.c_void_p, name: str) -> ctypes.c_void_p:
    """Get a CUkernel handle from a CUlibrary by entry-point name."""
    kernel = ctypes.c_void_p()
    err = _libcuda().cuLibraryGetKernel(
        ctypes.byref(kernel), library, name.encode(),
    )
    if err != 0:
        raise RuntimeError(
            f"cuLibraryGetKernel failed for '{name}' with error {err}"
        )
    return kernel


# ---------------------------------------------------------------------------
# libcccl.c.parallel.so loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _libcccl():
    """Load the CCCL C parallel shared library."""
    # The library is loaded through cuda.compute bindings; find its path
    import cuda.compute._bindings as _b
    bindings_path = pathlib.Path(_b.__file__)
    # _bindings_impl.so is in cu{12,13}/, libcccl is in cu{12,13}/cccl/
    cccl_dir = bindings_path.parent / "cccl"
    so_path = cccl_dir / "libcccl.c.parallel.so"
    if not so_path.exists():
        raise FileNotFoundError(f"Cannot find libcccl at {so_path}")
    return ctypes.CDLL(str(so_path))


# ---------------------------------------------------------------------------
# ctypes struct definitions matching CCCL C headers
# ---------------------------------------------------------------------------


class CcclTypeInfo(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("alignment", ctypes.c_size_t),
        ("type", ctypes.c_int),
    ]


class CcclScanBuild(ctypes.Structure):
    _fields_ = [
        ("cc", ctypes.c_int),
        ("cubin", ctypes.c_void_p),
        ("cubin_size", ctypes.c_size_t),
        ("library", ctypes.c_void_p),
        ("accumulator_type", CcclTypeInfo),
        ("init_kernel", ctypes.c_void_p),
        ("scan_kernel", ctypes.c_void_p),
        ("force_inclusive", ctypes.c_bool),
        ("init_kind", ctypes.c_int),
        ("description_bytes_per_tile", ctypes.c_size_t),
        ("payload_bytes_per_tile", ctypes.c_size_t),
        ("runtime_policy", ctypes.c_void_p),
    ]


class CcclReduceBuild(ctypes.Structure):
    _fields_ = [
        ("cc", ctypes.c_int),
        ("cubin", ctypes.c_void_p),
        ("cubin_size", ctypes.c_size_t),
        ("library", ctypes.c_void_p),
        ("accumulator_size", ctypes.c_uint64),
        ("single_tile_kernel", ctypes.c_void_p),
        ("single_tile_second_kernel", ctypes.c_void_p),
        ("reduction_kernel", ctypes.c_void_p),
        ("nondeterministic_atomic_kernel", ctypes.c_void_p),
        ("determinism", ctypes.c_int),
        ("runtime_policy", ctypes.c_void_p),
    ]


class CcclSegReduceBuild(ctypes.Structure):
    _fields_ = [
        ("cc", ctypes.c_int),
        ("cubin", ctypes.c_void_p),
        ("cubin_size", ctypes.c_size_t),
        ("library", ctypes.c_void_p),
        ("accumulator_size", ctypes.c_uint64),
        ("segmented_reduce_kernel", ctypes.c_void_p),
        ("runtime_policy", ctypes.c_void_p),
    ]


class CcclRadixSortBuild(ctypes.Structure):
    _fields_ = [
        ("cc", ctypes.c_int),
        ("cubin", ctypes.c_void_p),
        ("cubin_size", ctypes.c_size_t),
        ("library", ctypes.c_void_p),
        ("key_type", CcclTypeInfo),
        ("value_type", CcclTypeInfo),
        ("single_tile_kernel", ctypes.c_void_p),
        ("upsweep_kernel", ctypes.c_void_p),
        ("alt_upsweep_kernel", ctypes.c_void_p),
        ("scan_bins_kernel", ctypes.c_void_p),
        ("downsweep_kernel", ctypes.c_void_p),
        ("alt_downsweep_kernel", ctypes.c_void_p),
        ("histogram_kernel", ctypes.c_void_p),
        ("exclusive_sum_kernel", ctypes.c_void_p),
        ("onesweep_kernel", ctypes.c_void_p),
        ("order", ctypes.c_int),
        ("runtime_policy", ctypes.c_void_p),
    ]


class CcclMergeSortBuild(ctypes.Structure):
    _fields_ = [
        ("cc", ctypes.c_int),
        ("cubin", ctypes.c_void_p),
        ("cubin_size", ctypes.c_size_t),
        ("library", ctypes.c_void_p),
        ("key_type", CcclTypeInfo),
        ("item_type", CcclTypeInfo),
        ("block_sort_kernel", ctypes.c_void_p),
        ("partition_kernel", ctypes.c_void_p),
        ("merge_kernel", ctypes.c_void_p),
        ("runtime_policy", ctypes.c_void_p),
    ]


class CcclUniqueByKeyBuild(ctypes.Structure):
    _fields_ = [
        ("cc", ctypes.c_int),
        ("cubin", ctypes.c_void_p),
        ("cubin_size", ctypes.c_size_t),
        ("library", ctypes.c_void_p),
        ("compact_init_kernel", ctypes.c_void_p),
        ("sweep_kernel", ctypes.c_void_p),
        ("description_bytes_per_tile", ctypes.c_size_t),
        ("payload_bytes_per_tile", ctypes.c_size_t),
        ("runtime_policy", ctypes.c_void_p),
    ]


class CcclBinarySearchBuild(ctypes.Structure):
    _fields_ = [
        ("cc", ctypes.c_int),
        ("cubin", ctypes.c_void_p),
        ("cubin_size", ctypes.c_size_t),
        ("library", ctypes.c_void_p),
        ("kernel", ctypes.c_void_p),
    ]


# Map of family name → (struct_type, kernel_field_names)
_FAMILY_STRUCTS: dict[str, tuple[type, list[str]]] = {
    "exclusive_scan": (CcclScanBuild, ["init_kernel", "scan_kernel"]),
    "reduce_into": (CcclReduceBuild, [
        "single_tile_kernel", "single_tile_second_kernel",
        "reduction_kernel", "nondeterministic_atomic_kernel",
    ]),
    "segmented_reduce": (CcclSegReduceBuild, ["segmented_reduce_kernel"]),
    "radix_sort": (CcclRadixSortBuild, [
        "single_tile_kernel", "upsweep_kernel", "alt_upsweep_kernel",
        "scan_bins_kernel", "downsweep_kernel", "alt_downsweep_kernel",
        "histogram_kernel", "exclusive_sum_kernel", "onesweep_kernel",
    ]),
    "merge_sort": (CcclMergeSortBuild, [
        "block_sort_kernel", "partition_kernel", "merge_kernel",
    ]),
    "unique_by_key": (CcclUniqueByKeyBuild, [
        "compact_init_kernel", "sweep_kernel",
    ]),
    "lower_bound": (CcclBinarySearchBuild, ["kernel"]),
    "upper_bound": (CcclBinarySearchBuild, ["kernel"]),
}


# ---------------------------------------------------------------------------
# CUBIN normalization — zero the nvJitLink session hash
# ---------------------------------------------------------------------------


def _normalize_cubin(cubin: bytes) -> bytes:
    """Zero the nvJitLink session hash for content-addressable caching.

    nvJitLink embeds ``_INTERNAL_..._XXXXXXXX_`` symbols with a unique
    8-character hex session hash per build.  We zero all occurrences to
    make the CUBIN content-addressable.
    """
    pattern = rb'_INTERNAL_\w+?_([0-9a-fA-F]{8})_'
    hashes = set(re.findall(pattern, cubin))
    if len(hashes) != 1:
        # Can't normalize (unexpected pattern), return as-is
        return cubin
    session_hash = hashes.pop()
    return cubin.replace(session_hash, b'0' * len(session_hash))


# ---------------------------------------------------------------------------
# ELF kernel name extraction (minimal pure-Python ELF parser)
# ---------------------------------------------------------------------------


def _extract_kernel_names(cubin: bytes) -> list[str]:
    """Extract global function (kernel entry-point) names from a CUBIN ELF.

    Returns names of symbols with STT_FUNC type and STB_GLOBAL binding
    that are defined (section index != SHN_UNDEF).
    """
    if len(cubin) < 64 or cubin[:4] != b'\x7fELF':
        return []

    # Parse ELF64 header
    (ei_class,) = pystruct.unpack_from('B', cubin, 4)
    if ei_class != 2:  # Must be 64-bit
        return []

    (ei_data,) = pystruct.unpack_from('B', cubin, 5)
    endian = '<' if ei_data == 1 else '>'

    e_shoff = pystruct.unpack_from(f'{endian}Q', cubin, 40)[0]
    e_shentsize = pystruct.unpack_from(f'{endian}H', cubin, 58)[0]
    e_shnum = pystruct.unpack_from(f'{endian}H', cubin, 60)[0]

    if e_shoff == 0 or e_shnum == 0:
        return []

    # Find SHT_SYMTAB section (type == 2)
    symtab_off = 0
    symtab_size = 0
    symtab_entsize = 0
    strtab_off = 0

    for i in range(e_shnum):
        sh_base = e_shoff + i * e_shentsize
        if sh_base + e_shentsize > len(cubin):
            break
        sh_type = pystruct.unpack_from(f'{endian}I', cubin, sh_base + 4)[0]
        if sh_type == 2:  # SHT_SYMTAB
            sh_offset = pystruct.unpack_from(f'{endian}Q', cubin, sh_base + 24)[0]
            sh_size = pystruct.unpack_from(f'{endian}Q', cubin, sh_base + 32)[0]
            sh_link = pystruct.unpack_from(f'{endian}I', cubin, sh_base + 40)[0]
            sh_entsize = pystruct.unpack_from(f'{endian}Q', cubin, sh_base + 56)[0]
            symtab_off = sh_offset
            symtab_size = sh_size
            symtab_entsize = sh_entsize if sh_entsize else 24

            # Get linked string table
            str_base = e_shoff + sh_link * e_shentsize
            if str_base + e_shentsize <= len(cubin):
                strtab_off = pystruct.unpack_from(f'{endian}Q', cubin, str_base + 24)[0]
            break

    if symtab_off == 0 or strtab_off == 0:
        return []

    names = []
    num_symbols = symtab_size // symtab_entsize
    for i in range(num_symbols):
        sym_base = symtab_off + i * symtab_entsize
        if sym_base + symtab_entsize > len(cubin):
            break
        st_name = pystruct.unpack_from(f'{endian}I', cubin, sym_base)[0]
        st_info = pystruct.unpack_from('B', cubin, sym_base + 4)[0]
        st_shndx = pystruct.unpack_from(f'{endian}H', cubin, sym_base + 6)[0]

        st_type = st_info & 0xF
        st_bind = st_info >> 4

        # STT_FUNC (2) + (STB_GLOBAL (1) or STB_WEAK (2)) + defined (shndx != 0)
        if st_type == 2 and st_bind in (1, 2) and st_shndx != 0:
            # Read null-terminated string from strtab
            name_start = strtab_off + st_name
            if name_start < len(cubin):
                name_end = cubin.index(b'\x00', name_start)
                name = cubin[name_start:name_end].decode('ascii', errors='replace')
                if name:
                    names.append(name)
    return names


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """Serializable cache entry for a single CCCL spec."""
    spec_name: str
    family: str
    cubin_bytes: bytes
    kernel_names: dict[str, str]  # struct_field_name → kernel_entry_name
    runtime_policy_bytes: bytes
    metadata: dict[str, Any]  # all scalar fields from build result


# ---------------------------------------------------------------------------
# Build result extraction — read from Cython object after first build
# ---------------------------------------------------------------------------


def _find_build_data_offset(build_result_obj: Any, expected_cubin_size: int) -> int:
    """Find the byte offset of build_data within a Cython DeviceBuildResult object.

    Searches the object's memory for the cubin_size value to locate the struct.
    The cubin_size field is at offset +16 in all build result structs
    (after cc(4) + pad(4) + cubin(8)), so struct_start = match_offset - 16.
    """
    obj_addr = id(build_result_obj)
    basic_size = type(build_result_obj).__basicsize__
    scan_bytes = min(basic_size + 64, 512)
    raw = bytes((ctypes.c_char * scan_bytes).from_address(obj_addr))
    target = pystruct.pack('<Q', expected_cubin_size)
    idx = raw.find(target)
    if idx < 0:
        raise ValueError(
            f"Cannot locate build_data in {type(build_result_obj).__name__} "
            f"(cubin_size={expected_cubin_size} not found in object memory)"
        )
    # cubin_size is the 3rd field: cc(4) + pad(4) + cubin_ptr(8) = offset 16
    return idx - 16


def _read_build_struct(build_result_obj: Any, struct_type: type, cubin_size: int) -> Any:
    """Read the C build_data struct from a Cython DeviceBuildResult object."""
    offset = _find_build_data_offset(build_result_obj, cubin_size)
    obj_addr = id(build_result_obj)
    return struct_type.from_address(obj_addr + offset)


def _read_runtime_policy(policy_ptr: int) -> bytes:
    """Read runtime_policy bytes using malloc_usable_size to determine size."""
    if not policy_ptr:
        return b''
    try:
        usable_size = _libc().malloc_usable_size(policy_ptr)
        if usable_size == 0 or usable_size > 4096:
            # Sanity bound — policy structs are small
            usable_size = min(usable_size, 4096) if usable_size else 256
        return bytes((ctypes.c_char * usable_size).from_address(policy_ptr))
    except Exception:
        return b''


def _map_kernel_names(
    library_handle: int,
    elf_names: list[str],
    struct_data: Any,
    kernel_fields: list[str],
) -> dict[str, str]:
    """Map kernel entry-point names to build result struct fields.

    Uses cuLibraryGetKernel on the build's own library to match handles.
    """
    # Get handles for all ELF names from the build's library
    name_to_handle: dict[str, int] = {}
    for name in elf_names:
        try:
            h = _cu_library_get_kernel(ctypes.c_void_p(library_handle), name)
            name_to_handle[name] = h.value
        except RuntimeError:
            continue

    # Match against struct fields
    mapping: dict[str, str] = {}
    for field_name in kernel_fields:
        field_handle = getattr(struct_data, field_name, 0)
        if not field_handle:
            continue
        for name, handle in name_to_handle.items():
            if handle == field_handle and name not in mapping.values():
                mapping[field_name] = name
                break

    return mapping


def extract_cache_entry(
    spec_name: str,
    family: str,
    callable_obj: Any,
) -> CacheEntry | None:
    """Extract a CacheEntry from a freshly built CCCL callable.

    callable_obj is e.g. a _Scan, _Reduce, etc. from cuda.compute.algorithms.
    """
    if family not in _FAMILY_STRUCTS:
        return None

    struct_type, kernel_fields = _FAMILY_STRUCTS[family]

    try:
        build_result = callable_obj.build_result
        cubin_bytes = build_result._get_cubin()
        if not cubin_bytes or len(cubin_bytes) < 16:
            return None

        # Read the C struct from the Cython object (auto-detect offset)
        struct_data = _read_build_struct(
            build_result, struct_type, len(cubin_bytes),
        )

        # Extract kernel names from CUBIN ELF
        elf_names = _extract_kernel_names(cubin_bytes)

        # Map names to struct fields
        kernel_mapping = _map_kernel_names(
            struct_data.library, elf_names, struct_data, kernel_fields,
        )

        if len(kernel_mapping) < len(kernel_fields):
            # Some kernels have NULL handles (valid for some families)
            # Fill missing with empty string — reconstruction will skip them
            for f in kernel_fields:
                if f not in kernel_mapping:
                    kernel_mapping[f] = ""

        # Read runtime_policy bytes
        policy_ptr = getattr(struct_data, 'runtime_policy', 0)
        policy_bytes = _read_runtime_policy(policy_ptr)

        # Collect all scalar metadata from the struct
        metadata: dict[str, Any] = {}
        for field_name, field_type in struct_type._fields_:
            if field_name in ('cubin', 'cubin_size', 'library', 'runtime_policy'):
                continue
            if field_name in kernel_fields:
                continue
            val = getattr(struct_data, field_name)
            if isinstance(val, CcclTypeInfo):
                metadata[field_name] = {
                    'size': val.size, 'alignment': val.alignment, 'type': val.type,
                }
            else:
                metadata[field_name] = int(val) if hasattr(val, '__int__') else val

        return CacheEntry(
            spec_name=spec_name,
            family=family,
            cubin_bytes=cubin_bytes,
            kernel_names=kernel_mapping,
            runtime_policy_bytes=policy_bytes,
            metadata=metadata,
        )
    except Exception:
        logger.debug("CCCL cache: extraction failed for %s", spec_name, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Build result reconstruction — from cached data to ctypes struct
# ---------------------------------------------------------------------------


class ReconstructedBuild:
    """Holds a reconstructed ctypes build result and its backing resources."""

    __slots__ = ('_refs', 'struct')

    def __init__(self, struct: Any, refs: list[Any]):
        self.struct = struct
        self._refs = refs  # prevent GC of cubin buffer, policy buffer, etc.


def reconstruct_build(entry: CacheEntry) -> ReconstructedBuild:
    """Reconstruct a build result struct from a CacheEntry."""
    struct_type, kernel_fields = _FAMILY_STRUCTS[entry.family]
    refs: list[Any] = []

    # 1. Load CUBIN → CUlibrary
    library, cubin_buf = _cu_library_load_data(entry.cubin_bytes)
    refs.append(cubin_buf)

    # 2. Get kernel handles
    kernel_handles: dict[str, int] = {}
    for field_name, kernel_name in entry.kernel_names.items():
        if kernel_name:
            h = _cu_library_get_kernel(library, kernel_name)
            kernel_handles[field_name] = h.value
        else:
            kernel_handles[field_name] = 0

    # 3. Allocate runtime_policy from cached bytes
    policy_ptr = 0
    if entry.runtime_policy_bytes:
        policy_buf = ctypes.create_string_buffer(entry.runtime_policy_bytes)
        policy_ptr = ctypes.cast(policy_buf, ctypes.c_void_p).value
        refs.append(policy_buf)

    # 4. Construct the struct
    build = struct_type()
    build.cubin = ctypes.cast(cubin_buf, ctypes.c_void_p).value
    build.cubin_size = len(entry.cubin_bytes)
    build.library = library.value

    # Set kernel handles
    for field_name, handle in kernel_handles.items():
        setattr(build, field_name, handle)

    # Set runtime_policy (if the struct has the field)
    if hasattr(build, 'runtime_policy'):
        build.runtime_policy = policy_ptr

    # Set scalar metadata
    for field_name, value in entry.metadata.items():
        if isinstance(value, dict) and 'size' in value:
            # CcclTypeInfo
            ti = CcclTypeInfo(value['size'], value['alignment'], value['type'])
            setattr(build, field_name, ti)
        else:
            setattr(build, field_name, value)

    return ReconstructedBuild(build, refs)


# ---------------------------------------------------------------------------
# Opaque ctypes wrappers for CCCL Iterator / Op / Value (sized at runtime)
# ---------------------------------------------------------------------------

_opaque_types: dict[str, type] = {}


def _get_opaque_type(name: str, size: int) -> type:
    """Get or create an opaque ctypes Structure of the given size."""
    key = f"{name}_{size}"
    if key not in _opaque_types:
        _opaque_types[key] = type(
            f"Opaque{name}{size}",
            (ctypes.Structure,),
            {"_fields_": [("_raw", ctypes.c_char * size)]},
        )
    return _opaque_types[key]


# ---------------------------------------------------------------------------
# C function setup for ctypes calls
# ---------------------------------------------------------------------------

_cfuncs: dict[str, Any] = {}


def _setup_cfunc(
    func_name: str,
    build_type: type,
    iter_type: type,
    op_type: type,
    value_type: type,
    *,
    extra_argtypes: list[Any] | None = None,
) -> Any:
    """Set up a ctypes function wrapper for a CCCL C compute function."""
    if func_name in _cfuncs:
        return _cfuncs[func_name]

    lib = _libcccl()
    func = getattr(lib, func_name)

    if extra_argtypes:
        func.argtypes = [build_type] + extra_argtypes
    # Individual families set up argtypes themselves
    func.restype = ctypes.c_int
    _cfuncs[func_name] = func
    return func


# ---------------------------------------------------------------------------
# Disk I/O — safe binary format, no pickle
#
# File layout:
#   magic      (8 bytes)  "CCCLCCH\0"
#   header_len (4 bytes)  little-endian uint32
#   header     (JSON)     spec_name, family, kernel_names, metadata,
#                         cubin_size, policy_size
#   cubin      (bytes)    raw CUBIN binary
#   policy     (bytes)    raw runtime_policy bytes
# ---------------------------------------------------------------------------


def _cache_key(spec_name: str, cubin_bytes: bytes) -> str:
    """Build a filesystem-safe cache key."""
    cc = _compute_capability()
    normalized = _normalize_cubin(cubin_bytes)
    cubin_hash = hashlib.sha256(normalized).hexdigest()[:12]
    return (
        f"{_CACHE_FORMAT_VERSION}-sm{cc[0]}{cc[1]}"
        f"-cccl{_cccl_version()}-{spec_name}-{cubin_hash}"
    )


def _serialize_cache_entry(entry: CacheEntry) -> bytes:
    """Serialize a CacheEntry to the safe binary format."""
    header = {
        "spec_name": entry.spec_name,
        "family": entry.family,
        "kernel_names": entry.kernel_names,
        "metadata": entry.metadata,
        "cubin_size": len(entry.cubin_bytes),
        "policy_size": len(entry.runtime_policy_bytes),
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return b"".join([
        _CACHE_MAGIC,
        pystruct.pack("<I", len(header_bytes)),
        header_bytes,
        entry.cubin_bytes,
        entry.runtime_policy_bytes,
    ])


def _deserialize_cache_entry(data: bytes) -> CacheEntry | None:
    """Deserialize a CacheEntry from the safe binary format.

    Returns None if the data is malformed.  No executable deserialization —
    only JSON + raw byte slicing.
    """
    if len(data) < 12 or data[:8] != _CACHE_MAGIC:
        return None
    header_len = pystruct.unpack_from("<I", data, 8)[0]
    header_end = 12 + header_len
    if header_end > len(data):
        return None
    try:
        header = json.loads(data[12:header_end])
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    cubin_size = header.get("cubin_size", 0)
    policy_size = header.get("policy_size", 0)
    cubin_start = header_end
    policy_start = cubin_start + cubin_size
    expected_len = policy_start + policy_size
    if expected_len > len(data):
        return None
    return CacheEntry(
        spec_name=header["spec_name"],
        family=header["family"],
        cubin_bytes=data[cubin_start:policy_start],
        kernel_names=header["kernel_names"],
        runtime_policy_bytes=data[policy_start:expected_len],
        metadata=header["metadata"],
    )


def _write_cache_entry(entry: CacheEntry) -> None:
    """Atomically write a CacheEntry to disk."""
    global _disk_cache_writes_disabled
    if _disk_cache_writes_disabled:
        return

    key = _cache_key(entry.spec_name, entry.cubin_bytes)
    path = _get_cache_dir() / f"{key}.cache"
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _serialize_cache_entry(entry)
        tmp.write_bytes(data)
        os.replace(str(tmp), str(path))
        logger.debug("CCCL cache: wrote %s (%d bytes)", key, len(data))
    except OSError:
        _disk_cache_writes_disabled = True
        logger.debug("CCCL disk cache: write failed, disabling writes")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _read_cache_entry(spec_name: str) -> CacheEntry | None:
    """Read a CacheEntry from disk.  Returns None on miss."""
    cache_dir = _get_cache_dir()
    if not cache_dir.exists():
        return None

    cc = _compute_capability()
    prefix = f"{_CACHE_FORMAT_VERSION}-sm{cc[0]}{cc[1]}-cccl{_cccl_version()}-{spec_name}-"

    try:
        for path in cache_dir.iterdir():
            if path.name.startswith(prefix) and path.name.endswith('.cache'):
                data = path.read_bytes()
                entry = _deserialize_cache_entry(data)
                if entry is not None and entry.spec_name == spec_name:
                    return entry
    except Exception:
        logger.debug("CCCL cache: read failed for %s", spec_name, exc_info=True)
    return None


def _delete_cache_entry(spec_name: str) -> None:
    """Delete cached entries for a spec."""
    cache_dir = _get_cache_dir()
    if not cache_dir.exists():
        return
    cc = _compute_capability()
    prefix = f"{_CACHE_FORMAT_VERSION}-sm{cc[0]}{cc[1]}-cccl{_cccl_version()}-{spec_name}-"
    try:
        for path in cache_dir.iterdir():
            if path.name.startswith(prefix) and path.name.endswith('.cache'):
                path.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# _CachedAlgorithm — drop-in replacement for CCCL _Scan/_Reduce/etc.
# ---------------------------------------------------------------------------


class _CachedAlgorithm:
    """Base for cached CCCL algorithm callables.

    Holds a reconstructed build result struct and the CCCL Python
    iterator/op/value objects needed for compute calls.

    Not thread-safe: ``__call__`` mutates shared CCCL iterator state
    before snapshotting it via ``as_bytes()``.  A per-instance lock
    serialises concurrent callers.
    """

    __slots__ = ('_build', '_call_lock', '_spec_name')

    def __init__(self, build: ReconstructedBuild, spec_name: str):
        import threading
        self._build = build
        self._call_lock = threading.Lock()
        self._spec_name = spec_name

    # Subclasses also store the build_result attribute for compatibility
    # with code that accesses callable_obj.build_result._get_cubin()
    @property
    def build_result(self):
        return self


class _CachedScanReduce(_CachedAlgorithm):
    """Cached callable for exclusive_scan and reduce families.

    Both share the same C compute signature:
      (build, void*, size_t*, iter, iter, uint64, op, value, CUstream) → CUresult
    """

    __slots__ = (
        '_cfunc',
        '_d_in_cccl',
        '_d_out_cccl',
        '_init_cccl',
        '_iter_size',
        '_op_adapter',
        '_op_cccl',
        '_op_size',
        '_value_size',
    )

    def __init__(
        self,
        build: ReconstructedBuild,
        spec_name: str,
        c_func_name: str,
        d_in_cccl: Any,
        d_out_cccl: Any,
        op_cccl: Any,
        init_cccl: Any,
        op_adapter: Any,
    ):
        super().__init__(build, spec_name)
        self._d_in_cccl = d_in_cccl
        self._d_out_cccl = d_out_cccl
        self._op_cccl = op_cccl
        self._init_cccl = init_cccl
        self._op_adapter = op_adapter

        # Determine struct sizes from the live CCCL objects
        self._iter_size = len(d_in_cccl.as_bytes())
        self._op_size = len(op_cccl.as_bytes())
        self._value_size = len(init_cccl.as_bytes())

        # Set up the C function
        IterT = _get_opaque_type("Iter", self._iter_size)
        OpT = _get_opaque_type("Op", self._op_size)
        ValT = _get_opaque_type("Value", self._value_size)
        BuildT = type(build.struct)

        lib = _libcccl()
        self._cfunc = getattr(lib, c_func_name)
        self._cfunc.argtypes = [
            BuildT,                               # build result (by value)
            ctypes.c_void_p,                       # temp_storage
            ctypes.POINTER(ctypes.c_size_t),       # temp_storage_nbytes
            IterT,                                 # d_in
            IterT,                                 # d_out
            ctypes.c_uint64,                       # num_items
            OpT,                                   # op
            ValT,                                  # init_value
            ctypes.c_void_p,                       # stream
        ]
        self._cfunc.restype = ctypes.c_int

    def __call__(
        self, temp_storage, d_in, d_out, op, num_items, init_value, stream=None,
    ):
        with self._call_lock:
            from cuda.compute._cccl_interop import (
                set_cccl_iterator_state,
                to_cccl_value_state,
            )
            from cuda.compute._utils.protocols import get_data_pointer, validate_and_get_stream
            from cuda.compute.op import make_op_adapter

            set_cccl_iterator_state(self._d_in_cccl, d_in)
            set_cccl_iterator_state(self._d_out_cccl, d_out)

            op_adapter = make_op_adapter(op)
            op_adapter.update_op_state(self._op_cccl)

            if hasattr(self._init_cccl, 'state') and init_value is not None:
                import numpy as np
                if isinstance(init_value, np.ndarray):
                    self._init_cccl.state = to_cccl_value_state(init_value)

            stream_handle = validate_and_get_stream(stream)

            if temp_storage is None:
                temp_bytes = ctypes.c_size_t(0)
                d_temp = 0
            else:
                temp_bytes = ctypes.c_size_t(temp_storage.nbytes)
                d_temp = get_data_pointer(temp_storage)

            IterT = _get_opaque_type("Iter", self._iter_size)
            OpT = _get_opaque_type("Op", self._op_size)
            ValT = _get_opaque_type("Value", self._value_size)

            err = self._cfunc(
                self._build.struct,
                ctypes.c_void_p(d_temp),
                ctypes.byref(temp_bytes),
                IterT.from_buffer_copy(self._d_in_cccl.as_bytes()),
                IterT.from_buffer_copy(self._d_out_cccl.as_bytes()),
                ctypes.c_uint64(num_items),
                OpT.from_buffer_copy(self._op_cccl.as_bytes()),
                ValT.from_buffer_copy(self._init_cccl.as_bytes()),
                ctypes.c_void_p(stream_handle) if stream_handle else ctypes.c_void_p(0),
            )
            if err != 0:
                raise RuntimeError(
                    f"CCCL cached {self._spec_name} compute failed with error {err}"
                )
            return temp_bytes.value


class _CachedSegmentedReduce(_CachedAlgorithm):
    """Cached callable for segmented_reduce family.

    C signature: (build, void*, size_t*, iter_in, iter_out, uint64,
                  iter_starts, iter_ends, op, value, CUstream) → CUresult
    """

    __slots__ = (
        '_cfunc',
        '_d_ends_cccl',
        '_d_in_cccl',
        '_d_out_cccl',
        '_d_starts_cccl',
        '_init_cccl',
        '_iter_size',
        '_op_adapter',
        '_op_cccl',
        '_op_size',
        '_value_size',
    )

    def __init__(
        self, build, spec_name, d_in_cccl, d_out_cccl,
        d_starts_cccl, d_ends_cccl, op_cccl, init_cccl, op_adapter,
    ):
        super().__init__(build, spec_name)
        self._d_in_cccl = d_in_cccl
        self._d_out_cccl = d_out_cccl
        self._d_starts_cccl = d_starts_cccl
        self._d_ends_cccl = d_ends_cccl
        self._op_cccl = op_cccl
        self._init_cccl = init_cccl
        self._op_adapter = op_adapter

        self._iter_size = len(d_in_cccl.as_bytes())
        self._op_size = len(op_cccl.as_bytes())
        self._value_size = len(init_cccl.as_bytes())

        IterT = _get_opaque_type("Iter", self._iter_size)
        OpT = _get_opaque_type("Op", self._op_size)
        ValT = _get_opaque_type("Value", self._value_size)
        BuildT = type(build.struct)

        lib = _libcccl()
        self._cfunc = lib.cccl_device_segmented_reduce
        self._cfunc.argtypes = [
            BuildT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
            IterT, IterT, ctypes.c_uint64,
            IterT, IterT, OpT, ValT, ctypes.c_void_p,
        ]
        self._cfunc.restype = ctypes.c_int

    def __call__(
        self, temp_storage, d_in, d_out, d_starts, d_ends, op,
        num_segments, init_value, stream=None,
    ):
        with self._call_lock:
            from cuda.compute._cccl_interop import set_cccl_iterator_state, to_cccl_value_state
            from cuda.compute._utils.protocols import get_data_pointer, validate_and_get_stream
            from cuda.compute.op import make_op_adapter

            set_cccl_iterator_state(self._d_in_cccl, d_in)
            set_cccl_iterator_state(self._d_out_cccl, d_out)
            set_cccl_iterator_state(self._d_starts_cccl, d_starts)
            set_cccl_iterator_state(self._d_ends_cccl, d_ends)
            op_adapter = make_op_adapter(op)
            op_adapter.update_op_state(self._op_cccl)
            import numpy as np
            if isinstance(init_value, np.ndarray):
                self._init_cccl.state = to_cccl_value_state(init_value)
            stream_handle = validate_and_get_stream(stream)

            if temp_storage is None:
                temp_bytes = ctypes.c_size_t(0)
                d_temp = 0
            else:
                temp_bytes = ctypes.c_size_t(temp_storage.nbytes)
                d_temp = get_data_pointer(temp_storage)

            IterT = _get_opaque_type("Iter", self._iter_size)
            OpT = _get_opaque_type("Op", self._op_size)
            ValT = _get_opaque_type("Value", self._value_size)

            err = self._cfunc(
                self._build.struct,
                ctypes.c_void_p(d_temp), ctypes.byref(temp_bytes),
                IterT.from_buffer_copy(self._d_in_cccl.as_bytes()),
                IterT.from_buffer_copy(self._d_out_cccl.as_bytes()),
                ctypes.c_uint64(num_segments),
                IterT.from_buffer_copy(self._d_starts_cccl.as_bytes()),
                IterT.from_buffer_copy(self._d_ends_cccl.as_bytes()),
                OpT.from_buffer_copy(self._op_cccl.as_bytes()),
                ValT.from_buffer_copy(self._init_cccl.as_bytes()),
                ctypes.c_void_p(stream_handle) if stream_handle else ctypes.c_void_p(0),
            )
            if err != 0:
                raise RuntimeError(
                    f"CCCL cached {self._spec_name} compute failed with error {err}"
                )
            return temp_bytes.value


class _CachedBinarySearch(_CachedAlgorithm):
    """Cached callable for lower_bound/upper_bound families.

    C signature: (build, iter_data, uint64, iter_values, uint64,
                  iter_out, op, CUstream) → CUresult
    No temp_storage.
    """

    __slots__ = (
        '_cfunc',
        '_d_data_cccl',
        '_d_out_cccl',
        '_d_values_cccl',
        '_iter_size',
        '_op_cccl',
        '_op_size',
    )

    def __init__(
        self, build, spec_name, d_data_cccl, d_values_cccl, d_out_cccl, op_cccl,
    ):
        super().__init__(build, spec_name)
        self._d_data_cccl = d_data_cccl
        self._d_values_cccl = d_values_cccl
        self._d_out_cccl = d_out_cccl
        self._op_cccl = op_cccl

        self._iter_size = len(d_data_cccl.as_bytes())
        self._op_size = len(op_cccl.as_bytes())

        IterT = _get_opaque_type("Iter", self._iter_size)
        OpT = _get_opaque_type("Op", self._op_size)
        BuildT = type(build.struct)

        lib = _libcccl()
        self._cfunc = lib.cccl_device_binary_search
        self._cfunc.argtypes = [
            BuildT, IterT, ctypes.c_uint64,
            IterT, ctypes.c_uint64, IterT, OpT, ctypes.c_void_p,
        ]
        self._cfunc.restype = ctypes.c_int

    def __call__(
        self, temp_storage, d_data, d_values, d_out, num_items, num_values,
        stream=None,
    ):
        with self._call_lock:
            from cuda.compute._cccl_interop import set_cccl_iterator_state
            from cuda.compute._utils.protocols import validate_and_get_stream

            set_cccl_iterator_state(self._d_data_cccl, d_data)
            set_cccl_iterator_state(self._d_values_cccl, d_values)
            set_cccl_iterator_state(self._d_out_cccl, d_out)
            stream_handle = validate_and_get_stream(stream)

            IterT = _get_opaque_type("Iter", self._iter_size)
            OpT = _get_opaque_type("Op", self._op_size)

            # Binary search temp_storage query: return 1 (needs minimal temp)
            if temp_storage is None:
                return 1

            err = self._cfunc(
                self._build.struct,
                IterT.from_buffer_copy(self._d_data_cccl.as_bytes()),
                ctypes.c_uint64(num_items),
                IterT.from_buffer_copy(self._d_values_cccl.as_bytes()),
                ctypes.c_uint64(num_values),
                IterT.from_buffer_copy(self._d_out_cccl.as_bytes()),
                OpT.from_buffer_copy(self._op_cccl.as_bytes()),
                ctypes.c_void_p(stream_handle) if stream_handle else ctypes.c_void_p(0),
            )
            if err != 0:
                raise RuntimeError(
                    f"CCCL cached {self._spec_name} compute failed with error {err}"
                )
            return 0


class _CachedRadixSort(_CachedAlgorithm):
    """Cached callable for radix_sort family.

    C signature: (build, void*, size_t*, iter_kin, iter_kout, iter_vin,
                  iter_vout, op_decomp, size_t, int, int, bool, int*, CUstream)
    """

    __slots__ = (
        '_cfunc',
        '_d_keys_in_cccl',
        '_d_keys_out_cccl',
        '_d_vals_in_cccl',
        '_d_vals_out_cccl',
        '_decomposer_cccl',
        '_iter_size',
        '_op_size',
    )

    def __init__(
        self, build, spec_name,
        d_keys_in_cccl, d_keys_out_cccl,
        d_vals_in_cccl, d_vals_out_cccl,
        decomposer_cccl,
    ):
        super().__init__(build, spec_name)
        self._d_keys_in_cccl = d_keys_in_cccl
        self._d_keys_out_cccl = d_keys_out_cccl
        self._d_vals_in_cccl = d_vals_in_cccl
        self._d_vals_out_cccl = d_vals_out_cccl
        self._decomposer_cccl = decomposer_cccl

        self._iter_size = len(d_keys_in_cccl.as_bytes())
        self._op_size = len(decomposer_cccl.as_bytes())

        IterT = _get_opaque_type("Iter", self._iter_size)
        OpT = _get_opaque_type("Op", self._op_size)
        BuildT = type(build.struct)

        lib = _libcccl()
        self._cfunc = lib.cccl_device_radix_sort
        self._cfunc.argtypes = [
            BuildT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
            IterT, IterT, IterT, IterT, OpT,
            ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
            ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
        ]
        self._cfunc.restype = ctypes.c_int

    def __call__(
        self, temp_storage, d_keys_in, d_keys_out, d_vals_in, d_vals_out,
        num_items, stream=None,
    ):
        with self._call_lock:
            from cuda.compute._cccl_interop import set_cccl_iterator_state
            from cuda.compute._utils.protocols import get_data_pointer, validate_and_get_stream

            set_cccl_iterator_state(self._d_keys_in_cccl, d_keys_in)
            set_cccl_iterator_state(self._d_keys_out_cccl, d_keys_out)
            set_cccl_iterator_state(self._d_vals_in_cccl, d_vals_in)
            set_cccl_iterator_state(self._d_vals_out_cccl, d_vals_out)
            stream_handle = validate_and_get_stream(stream)

            if temp_storage is None:
                temp_bytes = ctypes.c_size_t(0)
                d_temp = 0
            else:
                temp_bytes = ctypes.c_size_t(temp_storage.nbytes)
                d_temp = get_data_pointer(temp_storage)

            IterT = _get_opaque_type("Iter", self._iter_size)
            OpT = _get_opaque_type("Op", self._op_size)
            selector = ctypes.c_int(0)

            import numpy as np
            key_dtype = d_keys_in.dtype if hasattr(d_keys_in, 'dtype') else np.int32
            end_bit = key_dtype.itemsize * 8

            err = self._cfunc(
                self._build.struct,
                ctypes.c_void_p(d_temp), ctypes.byref(temp_bytes),
                IterT.from_buffer_copy(self._d_keys_in_cccl.as_bytes()),
                IterT.from_buffer_copy(self._d_keys_out_cccl.as_bytes()),
                IterT.from_buffer_copy(self._d_vals_in_cccl.as_bytes()),
                IterT.from_buffer_copy(self._d_vals_out_cccl.as_bytes()),
                OpT.from_buffer_copy(self._decomposer_cccl.as_bytes()),
                ctypes.c_size_t(num_items),
                ctypes.c_int(0),       # begin_bit
                ctypes.c_int(end_bit), # end_bit
                ctypes.c_bool(False),  # is_overwrite_okay
                ctypes.byref(selector),
                ctypes.c_void_p(stream_handle) if stream_handle else ctypes.c_void_p(0),
            )
            if err != 0:
                raise RuntimeError(
                    f"CCCL cached {self._spec_name} compute failed with error {err}"
                )
            return temp_bytes.value


class _CachedMergeSort(_CachedAlgorithm):
    """Cached callable for merge_sort family.

    C signature: (build, void*, size_t*, iter_kin, iter_vin, iter_kout,
                  iter_vout, uint64, op, CUstream)
    """

    __slots__ = (
        '_cfunc',
        '_d_keys_in_cccl',
        '_d_keys_out_cccl',
        '_d_vals_in_cccl',
        '_d_vals_out_cccl',
        '_iter_size',
        '_op_adapter',
        '_op_cccl',
        '_op_size',
    )

    def __init__(
        self, build, spec_name,
        d_keys_in_cccl, d_vals_in_cccl,
        d_keys_out_cccl, d_vals_out_cccl,
        op_cccl, op_adapter,
    ):
        super().__init__(build, spec_name)
        self._d_keys_in_cccl = d_keys_in_cccl
        self._d_vals_in_cccl = d_vals_in_cccl
        self._d_keys_out_cccl = d_keys_out_cccl
        self._d_vals_out_cccl = d_vals_out_cccl
        self._op_cccl = op_cccl
        self._op_adapter = op_adapter

        self._iter_size = len(d_keys_in_cccl.as_bytes())
        self._op_size = len(op_cccl.as_bytes())

        IterT = _get_opaque_type("Iter", self._iter_size)
        OpT = _get_opaque_type("Op", self._op_size)
        BuildT = type(build.struct)

        lib = _libcccl()
        self._cfunc = lib.cccl_device_merge_sort
        self._cfunc.argtypes = [
            BuildT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
            IterT, IterT, IterT, IterT,
            ctypes.c_uint64, OpT, ctypes.c_void_p,
        ]
        self._cfunc.restype = ctypes.c_int

    def __call__(
        self, temp_storage, d_keys_in, d_vals_in, d_keys_out, d_vals_out,
        op, num_items, stream=None,
    ):
        with self._call_lock:
            from cuda.compute._cccl_interop import set_cccl_iterator_state
            from cuda.compute._utils.protocols import get_data_pointer, validate_and_get_stream
            from cuda.compute.op import make_op_adapter

            set_cccl_iterator_state(self._d_keys_in_cccl, d_keys_in)
            set_cccl_iterator_state(self._d_vals_in_cccl, d_vals_in)
            set_cccl_iterator_state(self._d_keys_out_cccl, d_keys_out)
            set_cccl_iterator_state(self._d_vals_out_cccl, d_vals_out)
            op_adapter = make_op_adapter(op)
            op_adapter.update_op_state(self._op_cccl)
            stream_handle = validate_and_get_stream(stream)

            if temp_storage is None:
                temp_bytes = ctypes.c_size_t(0)
                d_temp = 0
            else:
                temp_bytes = ctypes.c_size_t(temp_storage.nbytes)
                d_temp = get_data_pointer(temp_storage)

            IterT = _get_opaque_type("Iter", self._iter_size)
            OpT = _get_opaque_type("Op", self._op_size)

            err = self._cfunc(
                self._build.struct,
                ctypes.c_void_p(d_temp), ctypes.byref(temp_bytes),
                IterT.from_buffer_copy(self._d_keys_in_cccl.as_bytes()),
                IterT.from_buffer_copy(self._d_vals_in_cccl.as_bytes()),
                IterT.from_buffer_copy(self._d_keys_out_cccl.as_bytes()),
                IterT.from_buffer_copy(self._d_vals_out_cccl.as_bytes()),
                ctypes.c_uint64(num_items),
                OpT.from_buffer_copy(self._op_cccl.as_bytes()),
                ctypes.c_void_p(stream_handle) if stream_handle else ctypes.c_void_p(0),
            )
            if err != 0:
                raise RuntimeError(
                    f"CCCL cached {self._spec_name} compute failed with error {err}"
                )
            return temp_bytes.value


class _CachedUniqueByKey(_CachedAlgorithm):
    """Cached callable for unique_by_key family.

    C signature: (build, void*, size_t*, iter_kin, iter_vin, iter_kout,
                  iter_vout, iter_count, op, size_t, CUstream)
    """

    __slots__ = (
        '_cfunc',
        '_d_count_cccl',
        '_d_keys_in_cccl',
        '_d_keys_out_cccl',
        '_d_vals_in_cccl',
        '_d_vals_out_cccl',
        '_iter_size',
        '_op_adapter',
        '_op_cccl',
        '_op_size',
    )

    def __init__(
        self, build, spec_name,
        d_keys_in_cccl, d_vals_in_cccl,
        d_keys_out_cccl, d_vals_out_cccl, d_count_cccl,
        op_cccl, op_adapter,
    ):
        super().__init__(build, spec_name)
        self._d_keys_in_cccl = d_keys_in_cccl
        self._d_vals_in_cccl = d_vals_in_cccl
        self._d_keys_out_cccl = d_keys_out_cccl
        self._d_vals_out_cccl = d_vals_out_cccl
        self._d_count_cccl = d_count_cccl
        self._op_cccl = op_cccl
        self._op_adapter = op_adapter

        self._iter_size = len(d_keys_in_cccl.as_bytes())
        self._op_size = len(op_cccl.as_bytes())

        IterT = _get_opaque_type("Iter", self._iter_size)
        OpT = _get_opaque_type("Op", self._op_size)
        BuildT = type(build.struct)

        lib = _libcccl()
        self._cfunc = lib.cccl_device_unique_by_key
        self._cfunc.argtypes = [
            BuildT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
            IterT, IterT, IterT, IterT, IterT, OpT,
            ctypes.c_size_t, ctypes.c_void_p,
        ]
        self._cfunc.restype = ctypes.c_int

    def __call__(
        self, temp_storage, d_keys_in, d_vals_in, d_keys_out, d_vals_out,
        d_count, op, num_items, stream=None,
    ):
        with self._call_lock:
            from cuda.compute._cccl_interop import set_cccl_iterator_state
            from cuda.compute._utils.protocols import get_data_pointer, validate_and_get_stream
            from cuda.compute.op import make_op_adapter

            set_cccl_iterator_state(self._d_keys_in_cccl, d_keys_in)
            set_cccl_iterator_state(self._d_vals_in_cccl, d_vals_in)
            set_cccl_iterator_state(self._d_keys_out_cccl, d_keys_out)
            set_cccl_iterator_state(self._d_vals_out_cccl, d_vals_out)
            set_cccl_iterator_state(self._d_count_cccl, d_count)
            op_adapter = make_op_adapter(op)
            op_adapter.update_op_state(self._op_cccl)
            stream_handle = validate_and_get_stream(stream)

            if temp_storage is None:
                temp_bytes = ctypes.c_size_t(0)
                d_temp = 0
            else:
                temp_bytes = ctypes.c_size_t(temp_storage.nbytes)
                d_temp = get_data_pointer(temp_storage)

            IterT = _get_opaque_type("Iter", self._iter_size)
            OpT = _get_opaque_type("Op", self._op_size)

            err = self._cfunc(
                self._build.struct,
                ctypes.c_void_p(d_temp), ctypes.byref(temp_bytes),
                IterT.from_buffer_copy(self._d_keys_in_cccl.as_bytes()),
                IterT.from_buffer_copy(self._d_vals_in_cccl.as_bytes()),
                IterT.from_buffer_copy(self._d_keys_out_cccl.as_bytes()),
                IterT.from_buffer_copy(self._d_vals_out_cccl.as_bytes()),
                IterT.from_buffer_copy(self._d_count_cccl.as_bytes()),
                OpT.from_buffer_copy(self._op_cccl.as_bytes()),
                ctypes.c_size_t(num_items),
                ctypes.c_void_p(stream_handle) if stream_handle else ctypes.c_void_p(0),
            )
            if err != 0:
                raise RuntimeError(
                    f"CCCL cached {self._spec_name} compute failed with error {err}"
                )
            return temp_bytes.value


    # ---------------------------------------------------------------------------
    # Public API — called from cccl_precompile.py
    # ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """Whether the CCCL CUBIN disk cache is enabled."""
    return _cccl_cache_enabled()


def try_load_cached(spec_name: str, family: str) -> CacheEntry | None:
    """Try to load a cached entry for the given spec.  Returns None on miss."""
    if not _cccl_cache_enabled():
        return None
    if family not in _FAMILY_STRUCTS:
        return None
    return _read_cache_entry(spec_name)


def save_after_build(
    spec_name: str,
    family: str,
    callable_obj: Any,
) -> None:
    """Extract and save a cache entry after a successful CCCL build."""
    if not _cccl_cache_enabled():
        return
    if family not in _FAMILY_STRUCTS:
        return
    entry = extract_cache_entry(spec_name, family, callable_obj)
    if entry is not None:
        _write_cache_entry(entry)


def _cached_spec_name_set() -> frozenset[str]:
    """Return all spec names with cache files on disk for the current CC/CCCL version.

    Scans the cache directory once and extracts spec names from filenames.
    The result is NOT lru_cached because callers (CCCLPrecompiler.request)
    may need to re-probe after a precompile_all() populates the cache.
    However, the underlying helpers (_compute_capability, _cccl_version,
    _get_cache_dir) are all @lru_cache'd, so repeated calls are cheap.
    """
    if not _cccl_cache_enabled():
        return frozenset()
    cache_dir = _get_cache_dir()
    if not cache_dir.exists():
        return frozenset()
    cc = _compute_capability()
    prefix = f"{_CACHE_FORMAT_VERSION}-sm{cc[0]}{cc[1]}-cccl{_cccl_version()}-"
    suffix = ".cache"
    names: set[str] = set()
    try:
        for path in cache_dir.iterdir():
            fname = path.name
            if fname.startswith(prefix) and fname.endswith(suffix):
                # Filename: {prefix}{spec_name}-{cubin_hash_12}.cache
                rest = fname[len(prefix):-len(suffix)]
                # spec_name is everything before the last '-' (the hash)
                dash_idx = rest.rfind("-")
                if dash_idx > 0:
                    names.add(rest[:dash_idx])
    except OSError:
        pass
    return frozenset(names)


def is_cached(spec_name: str) -> bool:
    """Quick check: does a cache file exist on disk for this spec?

    Uses _cached_spec_name_set() which scans the directory once.
    """
    return spec_name in _cached_spec_name_set()


def clear_cache() -> int:
    """Delete all CCCL CUBIN cache files.  Returns count of files removed."""
    cache_dir = _get_cache_dir()
    if not cache_dir.exists():
        return 0
    count = 0
    try:
        for path in cache_dir.iterdir():
            if path.suffix == '.cache':
                path.unlink(missing_ok=True)
                count += 1
    except OSError:
        pass
    return count


def cache_stats() -> dict[str, Any]:
    """Return cache directory, file count, and total bytes."""
    cache_dir = _get_cache_dir()
    if not cache_dir.exists():
        return {
            "directory": str(cache_dir),
            "file_count": 0,
            "total_bytes": 0,
            "enabled": _cccl_cache_enabled(),
        }
    files = [f for f in cache_dir.iterdir() if f.suffix == '.cache']
    return {
        "directory": str(cache_dir),
        "file_count": len(files),
        "total_bytes": sum(f.stat().st_size for f in files),
        "enabled": _cccl_cache_enabled(),
    }
