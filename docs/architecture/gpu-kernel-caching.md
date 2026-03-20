# GPU Kernel Caching

<!-- DOC_HEADER:START
Scope: CCCL and NVRTC CUBIN on-disk caching, precompilation, ctypes replay, and cache management.
Read If: You are changing kernel compilation, caching, precompilation, JIT warmup, or CCCL/NVRTC integration.
STOP IF: Your task is docs-only or limited to vendored test maintenance.
Source Of Truth: Disk caching architecture for CCCL algorithm CUBINs and NVRTC kernel CUBINs.
Body Budget: 220/260 lines
Document: docs/architecture/gpu-kernel-caching.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-4 | Preamble |
| 5-10 | Request Signals |
| 11-15 | Open First |
| 16-18 | Verify |
| 19-22 | Risks |
| 23-40 | Two-tier caching architecture |
| 41-54 | Timing summary |
| 55-80 | CCCL CUBIN disk cache (how it works) |
| 81-90 | Why ctypes replay |
| 91-100 | CUBIN normalization |
| 101-115 | Cache key format and contents |
| 116-130 | Struct definitions |
| 131-142 | Families not cached |
| 143-155 | NVRTC CUBIN disk cache |
| 156-170 | Environment variables |
| 171-185 | Pre-compilation API |
| 186-195 | Cache management and source files |

Request Signals: kernel caching, CUBIN cache, precompile, warmup, JIT, CCCL cache, NVRTC cache, disk cache, precompile_all, startup latency

Open First:
- docs/architecture/gpu-kernel-caching.md
- src/vibespatial/cuda/cccl_cubin_cache.py
- src/vibespatial/cuda/cccl_precompile.py

Verify:
- uv run pytest tests/test_cccl_cubin_cache.py tests/test_cccl_precompile.py -q
- uv run python -c "from vibespatial.cuda.cccl_cubin_cache import cache_stats; print(cache_stats())"

Risks:
- ctypes struct layout mismatch if CCCL changes C ABI without version bump
- runtime_policy bytes may be invalid if policy format changes within a CCCL version
- cache file format uses JSON + raw bytes (no pickle / no executable deserialization)
DOC_HEADER:END -->

vibeSpatial JIT-compiles two families of GPU code at runtime:

- **NVRTC kernels** -- custom CUDA C kernels compiled via NVRTC (spatial
  query, point-in-polygon, overlay, bounds, WKB encode, etc.)
- **CCCL algorithms** -- CUB-based primitives compiled via CCCL's
  `make_*` API (scan, reduce, radix sort, merge sort, binary search,
  unique-by-key, segmented reduce, select, segmented sort)

Both families use on-disk CUBIN caches so the JIT cost is paid once per
install, not once per Python process.

## Two-tier caching architecture

```
Process start
  |
  |-- CCCL precompiler (8 threads)
  |     For each spec:
  |       1. Check CCCL CUBIN disk cache  (~/.cache/vibespatial/cccl/)
  |       2. Hit:  cuLibraryLoadData (1-40 ms)  --> ready
  |       3. Miss: CCCL make_* build (1,300-9,000 ms) --> extract CUBIN --> save to disk
  |
  |-- NVRTC precompiler (16 threads)
  |     For each unit:
  |       1. Check NVRTC CUBIN disk cache  (~/.cache/vibespatial/nvrtc/)
  |       2. Hit:  cuModuleLoadData (50-200 ms)  --> ready
  |       3. Miss: nvrtcCompileProgram (80-150 ms) --> save to disk
  |
  V
Pipeline execution (all kernels warm)
```

### Timing summary

| Scenario | CCCL (21 specs) | NVRTC (61 units) | Total wall |
|----------|----------------|-------------------|------------|
| Cold (no cache) | ~3,400 ms | ~200-400 ms | ~3.5 s |
| Warm (disk hit, lazy) | **~0.1 ms request** | **~0.1 ms request** | **~0.2 ms at import** |
| In-memory hit | 0.04 ms | <0.01 ms | instant |

With lazy warmup, disk-cached specs are deferred at `request_warmup()` time
and loaded on first `get_compiled()` call (~2 ms/spec).  No thread pool is
created when all specs are cached.  See "Lazy warmup" below.

## CCCL CUBIN disk cache

This is the novel component. CCCL has no built-in on-disk cache -- each
Python process re-runs NVRTC + nvJitLink on first use of each algorithm
spec. The CCCL CUBIN cache eliminates this by intercepting the build
result and replaying it via ctypes on subsequent starts.

### How it works

**First run (cache miss):**

1. `CCCLPrecompiler._compile_one()` calls `algorithms.make_exclusive_scan()`
   which triggers CCCL's C build function (NVRTC compile + nvJitLink, ~1,300 ms)
2. After the build, `extract_cache_entry()` reads the C `build_result` struct
   directly from the Cython object's memory via ctypes
3. It extracts: the compiled CUBIN bytes (`_get_cubin()`), kernel entry-point
   names (parsed from the CUBIN's ELF symbol table), runtime policy bytes
   (via `malloc_usable_size`), and all scalar metadata fields
4. The CUBIN is normalized (nvJitLink session hash zeroed) for
   content-addressable keying, then the entry is atomically written to disk

**Subsequent runs (cache hit):**

1. `_compile_one()` checks the disk cache before calling CCCL
2. On hit, `reconstruct_build()` loads the cached CUBIN via
   `cuLibraryLoadData` (~1-2 ms), gets kernel handles via
   `cuLibraryGetKernel`, restores the runtime policy, and populates
   a ctypes replica of the C build result struct
3. A `_CachedScan` (or `_CachedReduce`, etc.) wrapper is constructed with
   the same `__call__` protocol as CCCL's `_Scan`/`_Reduce`
4. On each compute call, iterator/op/value arguments are serialized via
   CCCL's `as_bytes()` method into opaque ctypes buffers, and the C
   compute function (`cccl_device_exclusive_scan()`, etc.) is called
   directly through `libcccl.c.parallel.so` via ctypes

**Fallback:** any exception during cache load silently falls back to the
standard CCCL build at zero additional cost.

### Why ctypes replay?

The CCCL C API functions (`cccl_device_exclusive_scan()`, etc.) take the
build result struct **by value** and handle all dispatch logic internally
(grid/block sizing, argument marshaling, CUB dispatch). We do not
replicate CUB internals -- we call the same C function CCCL uses, just
with a struct we populated from cache instead of from NVRTC.

The build result structs (defined in `cccl/c/scan.h`, `cccl/c/reduce.h`,
etc.) contain: compute capability, CUBIN pointer, CUlibrary handle,
CUkernel handles, runtime policy pointer, and algorithm-specific metadata
(accumulator type, tile sizes, sort order, etc.).

### CUBIN normalization

nvJitLink embeds `_INTERNAL_..._XXXXXXXX_` symbols with a unique 8-char
hex session hash per build. This hash differs between builds even when the
source is identical. We zero all occurrences (always exactly one unique
hash, at ~72 positions) to produce a content-addressable CUBIN. The
normalized CUBIN loads correctly -- the hash is in the ELF string table,
not in code sections.

### Cache key format

```
v1-sm{CC}-cccl{VERSION}-{spec_name}-{normalized_cubin_sha256_12}.cache
```

Example: `v1-sm89-cccl0.5.1-exclusive_scan_i32-dd7dbbd47276.cache`

Components: format version, compute capability, CCCL package version,
spec name, and truncated SHA-256 of the normalized CUBIN. A CCCL version
change automatically invalidates the entire cache.

### Cache file format

Each `.cache` file uses a safe binary format with no executable
deserialization (no pickle):

```
Offset  Size         Content
0       8            Magic: "CCCLCCH\0"
8       4            header_len (little-endian uint32)
12      header_len   JSON header (UTF-8)
12+N    cubin_size   Raw CUBIN bytes
...     policy_size  Raw runtime_policy bytes
```

The JSON header contains: `spec_name`, `family`, `kernel_names` (dict
mapping struct field names to ELF entry-point names), `metadata` (all
scalar fields: cc, tile sizes, accumulator type, etc.), `cubin_size`,
and `policy_size`.  The two large binary blobs (CUBIN and policy) are
appended as raw bytes after the header, referenced by size fields in
the JSON.  This avoids any code execution surface while keeping the
format self-describing and trivially auditable.

### Struct definitions

ctypes `Structure` subclasses mirror the CCCL C headers exactly. Sizes
are validated at extraction time by locating the `cubin_size` field in
the Cython object's memory.

| Family | C struct | sizeof | Kernels |
|--------|----------|--------|---------|
| Scan | `cccl_device_scan_build_result_t` | 104 | 2 |
| Reduce | `cccl_device_reduce_build_result_t` | 88 | 4 |
| SegmentedReduce | `cccl_device_segmented_reduce_build_result_t` | 56 | 1 |
| RadixSort | `cccl_device_radix_sort_build_result_t` | 168 | 9 |
| MergeSort | `cccl_device_merge_sort_build_result_t` | 112 | 3 |
| UniqueByKey | `cccl_device_unique_by_key_build_result_t` | 72 | 2 |
| BinarySearch | `cccl_device_binary_search_build_result_t` | 40 | 1 |

### Families not cached

- **Select** -- uses `DeviceThreeWayPartitionBuildResult` with Numba-compiled
  predicate LTOIR embedded in the build. The predicate's state arrays make
  caching non-trivial.
- **SegmentedSort** -- build result struct embeds `cccl_op_t` sub-structs
  with LTOIR code pointers that require special serialization.

Both fall through to the standard CCCL build path with no performance
regression.

## NVRTC CUBIN disk cache

The NVRTC disk cache (`cuda_runtime.py`) is simpler -- NVRTC produces a
standard CUBIN that can be loaded directly via `cuModuleLoadData`. This
cache predates the CCCL cache and uses the same patterns:

- **Cache key:** `v2-sm{CC}-nvrtc{VER}-{source_hash}[-opts-{hash}].cubin`
- **Atomic writes:** temp file + `os.replace()`
- **Corruption recovery:** if `cuModuleLoadData` fails on a cached CUBIN,
  the file is deleted and the kernel is recompiled
- **Location:** `~/.cache/vibespatial/nvrtc/`

## Lazy warmup for disk-cached specs

When `request_warmup()` or `request_nvrtc_warmup()` is called at module
scope, each spec/unit is probed against the disk cache before any work
is submitted:

1. **Batch probe:** `_cached_spec_name_set()` (CCCL) or
   `_nvrtc_cached_key_set()` (NVRTC) scans the cache directory once and
   returns the set of cached names.  The underlying helpers
   (`_compute_capability`, `_cccl_version`, `_get_cache_dir`) are all
   `@lru_cache`'d, so repeated probes are cheap.
2. **Defer on hit:** specs with disk cache entries are added to a
   `_deferred_disk` set.  No thread pool task is created for them.
3. **Lazy load:** on the first `get_compiled()` call (CCCL) or
   `compile_kernels()` call (NVRTC), the deferred spec is loaded from
   disk synchronously (~2 ms), then cached in memory.
4. **No thread pool if all cached:** the `ThreadPoolExecutor` is created
   lazily.  If every spec is deferred, no threads are spawned.

`ensure_warm()` / `ensure_pipelines_warm()` handle deferred specs
correctly -- they trigger lazy loads before waiting on futures.

The `status()` dict includes a `"deferred"` count alongside `"compiled"`
and `"pending"`.

## Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `VIBESPATIAL_CCCL_CACHE` | enabled | Set `0`/`false`/`off`/`no` to disable CCCL CUBIN disk cache |
| `VIBESPATIAL_NVRTC_CACHE` | enabled | Set `0`/`false`/`off`/`no` to disable NVRTC CUBIN disk cache |
| `VIBESPATIAL_CCCL_CACHE_DIR` | `~/.cache/vibespatial/cccl` | Override CCCL cache directory |
| `VIBESPATIAL_NVRTC_CACHE_DIR` | `~/.cache/vibespatial/nvrtc` | Override NVRTC cache directory |
| `VIBESPATIAL_PRECOMPILE` | enabled | Set `0` to disable background pre-compilation entirely |

All variables respect `XDG_CACHE_HOME` when the `_DIR` override is not set.

## Pre-compilation API

```python
from vibespatial.cuda.cccl_precompile import precompile_all

# Compile everything and block until done (CI warm-up, post-install)
result = precompile_all(timeout=120.0)
# {'cccl': {'compiled': 12, 'submitted': 21, ...},
#  'nvrtc': {'compiled': 61, 'submitted': 61, ...},
#  'cccl_cold': [], 'nvrtc_cold': []}
```

For demand-driven warmup (the default):

```python
from vibespatial.cuda.cccl_precompile import request_warmup, ensure_pipelines_warm

# Non-blocking: request specific specs (typically called at module scope)
request_warmup(["exclusive_scan_i32", "radix_sort_i32_i32"])

# Blocking: wait for all requested compilations before pipeline execution
cold = ensure_pipelines_warm(timeout=60.0)
```

## Cache management

```python
from vibespatial.cuda.cccl_cubin_cache import clear_cache, cache_stats
from vibespatial.cuda._runtime import clear_nvrtc_cache, nvrtc_cache_stats

# Inspect
print(cache_stats())       # CCCL: file count, total bytes, directory
print(nvrtc_cache_stats()) # NVRTC: file count, total bytes, directory

# Clear (e.g. after CUDA driver upgrade)
clear_cache()              # CCCL
clear_nvrtc_cache()        # NVRTC
```

## Key risks and mitigations

**ctypes struct layout mismatch:** if CCCL changes its C struct fields
between versions, the cached build result will have the wrong layout. This
is mitigated by including the CCCL version in the cache key (version change
= automatic cache miss) and by validating the `cubin_size` field offset at
extraction time.

**Runtime policy changes:** the runtime policy struct is an opaque
allocation whose size we read via `malloc_usable_size`. If the policy
format changes, the restored bytes may be invalid. The CCCL version key
handles this for inter-version changes; intra-version changes would
require a cache clear.

**Pointer invalidation:** kernel handles and library handles are
process-specific. On cache hit we call `cuLibraryLoadData` and
`cuLibraryGetKernel` to obtain fresh handles for the current process.

## Source files

| File | Role |
|------|------|
| `src/vibespatial/cuda/cccl_cubin_cache.py` | CCCL CUBIN cache: ctypes structs, extraction, reconstruction, disk I/O, cached algorithm wrappers |
| `src/vibespatial/cuda/cccl_precompile.py` | CCCL precompiler singleton, cache integration, `precompile_all()` |
| `src/vibespatial/cuda/_runtime.py` | NVRTC disk cache, CUDA driver runtime |
| `src/vibespatial/cuda/nvrtc_precompile.py` | NVRTC precompiler singleton |
| `tests/test_cccl_cubin_cache.py` | Cache unit tests (no GPU required) |
| `tests/test_cccl_precompile.py` | Precompiler unit tests |
| `tests/test_nvrtc_disk_cache.py` | NVRTC cache unit tests |
