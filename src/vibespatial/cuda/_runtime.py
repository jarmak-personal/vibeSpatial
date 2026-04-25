from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import pathlib
import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Any, TypeAlias

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

try:
    from cuda.bindings import driver as cu
    from cuda.bindings import nvrtc
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cu = None
    nvrtc = None

try:
    import rmm
    import rmm.mr
    from rmm.allocators.cupy import rmm_cupy_allocator
except ImportError:
    rmm = None
    rmm_cupy_allocator = None


def _require_bindings() -> None:
    if cu is None:
        raise RuntimeError("cuda-python driver bindings are not installed")


def _require_gpu_arrays() -> None:
    if cp is None:
        raise RuntimeError("CuPy is not installed; canonical GPU array support is unavailable")


def _free_cupy_pinned_memory_pool() -> None:
    if cp is None:
        return
    pool = cp.get_default_pinned_memory_pool()
    if pool is not None:
        pool.free_all_blocks()


def eager_pool_trim_enabled() -> bool:
    """Return whether hot paths should eagerly flush the GPU memory pool.

    This is intentionally opt-in. Calling ``free_all_blocks()`` on the CuPy pool
    can introduce large synchronization overhead in overlay/clip hot paths.
    Keep the default fast and expose eager trimming as an escape hatch for
    memory-pressure debugging or emergency OOM mitigation.
    """
    value = os.environ.get("VIBESPATIAL_EAGER_GPU_POOL_TRIM", "")
    if not value:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


def maybe_trim_pool_memory(runtime: Any | None = None) -> None:
    """Best-effort eager pool trim when explicitly enabled by environment."""
    if not eager_pool_trim_enabled():
        return
    try:
        target = runtime if runtime is not None else get_cuda_runtime()
        target.free_pool_memory()
    except Exception:
        pass


def _check_driver(result: tuple[Any, ...]) -> tuple[Any, ...]:
    err = result[0]
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {err.name}")
    return result[1:]


def _check_nvrtc(result: tuple[Any, ...]) -> tuple[Any, ...]:
    err = result[0]
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC call failed: {nvrtc.nvrtcGetErrorString(err)}")
    return result[1:]


@lru_cache(maxsize=1)
def cuda_device_count() -> int:
    if cu is None:
        return 0
    try:
        _check_driver(cu.cuInit(0))
        count, = _check_driver(cu.cuDeviceGetCount())
    except Exception:
        return 0
    return int(count)


def has_cuda_device() -> bool:
    return cuda_device_count() > 0


def has_nvrtc_support() -> bool:
    return nvrtc is not None


DeviceArray: TypeAlias = Any


# ---------------------------------------------------------------------------
# D2H transfer counter — lightweight instrumentation for zero-copy audits
# ---------------------------------------------------------------------------

_d2h_transfer_lock = threading.Lock()
_d2h_transfer_count: int = 0
_d2h_transfer_bytes: int = 0
_d2h_transfer_seconds: float = 0.0


@dataclass(frozen=True)
class RuntimeD2HTransferEvent:
    trigger: str
    reason: str
    item_count: int
    bytes_transferred: int
    elapsed_seconds: float
    pipeline: str | None = None
    dataset: str | None = None
    stage: str | None = None
    stage_category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger,
            "reason": self.reason,
            "item_count": self.item_count,
            "bytes_transferred": self.bytes_transferred,
            "elapsed_seconds": self.elapsed_seconds,
            "pipeline": self.pipeline,
            "dataset": self.dataset,
            "stage": self.stage,
            "stage_category": self.stage_category,
        }


_d2h_transfer_events: deque[RuntimeD2HTransferEvent] = deque(maxlen=1024)


def _device_array_size(device_array: Any) -> int:
    size = getattr(device_array, "size", 0)
    try:
        return int(size)
    except (TypeError, ValueError):
        return 0


def _device_array_nbytes(device_array: Any, host_array: np.ndarray | None = None) -> int:
    if host_array is not None:
        return int(host_array.nbytes)
    nbytes = getattr(device_array, "nbytes", 0)
    try:
        return int(nbytes)
    except (TypeError, ValueError):
        return 0


def _increment_d2h_transfer_count(
    bytes_transferred: int = 0,
    elapsed_seconds: float = 0.0,
) -> None:
    """Increment the global D2H transfer counter (thread-safe)."""
    global _d2h_transfer_count, _d2h_transfer_bytes, _d2h_transfer_seconds
    with _d2h_transfer_lock:
        _d2h_transfer_count += 1
        _d2h_transfer_bytes += max(int(bytes_transferred), 0)
        _d2h_transfer_seconds += max(float(elapsed_seconds), 0.0)


def _record_runtime_d2h_transfer_event(
    *,
    trigger: str,
    reason: str,
    item_count: int,
    bytes_transferred: int,
    elapsed_seconds: float,
) -> RuntimeD2HTransferEvent:
    context = None
    try:
        from vibespatial.runtime.materialization import current_materialization_context

        context = current_materialization_context()
    except Exception:
        context = None
    event = RuntimeD2HTransferEvent(
        trigger=trigger,
        reason=reason,
        item_count=max(int(item_count), 0),
        bytes_transferred=max(int(bytes_transferred), 0),
        elapsed_seconds=max(float(elapsed_seconds), 0.0),
        pipeline=None if context is None else context.pipeline,
        dataset=None if context is None else context.dataset,
        stage=None if context is None else context.stage,
        stage_category=None if context is None else context.stage_category,
    )
    with _d2h_transfer_lock:
        _d2h_transfer_events.append(event)
    return event


def _notify_runtime_d2h_transfer(
    device_array: Any,
    *,
    host_array: np.ndarray | None = None,
    trigger: str,
    reason: str,
    elapsed_seconds: float = 0.0,
) -> None:
    bytes_transferred = _device_array_nbytes(device_array, host_array)
    item_count = _device_array_size(device_array)
    _increment_d2h_transfer_count(bytes_transferred, elapsed_seconds)
    _record_runtime_d2h_transfer_event(
        trigger=trigger,
        reason=reason,
        item_count=item_count,
        bytes_transferred=bytes_transferred,
        elapsed_seconds=elapsed_seconds,
    )
    try:
        from vibespatial.runtime.execution_trace import notify_transfer

        notify_transfer(
            direction="d2h",
            trigger=trigger,
            reason=reason,
            source="cuda_runtime",
            item_count=item_count,
            bytes_transferred=bytes_transferred,
            elapsed_seconds=elapsed_seconds,
        )
    except Exception:
        # Transfer accounting must never make a production copy fail.
        pass


def get_d2h_transfer_count() -> int:
    """Return the current D2H transfer count."""
    with _d2h_transfer_lock:
        return _d2h_transfer_count


def get_d2h_transfer_bytes() -> int:
    """Return bytes copied from device to host through the CUDA runtime."""
    with _d2h_transfer_lock:
        return _d2h_transfer_bytes


def get_d2h_transfer_seconds() -> float:
    """Return wall time spent in synchronous CUDA-runtime D2H copies."""
    with _d2h_transfer_lock:
        return _d2h_transfer_seconds


def get_d2h_transfer_stats() -> tuple[int, int]:
    """Return ``(count, bytes)`` for CUDA-runtime D2H copies."""
    with _d2h_transfer_lock:
        return _d2h_transfer_count, _d2h_transfer_bytes


def get_d2h_transfer_profile() -> tuple[int, int, float]:
    """Return ``(count, bytes, seconds)`` for CUDA-runtime D2H copies."""
    with _d2h_transfer_lock:
        return _d2h_transfer_count, _d2h_transfer_bytes, _d2h_transfer_seconds


def get_d2h_transfer_events(*, clear: bool = False) -> list[RuntimeD2HTransferEvent]:
    """Return CUDA-runtime D2H transfer events for profile attribution."""
    with _d2h_transfer_lock:
        events = list(_d2h_transfer_events)
        if clear:
            _d2h_transfer_events.clear()
        return events


def reset_d2h_transfer_count() -> None:
    """Reset the D2H transfer profile counters and event log to zero."""
    global _d2h_transfer_count, _d2h_transfer_bytes, _d2h_transfer_seconds
    with _d2h_transfer_lock:
        _d2h_transfer_count = 0
        _d2h_transfer_bytes = 0
        _d2h_transfer_seconds = 0.0
        _d2h_transfer_events.clear()


@contextmanager
def assert_zero_d2h_transfers():
    """Assert no D2H transfers occur in the block.

    Records the transfer count at entry and verifies it has not changed
    at exit.  Raises ``AssertionError`` if any ``copy_device_to_host``
    calls occurred inside the block.

    Usage::

        with assert_zero_d2h_transfers():
            result = overlay(df1, df2)
        assert result.geometry.values._owned.residency == Residency.DEVICE

    This is complementary to the higher-level
    ``execution_trace.assert_no_transfers()`` context manager, which tracks
    transfers via the trace subsystem.  This counter operates at the CUDA
    runtime level and catches transfers that bypass the trace system.
    """
    with _d2h_transfer_lock:
        start_count = _d2h_transfer_count
    yield
    with _d2h_transfer_lock:
        end_count = _d2h_transfer_count
    if end_count != start_count:
        raise AssertionError(
            f"Expected zero D2H transfers, but {end_count - start_count} "
            f"occurred (counter went from {start_count} to {end_count})"
        )


# ---------------------------------------------------------------------------
# NVRTC disk cache — persists compiled CUBIN across process restarts
# ---------------------------------------------------------------------------

_NVRTC_CACHE_FORMAT_VERSION = "v2"
_disk_cache_writes_disabled = False


@lru_cache(maxsize=1)
def _nvrtc_version() -> tuple[int, int]:
    """Get NVRTC compiler version (cached)."""
    if nvrtc is None:
        return (0, 0)
    major, minor = _check_nvrtc(nvrtc.nvrtcVersion())
    return (int(major), int(minor))


@lru_cache(maxsize=1)
def _disk_cache_enabled() -> bool:
    """Check if NVRTC disk cache is enabled via environment."""
    value = os.environ.get("VIBESPATIAL_NVRTC_CACHE", "")
    if not value:
        return True
    return value.lower() not in {"0", "false", "off", "no"}


@lru_cache(maxsize=1)
def _get_cache_dir() -> pathlib.Path:
    """Return the disk cache directory for NVRTC CUBIN files."""
    env_dir = os.environ.get("VIBESPATIAL_NVRTC_CACHE_DIR")
    if env_dir:
        return pathlib.Path(env_dir)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return pathlib.Path(xdg) / "vibespatial" / "nvrtc"
    return pathlib.Path.home() / ".cache" / "vibespatial" / "nvrtc"


def _disk_cache_key(
    cache_key: str,
    compute_cap: tuple[int, int],
    options: tuple[str, ...],
    nvrtc_ver: tuple[int, int],
) -> str:
    """Build a filesystem-safe cache key for disk CUBIN storage."""
    parts = [
        _NVRTC_CACHE_FORMAT_VERSION,
        f"sm{compute_cap[0]}{compute_cap[1]}",
        f"nvrtc{nvrtc_ver[0]}.{nvrtc_ver[1]}",
        cache_key,
    ]
    if options:
        opts_hash = hashlib.sha1("|".join(sorted(options)).encode()).hexdigest()[:8]
        parts.append(f"opts-{opts_hash}")
    return "-".join(parts)


def _read_cached_cubin(disk_key: str) -> bytes | None:
    """Read cached CUBIN from disk. Returns None on miss or error."""
    path = _get_cache_dir() / f"{disk_key}.cubin"
    try:
        data = path.read_bytes()
        if len(data) < 16:  # too small to be valid CUBIN
            path.unlink(missing_ok=True)
            return None
        return data
    except OSError:
        return None


def _write_cached_cubin(disk_key: str, cubin: bytes) -> None:
    """Atomically write CUBIN to disk cache. Errors silently disable writes."""
    global _disk_cache_writes_disabled
    if _disk_cache_writes_disabled:
        return
    path = _get_cache_dir() / f"{disk_key}.cubin"
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(cubin)
        os.replace(str(tmp), str(path))
    except OSError:
        _disk_cache_writes_disabled = True
        logger.debug("NVRTC disk cache: write failed, disabling writes for this process")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _delete_cached_cubin(disk_key: str) -> None:
    """Delete a corrupt cache file."""
    path = _get_cache_dir() / f"{disk_key}.cubin"
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _make_oom_callback(max_retries: int = 3):
    """Create a thread-safe OOM callback with bounded retry counter.

    RMM's ``FailureCallbackResourceAdaptor`` calls this in a loop for a
    single failing allocation.  The counter bounds retries *per allocation
    attempt*.  A time-based reset ensures that independent OOM events
    (separated by >1 s of successful allocations) each get the full retry
    budget, preventing the counter from silently eroding over the lifetime
    of the process.
    """
    import time as _time

    lock = threading.Lock()
    retry_state = {"count": 0, "last_ts": 0.0}

    def _callback(nbytes: int) -> bool:
        with lock:
            now = _time.monotonic()
            # Reset for a fresh OOM event (>1 s since last callback call)
            if now - retry_state["last_ts"] > 1.0:
                retry_state["count"] = 0
            retry_state["last_ts"] = now
            if retry_state["count"] >= max_retries:
                retry_state["count"] = 0
                return False  # give up, let OOM propagate
            retry_state["count"] += 1
        import gc
        gc.collect()
        # Also free pinned memory — pinned allocations can hold
        # significant host+device resources.
        _free_cupy_pinned_memory_pool()
        return True  # tell RMM to retry the allocation

    return _callback


@dataclass(frozen=True, slots=True)
class CudaStream:
    """Lightweight wrapper around a CUDA stream handle."""

    handle: Any  # CUstream from cuda.bindings.driver

    def __cuda_stream__(self) -> tuple[int, int]:
        """Expose the CUDA Array Interface stream protocol."""
        return (0, int(self.handle))

    def synchronize(self) -> None:
        """Block the host until all operations on this stream complete."""
        _check_driver(cu.cuStreamSynchronize(self.handle))


def _normalize_stream_handle(stream: Any | None) -> Any | None:
    """Return a driver-compatible CUDA stream handle.

    Accepts repo-native ``CudaStream`` objects plus foreign stream objects
    that implement the CUDA stream protocol or expose a raw ``ptr``.
    """
    if stream is None:
        return None
    if isinstance(stream, CudaStream):
        return stream.handle
    handle = getattr(stream, "handle", None)
    if handle is not None:
        return handle
    protocol = getattr(stream, "__cuda_stream__", None)
    if callable(protocol):
        version, handle = protocol()
        if int(version) != 0:
            raise ValueError(
                f"unsupported CUDA stream protocol version {version!r}; expected 0"
            )
        return cu.CUstream(int(handle))
    pointer = getattr(stream, "ptr", None)
    if pointer is not None:
        return cu.CUstream(int(pointer))
    raise TypeError(
        "stream must be None, a CudaStream, or expose __cuda_stream__()/ptr"
    )


@dataclass(frozen=True, slots=True)
class CompiledKernel:
    name: str
    function: Any


class CudaDriverRuntime:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._context: Any | None = None
        self._device: Any | None = None
        self._compute_capability: tuple[int, int] | None = None
        self._fp64_to_fp32_ratio: float | None = None
        self._driver_cuda_version: tuple[int, int] = (0, 0)
        self._module_cache: dict[str, dict[str, Any]] = {}
        self._module_cache_lock = threading.Lock()
        self._block_size_cache: dict[int, int] = {}
        self._memory_pool: Any | None = None
        self._memory_backend: str = "none"
        self._rmm_mr = None
        self._rmm_configured: bool = False
        self._configure_memory_pool()

    def _configure_memory_pool(self) -> None:
        """Configure GPU memory pool. RMM setup is deferred until CUDA context exists."""
        if cp is None:
            return
        if rmm is not None:
            # Defer RMM setup — needs CUDA context (established in _ensure_context)
            self._rmm_configured = False
            return
        # No RMM: use CuPy pool immediately (no CUDA context needed)
        self._configure_cupy_pool()

    def _configure_cupy_pool(self) -> None:
        """Fallback: CuPy MemoryPool (current behavior)."""
        pool = cp.cuda.MemoryPool()
        pool_limit = os.environ.get("VIBESPATIAL_GPU_POOL_LIMIT")
        if pool_limit:
            pool.set_limit(size=int(pool_limit))
        cp.cuda.set_allocator(pool.malloc)
        self._memory_pool = pool
        self._memory_backend = "cupy"

    def _configure_rmm_pool(self) -> None:
        """Configure RMM tiered memory resource.

        Tier selection (highest priority first):

        * **Tier C** — ``VIBESPATIAL_GPU_MANAGED_MEMORY=1``: bare managed
          memory (automatic page migration, no pool).
        * **Tier A** — ``VIBESPATIAL_GPU_OOM_SAFETY=0``: raw pool, no
          failure callback.  Faster allocation but OOM is fatal.
        * **Tier B** (default) — pool + ``FailureCallbackResourceAdaptor``
          that runs ``gc.collect()`` on allocation failure, giving Python
          a chance to free unreferenced CuPy arrays and return blocks to
          the pool before the OOM propagates.
        """
        pool_limit_str = os.environ.get("VIBESPATIAL_GPU_POOL_LIMIT")
        pool_limit = int(pool_limit_str) if pool_limit_str else None
        managed = os.environ.get("VIBESPATIAL_GPU_MANAGED_MEMORY", "").strip()
        oom_safety = os.environ.get("VIBESPATIAL_GPU_OOM_SAFETY", "").strip()

        if managed in ("1", "true", "yes"):
            # Tier C: bare managed memory (no pool wrapping)
            mr = rmm.mr.ManagedMemoryResource()
            self._memory_backend = "rmm-managed"
            logger.info("RMM memory backend: managed memory (Tier C)")
        elif oom_safety in ("0", "false", "no"):
            # Tier A: raw pool, no OOM callback (opt-in)
            base = rmm.mr.CudaMemoryResource()
            mr = rmm.mr.PoolMemoryResource(
                base,
                initial_pool_size=0,
                maximum_pool_size=pool_limit if pool_limit else None,
            )
            self._memory_backend = "rmm-pool"
            logger.info("RMM memory backend: pool without OOM safety (Tier A)")
        else:
            # Tier B: pool + OOM callback (default)
            base = rmm.mr.CudaMemoryResource()
            pool = rmm.mr.PoolMemoryResource(
                base,
                initial_pool_size=0,
                maximum_pool_size=pool_limit if pool_limit else None,
            )
            callback = _make_oom_callback(max_retries=3)
            mr = rmm.mr.FailureCallbackResourceAdaptor(pool, callback)
            self._memory_backend = "rmm-safe"
            logger.info("RMM memory backend: pool with OOM safety (Tier B, default)")

        rmm.mr.set_current_device_resource(mr)
        cp.cuda.set_allocator(rmm_cupy_allocator)
        self._memory_pool = None  # CuPy pool is not used
        self._rmm_mr = mr  # prevent GC of resource chain

    def memory_pool_stats(self) -> dict[str, int]:
        """Return current memory pool statistics.

        The returned dict always contains ``used_bytes`` and ``total_bytes``
        when stats are available.  Additional backend-specific keys:

        * CuPy pool: ``free_bytes``
        * RMM backends: ``peak_bytes``, ``total_allocations``
          (requires ``rmm.statistics.enable_statistics()`` to be active)
        """
        if self._memory_backend == "none":
            return {}
        if self._memory_backend == "cupy":
            if self._memory_pool is None:
                return {}
            pool = self._memory_pool
            return {
                "used_bytes": pool.used_bytes(),
                "total_bytes": pool.total_bytes(),
                "free_bytes": pool.free_bytes(),
            }

        if rmm is None:
            return {}

        stats: dict[str, int] = {}
        try:
            from rmm import statistics as rmm_stats
            s = rmm_stats.get_statistics()
            if s is not None:
                stats["used_bytes"] = int(s.current_bytes)
                stats["peak_bytes"] = int(s.peak_bytes)
                stats["total_bytes"] = int(s.total_bytes)
                stats["total_allocations"] = int(s.total_count)
        except (ImportError, AttributeError):
            pass

        return stats

    def free_pool_memory(self) -> None:
        """Release cached memory back to the device.

        With CuPy pool: releases all cached blocks to the CUDA driver.
        With RMM pool: runs ``gc.collect()`` to ensure Python-side
        references to dead CuPy arrays are cleaned up, returning their
        underlying device blocks to the RMM pool for reuse.  The RMM
        pool itself does not shrink, but freed blocks become available
        for subsequent allocations.
        In both cases, clears the pinned memory pool if present.
        """
        if self._memory_backend == "cupy":
            if self._memory_pool is not None:
                self._memory_pool.free_all_blocks()
        elif self._memory_backend.startswith("rmm"):
            # RMM pool coalesces freed blocks internally but Python GC
            # must run first so that unreferenced CuPy arrays actually
            # return their allocations to the pool.
            import gc
            gc.collect()
        # Clear pinned memory pool (separate from device pool)
        _free_cupy_pinned_memory_pool()

    def available(self) -> bool:
        return has_cuda_device() and cp is not None

    def _ensure_context(self) -> Any:
        _require_bindings()
        if not self.available():
            raise RuntimeError("GPU execution was requested, but no CUDA device is available")
        with self._lock:
            if self._context is not None:
                return self._context
            _check_driver(cu.cuInit(0))
            device, = _check_driver(cu.cuDeviceGet(0))
            context, = _check_driver(cu.cuDevicePrimaryCtxRetain(device))
            major, = _check_driver(
                cu.cuDeviceGetAttribute(
                    cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    device,
                )
            )
            minor, = _check_driver(
                cu.cuDeviceGetAttribute(
                    cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    device,
                )
            )
            try:
                fp64_ratio_raw, = _check_driver(
                    cu.cuDeviceGetAttribute(
                        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO,
                        device,
                    )
                )
                fp64_ratio = 1.0 / float(int(fp64_ratio_raw)) if int(fp64_ratio_raw) > 0 else 1.0 / 32.0
            except Exception:
                fp64_ratio = 1.0 / 32.0  # conservative consumer default
            # Query driver CUDA version for NVRTC compatibility diagnostics
            driver_version = 0
            try:
                dv, = _check_driver(cu.cuDriverGetVersion())
                driver_version = int(dv)
            except Exception:
                pass
            self._driver_cuda_version = (
                (driver_version // 1000, (driver_version % 1000) // 10)
                if driver_version > 0
                else (0, 0)
            )
            self._device = device
            self._context = context
            self._compute_capability = (int(major), int(minor))
            self._fp64_to_fp32_ratio = fp64_ratio

            # Warn if NVRTC major version exceeds driver capability.
            # CUBIN compilation avoids PTX JIT so this is non-fatal,
            # but worth flagging for diagnostics.
            nvrtc_ver = _nvrtc_version()
            if nvrtc_ver[0] > 0 and self._driver_cuda_version[0] > 0:
                if nvrtc_ver[0] > self._driver_cuda_version[0]:
                    logger.warning(
                        "NVRTC %d.%d is newer than the CUDA driver (%d.%d). "
                        "vibespatial compiles to CUBIN (native machine code) so "
                        "this is safe, but PTX-based workflows would fail. "
                        "Consider upgrading your NVIDIA driver or installing "
                        "a cuda-python version matching your driver.",
                        nvrtc_ver[0],
                        nvrtc_ver[1],
                        self._driver_cuda_version[0],
                        self._driver_cuda_version[1],
                    )

            # Deferred RMM setup — requires active CUDA context
            if rmm is not None and not self._rmm_configured:
                try:
                    self._configure_rmm_pool()
                except Exception:
                    logger.warning(
                        "RMM pool setup failed; falling back to CuPy memory pool",
                        exc_info=True,
                    )
                    self._configure_cupy_pool()
                self._rmm_configured = True

            return self._context

    @property
    def compute_capability(self) -> tuple[int, int]:
        self._ensure_context()
        assert self._compute_capability is not None
        return self._compute_capability

    @property
    def fp64_to_fp32_ratio(self) -> float:
        """Return fp64:fp32 throughput ratio from hardware (e.g., 0.5 for A100, 1/64 for RTX 4090)."""
        self._ensure_context()
        assert self._fp64_to_fp32_ratio is not None
        return self._fp64_to_fp32_ratio

    @property
    def driver_cuda_version(self) -> tuple[int, int]:
        """Return the max CUDA version supported by the installed driver, e.g. ``(13, 0)``."""
        self._ensure_context()
        return self._driver_cuda_version

    @contextmanager
    def activate(self):
        context = self._ensure_context()
        _check_driver(cu.cuCtxPushCurrent(context))
        try:
            yield self
        finally:
            _check_driver(cu.cuCtxPopCurrent())

    def allocate(self, shape: tuple[int, ...], dtype: np.dtype[Any], *, zero: bool = False) -> DeviceArray:
        _require_gpu_arrays()
        normalized_shape = tuple(int(dim) for dim in shape)
        normalized_dtype = np.dtype(dtype)
        if not normalized_shape:
            raise ValueError("device arrays must have at least one dimension")
        with self.activate():
            if zero:
                return cp.zeros(normalized_shape, dtype=normalized_dtype)
            return cp.empty(normalized_shape, dtype=normalized_dtype)

    def from_host(self, host_array: np.ndarray | DeviceArray) -> DeviceArray:
        _require_gpu_arrays()
        with self.activate():
            if hasattr(host_array, "__cuda_array_interface__"):
                return cp.asarray(host_array)
            host = np.ascontiguousarray(host_array)
            return cp.asarray(host)

    def copy_host_to_device(self, host_array: np.ndarray, device_array: DeviceArray) -> None:
        _require_gpu_arrays()
        host = np.ascontiguousarray(host_array)
        with self.activate():
            device_array[...] = cp.asarray(host)

    def copy_device_to_host(self, device_array: DeviceArray, host_array: np.ndarray | None = None) -> np.ndarray:
        _require_gpu_arrays()
        started = perf_counter()
        with self.activate():
            host = cp.asnumpy(device_array)
        elapsed = perf_counter() - started
        _notify_runtime_d2h_transfer(
            device_array,
            host_array=host if host_array is None else host_array,
            trigger="cuda-runtime-copy",
            reason="CudaRuntime.copy_device_to_host",
            elapsed_seconds=elapsed,
        )
        if host_array is None:
            return host
        host_array[...] = host
        return host_array

    def pointer(self, device_array: DeviceArray | None) -> int:
        if device_array is None:
            return 0
        return int(device_array.data.ptr)

    def free(self, device_array: DeviceArray | None) -> None:
        if device_array is None:
            return
        memory = getattr(getattr(device_array, "data", None), "mem", None)
        if memory is not None:
            try:
                memory.free()
            except Exception:
                return

    def synchronize(self) -> None:
        with self.activate():
            _check_driver(cu.cuCtxSynchronize())

    # ------------------------------------------------------------------
    # Stream management
    # ------------------------------------------------------------------

    def create_stream(self) -> CudaStream:
        """Create a new CUDA stream for concurrent execution.

        The returned stream can be passed to :meth:`launch`,
        :meth:`copy_device_to_host_async`, and other stream-aware methods
        to enable overlap of kernel execution and data transfers.
        """
        with self.activate():
            stream, = _check_driver(cu.cuStreamCreate(0))
        return CudaStream(handle=stream)

    def destroy_stream(self, stream: CudaStream) -> None:
        """Destroy a CUDA stream.

        The caller must ensure all work submitted to the stream has
        completed before calling this method.
        """
        with self.activate():
            _check_driver(cu.cuStreamDestroy(stream.handle))

    @contextmanager
    def stream_context(self):
        """Create a stream, yield it, and destroy it on exit.

        The stream is automatically synchronised before destruction::

            with runtime.stream_context() as stream:
                runtime.launch(kernel, ..., stream=stream)
                # stream.synchronize() called automatically
        """
        stream = self.create_stream()
        try:
            yield stream
        finally:
            stream.synchronize()
            self.destroy_stream(stream)

    # ------------------------------------------------------------------
    # Async transfer helpers
    # ------------------------------------------------------------------

    def copy_device_to_host_async(
        self,
        device_array: DeviceArray,
        stream: Any,
        host_array: np.ndarray | None = None,
    ) -> np.ndarray:
        """Enqueue an asynchronous device-to-host copy on *stream*.

        If *host_array* is ``None`` a new contiguous host array is
        allocated.  For true asynchronous behaviour the destination
        should be pinned memory (see :meth:`allocate_pinned`).

        The caller **must** synchronise the stream before reading the
        returned array.
        """
        _require_gpu_arrays()
        stream_handle = _normalize_stream_handle(stream)
        with self.activate():
            if host_array is None:
                host_array = np.empty(device_array.shape, dtype=device_array.dtype)
            nbytes = host_array.nbytes
            _check_driver(cu.cuMemcpyDtoHAsync(
                host_array.ctypes.data, int(device_array.data.ptr), nbytes, stream_handle,
            ))
        _notify_runtime_d2h_transfer(
            device_array,
            host_array=host_array,
            trigger="cuda-runtime-copy-async",
            reason="CudaRuntime.copy_device_to_host_async",
        )
        return host_array

    def copy_device_to_device_async(
        self,
        source_array: DeviceArray,
        destination_array: DeviceArray,
        stream: Any,
    ) -> None:
        """Enqueue an asynchronous device-to-device copy on *stream*."""
        _require_gpu_arrays()
        source_nbytes = int(getattr(source_array, "nbytes", 0))
        destination_nbytes = int(getattr(destination_array, "nbytes", 0))
        if source_nbytes != destination_nbytes:
            raise ValueError(
                "device-to-device copy requires equal byte sizes "
                f"({source_nbytes} != {destination_nbytes})"
            )
        stream_handle = _normalize_stream_handle(stream)
        with self.activate():
            _check_driver(cu.cuMemcpyDtoDAsync(
                int(destination_array.data.ptr),
                int(source_array.data.ptr),
                source_nbytes,
                stream_handle,
            ))

    def copy_host_to_device_async(
        self,
        host_array: np.ndarray,
        device_array: DeviceArray,
        stream: Any,
    ) -> None:
        """Enqueue an asynchronous host-to-device copy on *stream*.

        For true asynchronous behaviour *host_array* should reside in
        pinned memory (see :meth:`allocate_pinned`).

        The caller **must** synchronise the stream before reusing or
        freeing either buffer.
        """
        _require_gpu_arrays()
        host = np.ascontiguousarray(host_array)
        stream_handle = _normalize_stream_handle(stream)
        with self.activate():
            nbytes = host.nbytes
            _check_driver(cu.cuMemcpyHtoDAsync(
                int(device_array.data.ptr), host.ctypes.data, nbytes, stream_handle,
            ))

    def allocate_pinned(self, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
        """Allocate page-locked (pinned) host memory for async transfers.

        Returns a normal :class:`numpy.ndarray` backed by CUDA
        pinned memory.  Pinned memory enables truly asynchronous
        DMA transfers that overlap with kernel execution.
        """
        _require_gpu_arrays()
        n_elements = int(np.prod(shape))
        nbytes = n_elements * np.dtype(dtype).itemsize
        mem = cp.cuda.alloc_pinned_memory(nbytes)
        # frombuffer may see the full allocation (which can be larger than
        # requested due to driver alignment); slice to the exact count.
        arr = np.frombuffer(mem, dtype=dtype)[:n_elements]
        return arr.reshape(shape)

    def launch(
        self,
        kernel: CompiledKernel,
        *,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        params: tuple[tuple[Any, ...], tuple[Any, ...]],
        shared_mem_bytes: int = 0,
        stream: Any | None = None,
    ) -> None:
        stream_handle = _normalize_stream_handle(stream)
        with self.activate():
            _check_driver(
                cu.cuLaunchKernel(
                    kernel.function,
                    int(grid[0]),
                    int(grid[1]),
                    int(grid[2]),
                    int(block[0]),
                    int(block[1]),
                    int(block[2]),
                    int(shared_mem_bytes),
                    stream_handle,
                    params,
                    0,
                )
            )

    def optimal_block_size(
        self,
        kernel: CompiledKernel,
        shared_mem_bytes: int = 0,
    ) -> int:
        """Return optimal threads-per-block for *kernel* using CUDA occupancy API."""
        cache_key = id(kernel.function)
        cached = self._block_size_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            with self.activate():
                _min_grid, block_size = _check_driver(
                    cu.cuOccupancyMaxPotentialBlockSize(
                        kernel.function, 0, shared_mem_bytes, 0
                    )
                )
            result = int(block_size)
        except (AttributeError, RuntimeError):
            result = 256
        self._block_size_cache[cache_key] = result
        return result

    def launch_config(
        self,
        kernel: CompiledKernel,
        item_count: int,
        shared_mem_bytes: int = 0,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Return ``(grid, block)`` tuples sized for *item_count* work-items."""
        block_size = self.optimal_block_size(kernel, shared_mem_bytes)
        grid_x = max(1, (item_count + block_size - 1) // block_size)
        return (grid_x, 1, 1), (block_size, 1, 1)

    def compile_kernels(
        self,
        *,
        cache_key: str,
        source: str,
        kernel_names: tuple[str, ...],
        options: tuple[str, ...] = (),
    ) -> dict[str, CompiledKernel]:
        # Fast path: GIL-atomic dict read, no lock needed.
        if cache_key in self._module_cache:
            return self._module_cache[cache_key]
        if nvrtc is None:
            raise RuntimeError("NVRTC is not installed; cuda-python kernel compilation is unavailable")

        # Double-checked locking: acquire lock only on cache miss.
        with self._module_cache_lock:
            if cache_key in self._module_cache:
                return self._module_cache[cache_key]
            with self.activate():
                cc = self.compute_capability
                arch = f"--gpu-architecture=sm_{cc[0]}{cc[1]}"
                compile_options = tuple(options) + (arch,)

                # --- Disk cache probe ---
                cubin = None
                disk_key = None
                disk_cache_hit = False
                if _disk_cache_enabled():
                    disk_key = _disk_cache_key(cache_key, cc, options, _nvrtc_version())
                    cubin = _read_cached_cubin(disk_key)
                    disk_cache_hit = cubin is not None

                if cubin is None:
                    # NVRTC compilation (expensive path: 20-400ms)
                    program_name = f"{cache_key}.cu".encode()
                    program, = _check_nvrtc(
                        nvrtc.nvrtcCreateProgram(source.encode(), program_name, 0, [], [])
                    )
                    encoded_options = [option.encode() for option in compile_options]
                    result = nvrtc.nvrtcCompileProgram(program, len(encoded_options), encoded_options)
                    if result[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                        log_size, = _check_nvrtc(nvrtc.nvrtcGetProgramLogSize(program))
                        log = bytearray(log_size)
                        _check_nvrtc(nvrtc.nvrtcGetProgramLog(program, log))
                        raise RuntimeError(bytes(log).decode(errors="replace"))
                    cubin_size, = _check_nvrtc(nvrtc.nvrtcGetCUBINSize(program))
                    cubin = bytearray(cubin_size)
                    _check_nvrtc(nvrtc.nvrtcGetCUBIN(program, cubin))
                    cubin = bytes(cubin)

                    # Write to disk cache
                    if disk_key is not None:
                        _write_cached_cubin(disk_key, cubin)

                # Load CUBIN into driver module
                try:
                    module, = _check_driver(cu.cuModuleLoadData(cubin))
                except RuntimeError:
                    if disk_cache_hit:
                        # Corrupt cache file — delete and recompile
                        _delete_cached_cubin(disk_key)
                        return self._compile_kernels_no_disk_cache(
                            cache_key=cache_key, source=source,
                            kernel_names=kernel_names, options=options,
                        )
                    raise

                kernels = {
                    name: CompiledKernel(
                        name=name,
                        function=_check_driver(cu.cuModuleGetFunction(module, name.encode()))[0],
                    )
                    for name in kernel_names
                }
            self._module_cache[cache_key] = kernels
            return kernels

    def _compile_kernels_no_disk_cache(
        self,
        *,
        cache_key: str,
        source: str,
        kernel_names: tuple[str, ...],
        options: tuple[str, ...] = (),
    ) -> dict[str, CompiledKernel]:
        """Fallback compilation without disk cache (used on corrupt cache recovery)."""
        cc = self.compute_capability
        arch = f"--gpu-architecture=sm_{cc[0]}{cc[1]}"
        compile_options = tuple(options) + (arch,)
        program_name = f"{cache_key}.cu".encode()
        program, = _check_nvrtc(
            nvrtc.nvrtcCreateProgram(source.encode(), program_name, 0, [], [])
        )
        encoded_options = [option.encode() for option in compile_options]
        result = nvrtc.nvrtcCompileProgram(program, len(encoded_options), encoded_options)
        if result[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            log_size, = _check_nvrtc(nvrtc.nvrtcGetProgramLogSize(program))
            log = bytearray(log_size)
            _check_nvrtc(nvrtc.nvrtcGetProgramLog(program, log))
            raise RuntimeError(bytes(log).decode(errors="replace"))
        cubin_size, = _check_nvrtc(nvrtc.nvrtcGetCUBINSize(program))
        cubin = bytearray(cubin_size)
        _check_nvrtc(nvrtc.nvrtcGetCUBIN(program, cubin))
        cubin = bytes(cubin)
        module, = _check_driver(cu.cuModuleLoadData(cubin))
        kernels = {
            name: CompiledKernel(
                name=name,
                function=_check_driver(cu.cuModuleGetFunction(module, name.encode()))[0],
            )
            for name in kernel_names
        }
        self._module_cache[cache_key] = kernels
        return kernels

_CUDA_RUNTIME = CudaDriverRuntime()


def get_cuda_runtime() -> CudaDriverRuntime:
    return _CUDA_RUNTIME


def count_scatter_total(
    runtime: CudaDriverRuntime,
    device_counts: DeviceArray,
    device_offsets: DeviceArray,
) -> int:
    """Get total output size from count-scatter arrays.

    Packs the final count and final offset into one tiny device buffer,
    then performs a single D2H copy on a dedicated stream.  This keeps the
    unavoidable allocation scalar fence to one host transfer event instead
    of two independent last-element copies.
    """
    dtype = np.dtype(device_counts.dtype)
    if np.dtype(device_offsets.dtype) != dtype:
        raise TypeError("count_scatter_total requires counts and offsets with matching dtypes")
    with runtime.stream_context() as xfer:
        d_buf = runtime.allocate((2,), dtype)
        h_buf = runtime.allocate_pinned((2,), dtype)
        runtime.copy_device_to_device_async(device_counts[-1:], d_buf[:1], xfer)
        runtime.copy_device_to_device_async(device_offsets[-1:], d_buf[1:], xfer)
        runtime.copy_device_to_host_async(d_buf, xfer, h_buf)
    return int(h_buf[0]) + int(h_buf[1])


def count_scatter_totals(
    runtime: CudaDriverRuntime,
    count_offset_pairs: list[tuple[DeviceArray, DeviceArray]],
) -> list[int]:
    """Get multiple count-scatter totals with one transfer-stream sync.

    For same-dtype pairs, packs every final count/final offset pair into a
    compact device buffer and performs one D2H copy for the whole batch.
    Mixed dtypes fall back to the single-total helper per pair.
    """
    if not count_offset_pairs:
        return []

    dtypes = [
        np.dtype(device_counts.dtype)
        for device_counts, _device_offsets in count_offset_pairs
    ]
    offset_dtypes = [
        np.dtype(device_offsets.dtype)
        for _device_counts, device_offsets in count_offset_pairs
    ]
    if any(offset_dtype != dtype for dtype, offset_dtype in zip(dtypes, offset_dtypes, strict=True)):
        raise TypeError("count_scatter_totals requires counts and offsets with matching dtypes")
    if any(dtype != dtypes[0] for dtype in dtypes):
        return [
            count_scatter_total(runtime, device_counts, device_offsets)
            for device_counts, device_offsets in count_offset_pairs
        ]

    dtype = dtypes[0]
    n_pairs = len(count_offset_pairs)
    with runtime.stream_context() as xfer:
        d_buf = runtime.allocate((2 * n_pairs,), dtype)
        h_buf = runtime.allocate_pinned((2 * n_pairs,), dtype)
        for pair_index, (device_counts, device_offsets) in enumerate(count_offset_pairs):
            base = 2 * pair_index
            runtime.copy_device_to_device_async(device_counts[-1:], d_buf[base:base + 1], xfer)
            runtime.copy_device_to_device_async(device_offsets[-1:], d_buf[base + 1:base + 2], xfer)
        runtime.copy_device_to_host_async(d_buf, xfer, h_buf)
    return [
        int(h_buf[2 * pair_index]) + int(h_buf[2 * pair_index + 1])
        for pair_index in range(n_pairs)
    ]


def count_scatter_total_with_transfer(
    runtime: CudaDriverRuntime,
    device_counts: DeviceArray,
    device_offsets: DeviceArray,
    *,
    precomputed_total: int | None = None,
) -> tuple[int, CudaStream, np.ndarray]:
    """Get total and start async full-counts transfer on a background stream.

    Returns ``(total_verts, xfer_stream, pinned_host_counts)``.  The
    caller **must** call ``xfer_stream.synchronize()`` before reading
    *pinned_host_counts*, and ``runtime.destroy_stream(xfer_stream)``
    when done.  The transfer overlaps with subsequent null-stream
    kernel launches (e.g. the scatter pass).
    """
    dtype = np.dtype(device_counts.dtype)
    n = int(device_counts.size)

    # 1. Get total via single-sync async transfer of last elements.
    total = (
        int(precomputed_total)
        if precomputed_total is not None
        else count_scatter_total(runtime, device_counts, device_offsets)
    )

    # 2. Start async full-array transfer on a dedicated stream.
    xfer = runtime.create_stream()
    h_counts = runtime.allocate_pinned((n,), dtype)
    runtime.copy_device_to_host_async(device_counts, xfer, h_counts)

    return total, xfer, h_counts


def compile_kernel_group(name: str, source: str, kernel_names: tuple[str, ...]):
    """Compile a named group of NVRTC kernels, using the per-runtime cache."""
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(name, source)
    return runtime.compile_kernels(cache_key=cache_key, source=source, kernel_names=kernel_names)


def _compile_precision_kernel(
    name_prefix: str,
    fp64_source: str,
    fp32_source: str,
    kernel_names: tuple[str, ...],
    compute_type: str = "double",
):
    """Compile the fp64 or fp32 kernel variant for a named kernel group."""
    source = fp64_source if compute_type == "double" else fp32_source
    suffix = "fp64" if compute_type == "double" else "fp32"
    return compile_kernel_group(f"{name_prefix}-{suffix}", source, kernel_names)


def make_kernel_cache_key(prefix: str, source: str) -> str:
    digest = hashlib.sha1(source.encode()).hexdigest()
    return f"{prefix}-{digest}"


def _nvrtc_cached_key_set() -> frozenset[str]:
    """Return all disk cache keys with CUBIN files on disk.

    Scans the NVRTC cache directory once and returns the set of
    disk_key stems (filename without .cubin suffix).  Not lru_cached
    because callers may need to re-probe after compilation populates
    the cache; the underlying _get_cache_dir() is already cached.
    """
    if not _disk_cache_enabled():
        return frozenset()
    cache_dir = _get_cache_dir()
    if not cache_dir.exists():
        return frozenset()
    keys: set[str] = set()
    try:
        for path in cache_dir.iterdir():
            if path.suffix == ".cubin":
                keys.add(path.stem)
    except OSError:
        pass
    return frozenset(keys)


def nvrtc_is_cached(
    cache_key: str,
    compute_cap: tuple[int, int],
    options: tuple[str, ...] = (),
) -> bool:
    """Check if an NVRTC CUBIN exists on disk for the given cache key.

    Cheap: file-existence only, no deserialization.
    Requires compute_cap to be passed in (avoids requiring CUDA context
    initialization just for the probe).
    """
    if not _disk_cache_enabled():
        return False
    disk_key = _disk_cache_key(cache_key, compute_cap, options, _nvrtc_version())
    return disk_key in _nvrtc_cached_key_set()


def clear_nvrtc_cache() -> int:
    """Delete all cached NVRTC CUBIN files from disk.

    Returns the number of files removed.  Also cleans up any legacy
    ``.ptx`` files left over from the v1 cache format.
    """
    cache_dir = _get_cache_dir()
    removed = 0
    try:
        for path in cache_dir.glob("*.cubin"):
            path.unlink()
            removed += 1
        # Clean up legacy v1 PTX cache files
        for path in cache_dir.glob("*.ptx"):
            path.unlink()
            removed += 1
    except OSError:
        pass
    return removed


def nvrtc_cache_stats() -> dict[str, int | str]:
    """Return disk cache location and size statistics."""
    cache_dir = _get_cache_dir()
    files = list(cache_dir.glob("*.cubin")) if cache_dir.exists() else []
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        "directory": str(cache_dir),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "enabled": _disk_cache_enabled(),
    }


KERNEL_PARAM_PTR = ctypes.c_void_p
KERNEL_PARAM_I32 = ctypes.c_int
KERNEL_PARAM_I64 = ctypes.c_longlong
KERNEL_PARAM_F64 = ctypes.c_double
