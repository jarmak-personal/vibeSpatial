"""Demand-driven background NVRTC kernel compilation.

ADR-0034, Level 2: Each kernel module declares its (prefix, source,
kernel_names) tuples via request_nvrtc_warmup() at module scope.
The singleton NVRTCPrecompiler deduplicates by cache key and submits
new compilations to background threads.

NVRTC compilation releases the GIL via Cython ``with nogil:``, so
threads achieve true CPU parallelism.  14 compilation units on 8+
threads finish in ~200-400ms wall time.

Uses the same VIBESPATIAL_PRECOMPILE toggle as CCCL warmup.
"""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NVRTCWarmupDiagnostic:
    cache_key: str
    elapsed_ms: float
    success: bool
    error: str = ""


class NVRTCPrecompiler:
    """Demand-driven background NVRTC compilation.

    Follows the same singleton + request() + dedup pattern
    as CCCLPrecompiler, but simpler: no type specs, just
    (prefix, source, kernel_names) tuples.
    """

    _instance: NVRTCPrecompiler | None = None

    def __init__(self, max_workers: int | None = None) -> None:
        self._submitted: set[str] = set()
        self._futures: dict[str, Future[Any]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers or min(os.cpu_count() or 4, 16),
            thread_name_prefix="nvrtc-warmup",
        )
        self._start_time: float | None = None
        self._diagnostics: list[NVRTCWarmupDiagnostic] = []

    @classmethod
    def get(cls) -> NVRTCPrecompiler:
        """Lazy singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def _reset(cls) -> None:
        """Reset singleton for testing."""
        if cls._instance is not None:
            cls._instance.shutdown()
            cls._instance = None

    def request(
        self, units: list[tuple[str, str, tuple[str, ...]]],
    ) -> None:
        """Submit (prefix, source, kernel_names) for background compile.

        Idempotent.  Never blocks.
        """
        from vibespatial.cuda_runtime import get_cuda_runtime, make_kernel_cache_key

        runtime = get_cuda_runtime()
        with self._lock:
            if self._start_time is None:
                self._start_time = perf_counter()
            for prefix, source, kernel_names in units:
                cache_key = make_kernel_cache_key(prefix, source)
                if cache_key in self._submitted:
                    continue
                self._submitted.add(cache_key)
                self._futures[cache_key] = self._executor.submit(
                    self._compile_one, runtime, cache_key, source, kernel_names,
                )

    def _compile_one(
        self,
        runtime: Any,
        cache_key: str,
        source: str,
        kernel_names: tuple[str, ...],
    ) -> Any:
        """Compile one NVRTC unit on a background thread."""
        t0 = perf_counter()
        try:
            result = runtime.compile_kernels(
                cache_key=cache_key,
                source=source,
                kernel_names=kernel_names,
            )
            elapsed = (perf_counter() - t0) * 1000.0
            self._diagnostics.append(
                NVRTCWarmupDiagnostic(cache_key, elapsed, True),
            )
            logger.debug("NVRTC warmup: %s compiled in %.1fms", cache_key, elapsed)
            return result
        except Exception as exc:
            elapsed = (perf_counter() - t0) * 1000.0
            self._diagnostics.append(
                NVRTCWarmupDiagnostic(cache_key, elapsed, False, str(exc)),
            )
            logger.debug("NVRTC warmup: %s failed: %s", cache_key, exc)
            return None

    def status(self) -> dict[str, Any]:
        """Diagnostic snapshot."""
        return {
            "submitted": len(self._submitted),
            "compiled": sum(1 for d in self._diagnostics if d.success),
            "pending": sum(1 for f in self._futures.values() if not f.done()),
            "failed": sum(1 for d in self._diagnostics if not d.success),
            "wall_ms": (perf_counter() - self._start_time) * 1000
            if self._start_time
            else 0,
            "per_unit": [
                {"key": d.cache_key, "ms": round(d.elapsed_ms, 1), "ok": d.success}
                for d in self._diagnostics
            ],
        }

    def ensure_warm(self, timeout: float = 30.0) -> list[str]:
        """Block until all submitted NVRTC units are compiled.

        Returns a list of cache keys that timed out or failed.
        Used by Level 3 pipeline-aware warmup (ADR-0034).
        """
        cold: list[str] = []
        for cache_key, future in list(self._futures.items()):
            try:
                future.result(timeout=timeout)
            except Exception:
                cold.append(cache_key)
        return cold

    def shutdown(self) -> None:
        """Shut down the thread pool.  For testing cleanup."""
        self._executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def request_nvrtc_warmup(
    units: list[tuple[str, str, tuple[str, ...]]],
) -> None:
    """Non-blocking request to pre-compile NVRTC kernels.

    Safe to call at module scope.  No-op if GPU is not available
    or if precompilation is disabled via VIBESPATIAL_PRECOMPILE=0.
    """
    from vibespatial.cccl_precompile import precompile_enabled
    from vibespatial.runtime import has_gpu_runtime

    if not precompile_enabled():
        return
    if not has_gpu_runtime():
        return
    NVRTCPrecompiler.get().request(units)
