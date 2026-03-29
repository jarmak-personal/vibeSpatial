"""Runtime device-residency enforcement for tests.

Provides a context manager and pytest fixture that monkey-patch cupy and
numpy at runtime to raise ``DeviceResidencyViolation`` when GPU pipeline
code transfers data to the host.

This catches shortcuts that static analysis misses — the code actually
runs and the transfer is intercepted at the moment it happens.

Guarded transfer points:
  - ``cupy.ndarray.get()``
  - ``cupy.asnumpy()``
  - ``numpy.asarray()`` when given a cupy array (implicit D2H)

Usage in tests::

    def test_overlay_stays_on_device(strict_device_guard):
        result = overlay_intersection_gpu(polys_a, polys_b)
        # If overlay_intersection_gpu calls .get() anywhere, this test
        # raises DeviceResidencyViolation immediately.

Or as a context manager::

    with device_residency_guard("overlay_intersection"):
        result = overlay_intersection_gpu(polys_a, polys_b)
"""
from __future__ import annotations

import functools
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


class DeviceResidencyViolation(RuntimeError):
    """Raised when a D2H transfer occurs inside a guarded GPU pipeline."""
    pass


# ---------------------------------------------------------------------------
# Internal state: track whether we're inside a guarded scope
# ---------------------------------------------------------------------------
_guard_stack: list[str] = []


def _in_guarded_scope() -> bool:
    return len(_guard_stack) > 0


def _current_scope() -> str:
    return _guard_stack[-1] if _guard_stack else ""


# ---------------------------------------------------------------------------
# Allowlist: callers that are permitted to transfer even inside a guard.
# These are materialization boundaries (to_pandas, __repr__, etc.) and
# test oracle code that necessarily compares on host.
# ---------------------------------------------------------------------------
_ALLOWED_CALLERS = {
    # User-visible materialization (expected D2H)
    "to_pandas", "to_numpy", "to_host", "__repr__", "__str__",
    "_to_host_array", "_ensure_host_state", "to_wkt", "to_wkb",
    "to_pylist", "tolist",
    # Test oracle: needs host data to compare against shapely
    "assert_matches_shapely", "compare_with_shapely",
    "_compare_single", "_compare_results",
    # Diagnostic / profiling (not pipeline code)
    "record_dispatch_event", "record_fallback_event",
    "_record", "_log_diagnostic",
}


def _caller_is_allowed() -> bool:
    """Walk the call stack to check if the transfer is from an allowed caller."""
    for frame_info in traceback.extract_stack():
        if frame_info.name in _ALLOWED_CALLERS:
            return True
    return False


def _format_violation(method_name: str, extra: str = "") -> str:
    """Format a clear violation message with stack trace context."""
    scope = _current_scope()
    # Find the most relevant vibespatial frame
    relevant_frame = ""
    for frame_info in traceback.extract_stack():
        if "vibespatial" in frame_info.filename and "testing" not in frame_info.filename:
            relevant_frame = f"  at {frame_info.filename}:{frame_info.lineno} in {frame_info.name}()"

    msg = (
        f"DeviceResidencyViolation: {method_name} called inside guarded "
        f"GPU scope '{scope}'.\n"
        f"{relevant_frame}\n"
        f"{extra}"
        f"Data must stay on device. Use cupy operations instead of numpy."
    )
    return msg


# ---------------------------------------------------------------------------
# The guard: monkey-patches cupy and numpy transfer points
# ---------------------------------------------------------------------------
@contextmanager
def device_residency_guard(scope_name: str = "gpu_pipeline") -> Generator[None]:
    """Context manager that raises on any D2H transfer.

    Parameters
    ----------
    scope_name : str
        Label for the guarded scope (appears in error messages).
    """
    _guard_stack.append(scope_name)
    patches = _install_patches()
    try:
        yield
    finally:
        _guard_stack.pop()
        _uninstall_patches(patches)


def _install_patches() -> dict[str, Any]:
    """Install monkey-patches on cupy and numpy transfer points.

    Returns a dict of originals for restoration.
    """
    originals: dict[str, Any] = {}

    # --- cupy.ndarray.get ---
    try:
        import cupy as cp

        if not hasattr(cp.ndarray.get, "_device_guard_wrapped"):
            originals["cp_ndarray_get"] = cp.ndarray.get

            @functools.wraps(cp.ndarray.get)
            def _guarded_get(self: Any, *args: Any, **kwargs: Any) -> Any:
                if _in_guarded_scope() and not _caller_is_allowed():
                    raise DeviceResidencyViolation(
                        _format_violation(
                            ".get()",
                            f"  Array shape={self.shape}, dtype={self.dtype}\n",
                        )
                    )
                return originals["cp_ndarray_get"](self, *args, **kwargs)

            _guarded_get._device_guard_wrapped = True  # type: ignore[attr-defined]
            cp.ndarray.get = _guarded_get  # type: ignore[method-assign]

        # --- cupy.asnumpy ---
        if not hasattr(cp.asnumpy, "_device_guard_wrapped"):
            originals["cp_asnumpy"] = cp.asnumpy

            @functools.wraps(cp.asnumpy)
            def _guarded_asnumpy(a: Any, *args: Any, **kwargs: Any) -> Any:
                if _in_guarded_scope() and not _caller_is_allowed():
                    raise DeviceResidencyViolation(
                        _format_violation("cupy.asnumpy()")
                    )
                return originals["cp_asnumpy"](a, *args, **kwargs)

            _guarded_asnumpy._device_guard_wrapped = True  # type: ignore[attr-defined]
            cp.asnumpy = _guarded_asnumpy  # type: ignore[attr-defined]

    except ImportError:
        pass  # cupy not available — nothing to guard

    # --- numpy.asarray when given a cupy array (implicit D2H) ---
    try:
        import numpy as np

        if not hasattr(np.asarray, "_device_guard_wrapped"):
            originals["np_asarray"] = np.asarray

            @functools.wraps(np.asarray)
            def _guarded_np_asarray(a: Any, *args: Any, **kwargs: Any) -> Any:
                if (
                    _in_guarded_scope()
                    and hasattr(a, "__cuda_array_interface__")
                    and not _caller_is_allowed()
                ):
                    raise DeviceResidencyViolation(
                        _format_violation(
                            "numpy.asarray() on device-resident data",
                            "  This triggers an implicit D2H transfer.\n",
                        )
                    )
                return originals["np_asarray"](a, *args, **kwargs)

            _guarded_np_asarray._device_guard_wrapped = True  # type: ignore[attr-defined]
            np.asarray = _guarded_np_asarray  # type: ignore[attr-defined]

    except ImportError:
        pass

    return originals


def _uninstall_patches(originals: dict[str, Any]) -> None:
    """Restore original functions if we're exiting the last guard scope."""
    if _in_guarded_scope():
        return  # nested guard — keep patches active

    try:
        import cupy as cp
        if "cp_ndarray_get" in originals:
            cp.ndarray.get = originals["cp_ndarray_get"]  # type: ignore[method-assign]
        if "cp_asnumpy" in originals:
            cp.asnumpy = originals["cp_asnumpy"]  # type: ignore[attr-defined]
    except ImportError:
        pass

    try:
        import numpy as np
        if "np_asarray" in originals:
            np.asarray = originals["np_asarray"]  # type: ignore[attr-defined]
    except ImportError:
        pass
