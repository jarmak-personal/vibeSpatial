# Re-export the public API that was previously in runtime.py
from vibespatial.runtime._runtime import *  # noqa: F403
from vibespatial.runtime.residency import combined_residency

__all__ = [  # noqa: F405
    # from _runtime (star-import)
    "ExecutionMode",
    "RuntimeSelection",
    "has_gpu_runtime",
    "select_runtime",
    "set_execution_mode",
    "set_requested_mode",
    "get_requested_mode",
    "combined_residency",
]
