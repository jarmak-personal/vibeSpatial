# Re-export the public API that was previously in runtime.py
from vibespatial.runtime._runtime import *  # noqa: F403
from vibespatial.runtime.workload import WorkloadShape, detect_workload_shape

__all__ = [  # noqa: F405
    # from _runtime (star-import)
    "ExecutionMode",
    "RuntimeSelection",
    "has_gpu_runtime",
    "select_runtime",
    "set_execution_mode",
    "get_requested_mode",
    # from workload
    "WorkloadShape",
    "detect_workload_shape",
]
