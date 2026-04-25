# Re-export the public API that was previously in runtime.py
from vibespatial.runtime._runtime import *  # noqa: F403
from vibespatial.runtime.materialization import (
    MaterializationBoundary,
    MaterializationContext,
    MaterializationEvent,
    StrictNativeMaterializationError,
    clear_materialization_events,
    current_materialization_context,
    get_materialization_events,
    materialization_context,
    record_materialization_event,
)
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
    "MaterializationBoundary",
    "MaterializationContext",
    "MaterializationEvent",
    "StrictNativeMaterializationError",
    "clear_materialization_events",
    "current_materialization_context",
    "get_materialization_events",
    "materialization_context",
    "record_materialization_event",
]
