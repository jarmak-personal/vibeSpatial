from __future__ import annotations

import os
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from importlib import import_module
from time import perf_counter
from typing import Any

_TRACE_ENV_VAR = "VIBESPATIAL_HOTPATH_TRACE"
_NVTX_ENV_VAR = "VIBESPATIAL_HOTPATH_NVTX"
_NVTX_COLORS = {
    "setup": "blue",
    "sort": "purple",
    "filter": "green",
    "refine": "red",
    "emit": "orange",
    "other": "blue",
}


@dataclass(frozen=True)
class HotpathStageTrace:
    name: str
    category: str
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


_thread_local = threading.local()


def hotpath_trace_enabled() -> bool:
    value = os.environ.get(_TRACE_ENV_VAR, "0")
    return value.lower() not in {"0", "false", "no", "off", ""}


def hotpath_nvtx_enabled() -> bool:
    value = os.environ.get(_NVTX_ENV_VAR, "0")
    return value.lower() not in {"0", "false", "no", "off", ""}


def _maybe_nvtx_context(label: str, category: str):
    if not hotpath_nvtx_enabled():
        return nullcontext()
    try:
        nvtx = import_module("nvtx")
    except ImportError:
        return nullcontext()
    return nvtx.annotate(label, color=_NVTX_COLORS.get(category, "blue"))


def reset_hotpath_trace() -> None:
    _thread_local.hotpath_trace = []


def get_hotpath_trace() -> tuple[HotpathStageTrace, ...]:
    return tuple(getattr(_thread_local, "hotpath_trace", ()))


def summarize_hotpath_trace() -> list[dict[str, Any]]:
    totals: dict[tuple[str, str], dict[str, Any]] = {}
    for stage in get_hotpath_trace():
        key = (stage.name, stage.category)
        entry = totals.setdefault(
            key,
            {
                "name": stage.name,
                "category": stage.category,
                "calls": 0,
                "elapsed_seconds": 0.0,
            },
        )
        entry["calls"] += 1
        entry["elapsed_seconds"] += stage.elapsed_seconds
    return sorted(
        totals.values(),
        key=lambda item: (-float(item["elapsed_seconds"]), item["name"]),
    )


@contextmanager
def hotpath_stage(name: str, *, category: str = "other", metadata: dict[str, Any] | None = None):
    if not hotpath_trace_enabled():
        yield
        return

    trace = getattr(_thread_local, "hotpath_trace", None)
    if trace is None:
        trace = []
        _thread_local.hotpath_trace = trace

    started = perf_counter()
    with _maybe_nvtx_context(name, category):
        try:
            yield
        finally:
            trace.append(
                HotpathStageTrace(
                    name=name,
                    category=category,
                    elapsed_seconds=perf_counter() - started,
                    metadata=dict(metadata or {}),
                )
            )
