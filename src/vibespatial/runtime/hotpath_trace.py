from __future__ import annotations

import os
import threading
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import import_module
from time import perf_counter
from typing import Any

_TRACE_ENV_VAR = "VIBESPATIAL_HOTPATH_TRACE"
_NVTX_ENV_VAR = "VIBESPATIAL_HOTPATH_NVTX"
_TRACE_OFF_VALUES = frozenset(("0", "false", "no", "off", ""))
_TRACE_COUNTER_VALUES = frozenset(("counter", "counters", "level0"))
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
    timing_mode: str = "timing"
    metadata: dict[str, Any] = field(default_factory=dict)


_thread_local = threading.local()


def hotpath_trace_mode() -> str:
    """Return ``off``, ``counter``, or ``timing`` for the current process."""
    value = os.environ.get(_TRACE_ENV_VAR, "0").strip().lower()
    if value in _TRACE_OFF_VALUES:
        return "off"
    if value in _TRACE_COUNTER_VALUES:
        return "counter"
    return "timing"


def hotpath_trace_enabled() -> bool:
    return hotpath_trace_mode() != "off"


def hotpath_timing_enabled() -> bool:
    """Whether trace stages may synchronize to attribute GPU wall time."""
    return hotpath_trace_mode() == "timing"


def hotpath_nvtx_enabled() -> bool:
    value = os.environ.get(_NVTX_ENV_VAR, "0")
    return hotpath_timing_enabled() and value.lower() not in _TRACE_OFF_VALUES


def attach_work_amplification(
    metadata: dict[str, Any] | None,
    *,
    operation: str,
    metric_family: str,
    sums: dict[str, int | float],
    maxima: dict[str, int | float],
    unavailable: tuple[str, ...] = (),
    physical_shape: str | None = None,
    consumer_kind: str | None = None,
    semantic_contract: dict[str, Any] | None = None,
) -> None:
    """Attach one bounded host-known packet to an enabled hotpath stage."""
    if metadata is None:
        return
    packet: dict[str, Any] = {
        "schema_version": 1,
        "operation": operation,
        "metric_family": metric_family,
        "instrumentation_level": 0,
        "sum": dict(sums),
        "max": dict(maxima),
        "unavailable": list(unavailable),
    }
    if physical_shape is not None:
        packet["physical_shape"] = physical_shape
    if consumer_kind is not None:
        packet["consumer_kind"] = consumer_kind
    if semantic_contract is not None:
        packet["semantic_contract"] = dict(semantic_contract)
    metadata["work_amplification"] = packet


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
    _thread_local.hotpath_counter_summary = {}


def get_hotpath_trace() -> tuple[HotpathStageTrace, ...]:
    return tuple(getattr(_thread_local, "hotpath_trace", ()))


def _merge_work_amplification_packet(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    """Merge the explicit bounded counter packet without guessing semantics."""
    for field_name in ("sum", "max"):
        incoming_metrics = incoming.get(field_name)
        if not isinstance(incoming_metrics, dict):
            continue
        target_metrics = target.setdefault(field_name, {})
        if not isinstance(target_metrics, dict):
            continue
        for name, value in incoming_metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if field_name == "sum":
                target_metrics[name] = target_metrics.get(name, 0) + value
            else:
                target_metrics[name] = max(target_metrics.get(name, value), value)
    unavailable = incoming.get("unavailable")
    if isinstance(unavailable, list):
        target_unavailable = target.setdefault("unavailable", [])
        if isinstance(target_unavailable, list):
            for name in unavailable:
                if isinstance(name, str) and name not in target_unavailable:
                    target_unavailable.append(name)
    for field_name in (
        "schema_version",
        "operation",
        "metric_family",
        "physical_shape",
        "consumer_kind",
        "device",
        "measurement_boundary",
        "boundary_role",
        "instrumentation_level",
        "source_lineage",
        "semantic_contract",
    ):
        if field_name not in incoming:
            continue
        if field_name not in target:
            target[field_name] = deepcopy(incoming[field_name])
        elif target[field_name] != incoming[field_name]:
            target["mixed_context"] = True


def _merge_stage_metadata(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    packet = incoming.get("work_amplification")
    if isinstance(packet, dict):
        target_packet = target.setdefault("work_amplification", {})
        if isinstance(target_packet, dict):
            _merge_work_amplification_packet(target_packet, packet)


def aggregate_hotpath_summaries(
    summaries: tuple[list[dict[str, Any]], ...] | list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Combine bounded per-statement summaries without restoring raw calls."""
    totals: dict[tuple[str, str], dict[str, Any]] = {}
    for summary in summaries:
        for stage in summary:
            key = (str(stage["name"]), str(stage["category"]))
            entry = totals.setdefault(
                key,
                {
                    "name": key[0],
                    "category": key[1],
                    "calls": 0,
                    "elapsed_seconds": 0.0,
                    "timing_mode": stage.get("timing_mode", "timing"),
                    "metadata": {},
                },
            )
            entry["calls"] += int(stage.get("calls", 0))
            entry["elapsed_seconds"] += float(stage.get("elapsed_seconds", 0.0))
            metadata = stage.get("metadata")
            if isinstance(metadata, dict):
                _merge_stage_metadata(entry["metadata"], metadata)
    return sorted(
        totals.values(),
        key=lambda item: (-float(item["elapsed_seconds"]), item["name"]),
    )


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
                "timing_mode": stage.timing_mode,
            },
        )
        entry["calls"] += 1
        entry["elapsed_seconds"] += stage.elapsed_seconds
        _merge_stage_metadata(entry.setdefault("metadata", {}), stage.metadata)
    for entry in getattr(_thread_local, "hotpath_counter_summary", {}).values():
        totals[(entry["name"], entry["category"])] = deepcopy(entry)
    return sorted(
        totals.values(),
        key=lambda item: (-float(item["elapsed_seconds"]), item["name"]),
    )


@contextmanager
def hotpath_stage(name: str, *, category: str = "other", metadata: dict[str, Any] | None = None):
    mode = hotpath_trace_mode()
    if mode == "off":
        yield None
        return

    trace = getattr(_thread_local, "hotpath_trace", None)
    if trace is None:
        trace = []
        _thread_local.hotpath_trace = trace

    stage_metadata = dict(metadata or {})
    if mode == "counter":
        try:
            yield stage_metadata
        finally:
            summary = getattr(_thread_local, "hotpath_counter_summary", None)
            if summary is None:
                summary = {}
                _thread_local.hotpath_counter_summary = summary
            key = (name, category)
            entry = summary.setdefault(
                key,
                {
                    "name": name,
                    "category": category,
                    "calls": 0,
                    "elapsed_seconds": 0.0,
                    "timing_mode": "counter",
                    "metadata": {},
                },
            )
            entry["calls"] += 1
            _merge_stage_metadata(entry["metadata"], stage_metadata)
        return

    started = perf_counter()
    with _maybe_nvtx_context(name, category):
        try:
            yield stage_metadata
        finally:
            trace.append(
                HotpathStageTrace(
                    name=name,
                    category=category,
                    elapsed_seconds=(
                        perf_counter() - started
                    ),
                    timing_mode=mode,
                    metadata=stage_metadata,
                )
            )
