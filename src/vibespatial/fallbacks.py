from __future__ import annotations

import os
import warnings
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from vibespatial.event_log import append_event_record
from vibespatial.runtime import ExecutionMode

STRICT_NATIVE_ENV_VAR = "VIBESPATIAL_STRICT_NATIVE"


class StrictNativeFallbackError(RuntimeError):
    pass


@dataclass(frozen=True)
class FallbackEvent:
    surface: str
    requested: ExecutionMode
    selected: ExecutionMode
    reason: str
    detail: str = ""
    pipeline: str = ""
    d2h_transfer: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["requested"] = self.requested.value
        payload["selected"] = self.selected.value
        return payload


_FALLBACK_EVENTS: deque[FallbackEvent] = deque(maxlen=512)


def strict_native_mode_enabled() -> bool:
    value = os.environ.get(STRICT_NATIVE_ENV_VAR, "")
    return value.lower() in {"1", "true", "yes", "on"}


def record_fallback_event(
    *,
    surface: str,
    reason: str,
    detail: str = "",
    requested: ExecutionMode | str = ExecutionMode.AUTO,
    selected: ExecutionMode | str = ExecutionMode.CPU,
    pipeline: str = "",
    d2h_transfer: bool = False,
) -> FallbackEvent:
    event = FallbackEvent(
        surface=surface,
        requested=requested if isinstance(requested, ExecutionMode) else ExecutionMode(requested),
        selected=selected if isinstance(selected, ExecutionMode) else ExecutionMode(selected),
        reason=reason,
        detail=detail,
        pipeline=pipeline,
        d2h_transfer=d2h_transfer,
    )
    _FALLBACK_EVENTS.append(event)
    append_event_record("fallback", event.to_dict())
    from vibespatial.execution_trace import notify_dispatch

    notify_dispatch(
        surface=surface,
        operation="fallback",
        selected=ExecutionMode.CPU,
        implementation="cpu_fallback",
    )
    from vibespatial.execution_trace import VibeTraceWarning, _trace_warnings_enabled

    if _trace_warnings_enabled():
        parts = [f"[vibeSpatial] CPU fallback: {surface} -- {reason}"]
        if pipeline:
            parts.append(f"  pipeline: {pipeline}")
        if d2h_transfer:
            parts.append("  ** device-to-host transfer triggered **")
        if detail:
            parts.append(f"  detail: {detail}")
        warnings.warn("\n".join(parts), VibeTraceWarning, stacklevel=2)
    if strict_native_mode_enabled():
        raise StrictNativeFallbackError(
            f"strict native mode disallows geopandas fallback: {surface} :: {reason}"
        )
    return event


def get_fallback_events(*, clear: bool = False) -> list[FallbackEvent]:
    events = list(_FALLBACK_EVENTS)
    if clear:
        _FALLBACK_EVENTS.clear()
    return events


def clear_fallback_events() -> None:
    _FALLBACK_EVENTS.clear()
