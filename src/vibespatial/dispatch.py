from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from vibespatial.event_log import append_event_record
from vibespatial.runtime import ExecutionMode

DispatchCallable = Callable[..., Any]
DISPATCH_REGISTRY: dict[str, list[str]] = defaultdict(list)


@dataclass(frozen=True)
class DispatchEvent:
    surface: str
    operation: str
    requested: ExecutionMode
    selected: ExecutionMode
    implementation: str
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["requested"] = self.requested.value
        payload["selected"] = self.selected.value
        return payload


_DISPATCH_EVENTS: deque[DispatchEvent] = deque(maxlen=512)


def dispatches(target: str) -> Callable[[DispatchCallable], DispatchCallable]:
    """Tag a public GeoPandas-facing method with its dispatch contract."""

    def decorator(func: DispatchCallable) -> DispatchCallable:
        qualified_name = f"{func.__module__}.{func.__qualname__}"
        if qualified_name not in DISPATCH_REGISTRY[target]:
            DISPATCH_REGISTRY[target].append(qualified_name)
        return func

    return decorator


def record_dispatch_event(
    *,
    surface: str,
    operation: str,
    implementation: str,
    reason: str,
    detail: str = "",
    requested: ExecutionMode | str = ExecutionMode.AUTO,
    selected: ExecutionMode | str = ExecutionMode.CPU,
) -> DispatchEvent:
    event = DispatchEvent(
        surface=surface,
        operation=operation,
        requested=requested if isinstance(requested, ExecutionMode) else ExecutionMode(requested),
        selected=selected if isinstance(selected, ExecutionMode) else ExecutionMode(selected),
        implementation=implementation,
        reason=reason,
        detail=detail,
    )
    _DISPATCH_EVENTS.append(event)
    append_event_record("dispatch", event.to_dict())
    from vibespatial.execution_trace import notify_dispatch

    notify_dispatch(
        surface=event.surface,
        operation=event.operation,
        selected=event.selected,
        implementation=event.implementation,
    )
    return event


def get_dispatch_events(*, clear: bool = False) -> list[DispatchEvent]:
    events = list(_DISPATCH_EVENTS)
    if clear:
        _DISPATCH_EVENTS.clear()
    return events


def clear_dispatch_events() -> None:
    _DISPATCH_EVENTS.clear()
