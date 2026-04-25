from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from .event_log import append_event_record
from .fallbacks import strict_native_mode_enabled


class MaterializationBoundary(StrEnum):
    USER_EXPORT = "user-export"
    INTERNAL_HOST_CONVERSION = "internal-host-conversion"
    DEBUG = "debug"


class StrictNativeMaterializationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MaterializationContext:
    pipeline: str = ""
    dataset: str = ""
    stage: str = ""
    stage_category: str = ""


@dataclass(frozen=True)
class MaterializationEvent:
    surface: str
    boundary: MaterializationBoundary
    reason: str
    operation: str = ""
    detail: str = ""
    pipeline: str = ""
    dataset: str = ""
    stage: str = ""
    stage_category: str = ""
    d2h_transfer: bool = False
    strict_disallowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["boundary"] = self.boundary.value
        return payload


_MATERIALIZATION_EVENTS: deque[MaterializationEvent] = deque(maxlen=512)
_MATERIALIZATION_CONTEXT: ContextVar[MaterializationContext | None] = ContextVar(
    "vibespatial_materialization_context",
    default=None,
)


def current_materialization_context() -> MaterializationContext:
    context = _MATERIALIZATION_CONTEXT.get()
    if context is None:
        return MaterializationContext()
    return context


@contextmanager
def materialization_context(
    *,
    pipeline: str = "",
    dataset: str = "",
    stage: str = "",
    stage_category: str = "",
) -> Iterator[None]:
    parent = current_materialization_context()
    token = _MATERIALIZATION_CONTEXT.set(
        MaterializationContext(
            pipeline=pipeline or parent.pipeline,
            dataset=dataset or parent.dataset,
            stage=stage or parent.stage,
            stage_category=stage_category or parent.stage_category,
        )
    )
    try:
        yield
    finally:
        _MATERIALIZATION_CONTEXT.reset(token)


def record_materialization_event(
    *,
    surface: str,
    boundary: MaterializationBoundary | str,
    reason: str,
    operation: str = "",
    detail: str = "",
    pipeline: str = "",
    dataset: str = "",
    stage: str = "",
    stage_category: str = "",
    d2h_transfer: bool = False,
    strict_disallowed: bool = False,
) -> MaterializationEvent:
    context = current_materialization_context()
    event = MaterializationEvent(
        surface=surface,
        boundary=boundary
        if isinstance(boundary, MaterializationBoundary)
        else MaterializationBoundary(boundary),
        reason=reason,
        operation=operation,
        detail=detail,
        pipeline=pipeline or context.pipeline,
        dataset=dataset or context.dataset,
        stage=stage or context.stage,
        stage_category=stage_category or context.stage_category,
        d2h_transfer=d2h_transfer,
        strict_disallowed=strict_disallowed,
    )
    _MATERIALIZATION_EVENTS.append(event)
    append_event_record("materialization", event.to_dict())
    if strict_disallowed and strict_native_mode_enabled():
        raise StrictNativeMaterializationError(
            f"strict native mode disallows materialization: {surface} :: {reason}"
        )
    return event


def get_materialization_events(*, clear: bool = False) -> list[MaterializationEvent]:
    events = list(_MATERIALIZATION_EVENTS)
    if clear:
        _MATERIALIZATION_EVENTS.clear()
    return events


def clear_materialization_events() -> None:
    _MATERIALIZATION_EVENTS.clear()


__all__ = [
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
