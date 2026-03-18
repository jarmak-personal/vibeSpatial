from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np


class GeometryPresence(StrEnum):
    NULL = "null"
    EMPTY = "empty"
    VALUE = "value"


class UnaryNullPolicy(StrEnum):
    PROPAGATE = "propagate"


class EmptyMeasurementPolicy(StrEnum):
    ZERO = "zero"
    NAN_BOUNDS = "nan-bounds"


class PredicateNullPolicy(StrEnum):
    PROPAGATE = "propagate"


@dataclass(frozen=True)
class GeometrySemantics:
    presence: GeometryPresence
    geom_type: str | None = None


NULL_BOUNDS = (math.nan, math.nan, math.nan, math.nan)


def is_null_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, np.generic) and np.isnan(value):
        return True
    return False


def classify_geometry(value: Any) -> GeometrySemantics:
    if is_null_like(value):
        return GeometrySemantics(presence=GeometryPresence.NULL, geom_type=None)
    if hasattr(value, "is_empty") and bool(value.is_empty):
        geom_type = getattr(value, "geom_type", None)
        return GeometrySemantics(presence=GeometryPresence.EMPTY, geom_type=geom_type)
    geom_type = getattr(value, "geom_type", None)
    return GeometrySemantics(presence=GeometryPresence.VALUE, geom_type=geom_type)


def unary_result_for_missing_input(value: Any) -> None:
    semantics = classify_geometry(value)
    if semantics.presence is GeometryPresence.NULL:
        return None
    raise ValueError("unary_result_for_missing_input only applies to null inputs")


def measurement_result_for_geometry(value: Any, *, kind: str) -> float | tuple[float, float, float, float]:
    semantics = classify_geometry(value)
    if semantics.presence is GeometryPresence.NULL:
        return None  # type: ignore[return-value]
    if semantics.presence is GeometryPresence.EMPTY:
        if kind == "bounds":
            return NULL_BOUNDS
        if kind in {"area", "length"}:
            return 0.0
        raise ValueError(f"unsupported measurement kind: {kind}")
    raise ValueError("measurement_result_for_geometry only applies to null or empty geometry inputs")


def predicate_result_for_pair(left: Any, right: Any) -> bool | None:
    left_state = classify_geometry(left)
    right_state = classify_geometry(right)
    if GeometryPresence.NULL in {left_state.presence, right_state.presence}:
        return None
    if GeometryPresence.EMPTY in {left_state.presence, right_state.presence}:
        return False
    raise ValueError("predicate_result_for_pair only applies when at least one input is null or empty")
