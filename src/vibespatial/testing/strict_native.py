from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

from vibespatial.runtime import EXECUTION_MODE_ENV_VAR, set_execution_mode
from vibespatial.runtime.fallbacks import STRICT_NATIVE_ENV_VAR, StrictNativeFallbackError


@dataclass(frozen=True)
class StrictApiCallResult:
    surface: str
    ok: bool
    result_type: str | None = None
    result_len: int | None = None
    error_type: str | None = None
    error: str | None = None
    strict_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrictApiMatrixReport:
    fixture: str
    geometry_types: tuple[str, ...]
    calls: tuple[StrictApiCallResult, ...]

    def by_surface(self) -> dict[str, StrictApiCallResult]:
        return {result.surface: result for result in self.calls}

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture": self.fixture,
            "geometry_types": list(self.geometry_types),
            "calls": {
                result.surface: result.to_dict()
                for result in self.calls
            },
        }


def capture_strict_api_call(
    surface: str,
    invoke: Callable[[], object],
) -> StrictApiCallResult:
    try:
        value = invoke()
    except Exception as exc:
        return StrictApiCallResult(
            surface=surface,
            ok=False,
            error_type=type(exc).__name__,
            error=str(exc).splitlines()[0],
            strict_fallback=isinstance(exc, StrictNativeFallbackError),
        )

    result_len = len(value) if hasattr(value, "__len__") else None
    return StrictApiCallResult(
        surface=surface,
        ok=True,
        result_type=type(value).__name__,
        result_len=result_len,
    )


def run_strict_api_matrix(
    fixture: str,
    target: object,
    calls: Mapping[str, Callable[[object], object]],
    *,
    geometry_types: tuple[str, ...] = (),
) -> StrictApiMatrixReport:
    results = tuple(
        capture_strict_api_call(surface, lambda fn=fn: fn(target))
        for surface, fn in calls.items()
    )
    return StrictApiMatrixReport(
        fixture=fixture,
        geometry_types=geometry_types,
        calls=results,
    )


@contextmanager
def strict_native_environment(
    *,
    execution_mode: str = "gpu",
):
    previous_strict = os.environ.get(STRICT_NATIVE_ENV_VAR)
    previous_mode = os.environ.get(EXECUTION_MODE_ENV_VAR)
    set_execution_mode(None)
    os.environ[STRICT_NATIVE_ENV_VAR] = "1"
    os.environ[EXECUTION_MODE_ENV_VAR] = execution_mode
    try:
        yield
    finally:
        set_execution_mode(None)
        if previous_strict is None:
            os.environ.pop(STRICT_NATIVE_ENV_VAR, None)
        else:
            os.environ[STRICT_NATIVE_ENV_VAR] = previous_strict
        if previous_mode is None:
            os.environ.pop(EXECUTION_MODE_ENV_VAR, None)
        else:
            os.environ[EXECUTION_MODE_ENV_VAR] = previous_mode


__all__ = [
    "StrictApiCallResult",
    "StrictApiMatrixReport",
    "capture_strict_api_call",
    "run_strict_api_matrix",
    "strict_native_environment",
]
