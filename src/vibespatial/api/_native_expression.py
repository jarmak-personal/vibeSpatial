from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NativeExpression:
    """Private expression vector consumed only by admitted native operations."""

    operation: str
    values: Any
    source_token: str | None = None
    dtype: str | None = None
    precision: str | None = None


__all__ = ["NativeExpression"]
