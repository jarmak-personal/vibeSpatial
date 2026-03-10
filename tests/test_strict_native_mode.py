from __future__ import annotations

import pytest

from vibespatial.fallbacks import (
    StrictNativeFallbackError,
    record_fallback_event,
    strict_native_mode_enabled,
)


def test_strict_native_mode_disabled_by_default() -> None:
    assert strict_native_mode_enabled() is False


def test_record_fallback_event_raises_in_strict_native_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIBESPATIAL_STRICT_NATIVE", "1")

    with pytest.raises(StrictNativeFallbackError):
        record_fallback_event(surface="geopandas.array.contains", reason="explicit CPU fallback")
