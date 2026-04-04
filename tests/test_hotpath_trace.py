from __future__ import annotations

from vibespatial.runtime.hotpath_trace import (
    get_hotpath_trace,
    hotpath_stage,
    reset_hotpath_trace,
    summarize_hotpath_trace,
)


def test_hotpath_trace_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("VIBESPATIAL_HOTPATH_TRACE", raising=False)
    reset_hotpath_trace()
    with hotpath_stage("disabled", category="setup"):
        pass
    assert get_hotpath_trace() == ()


def test_hotpath_trace_collects_and_summarizes(monkeypatch) -> None:
    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()

    with hotpath_stage("segment.classify.launch_kernel", category="refine"):
        pass
    with hotpath_stage("segment.classify.launch_kernel", category="refine"):
        pass
    with hotpath_stage("overlay.split.scatter_pair_events", category="emit"):
        pass

    trace = get_hotpath_trace()
    assert len(trace) == 3
    summary = summarize_hotpath_trace()
    assert summary[0]["name"] == "segment.classify.launch_kernel"
    assert summary[0]["calls"] == 2
    assert summary[1]["name"] == "overlay.split.scatter_pair_events"
    assert summary[1]["calls"] == 1
