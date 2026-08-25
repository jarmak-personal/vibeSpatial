from __future__ import annotations

from vibespatial.runtime.hotpath_trace import (
    aggregate_hotpath_summaries,
    attach_work_amplification,
    get_hotpath_trace,
    hotpath_stage,
    hotpath_timing_enabled,
    hotpath_trace_mode,
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


def test_counter_mode_records_without_timing_or_nvtx(monkeypatch) -> None:
    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "counter")
    monkeypatch.setenv("VIBESPATIAL_HOTPATH_NVTX", "1")
    reset_hotpath_trace()

    with hotpath_stage("counter.stage", metadata={"candidate_pairs": 12}) as metadata:
        assert metadata == {"candidate_pairs": 12}

    assert hotpath_trace_mode() == "counter"
    assert hotpath_timing_enabled() is False
    assert get_hotpath_trace() == ()
    summary = summarize_hotpath_trace()
    assert summary[0]["timing_mode"] == "counter"
    assert summary[0]["calls"] == 1
    assert summary[0]["elapsed_seconds"] == 0.0


def test_counter_mode_aggregates_only_explicit_work_amplification_reducers(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "counter")
    reset_hotpath_trace()

    for candidate_pairs in (3, 5):
        with hotpath_stage("relation.reduce", category="filter") as metadata:
            assert metadata is not None
            metadata["work_amplification"] = {
                "schema_version": 1,
                "operation": "query_aggregate",
                "metric_family": "relation",
                "sum": {"candidate_pairs": candidate_pairs},
                "max": {"live_pair_capacity": candidate_pairs},
                "unavailable": ["survivors"],
            }

    summary = summarize_hotpath_trace()[0]
    packet = summary["metadata"]["work_amplification"]
    assert summary["calls"] == 2
    assert packet["sum"] == {"candidate_pairs": 8}
    assert packet["max"] == {"live_pair_capacity": 5}
    assert packet["unavailable"] == ["survivors"]
    assert "mixed_context" not in packet


def test_attach_work_amplification_is_bounded_and_disabled_safe() -> None:
    metadata: dict[str, object] = {}
    attach_work_amplification(
        metadata,
        operation="query_aggregate",
        metric_family="relation",
        sums={"candidate_pairs": 10},
        maxima={"live_pair_capacity": 4},
        unavailable=("survivors",),
        physical_shape="candidate_pairs",
        consumer_kind="count",
        semantic_contract={"predicate": "within"},
    )
    attach_work_amplification(
        None,
        operation="disabled",
        metric_family="relation",
        sums={},
        maxima={},
    )

    assert metadata["work_amplification"] == {
        "schema_version": 1,
        "operation": "query_aggregate",
        "metric_family": "relation",
        "instrumentation_level": 0,
        "sum": {"candidate_pairs": 10},
        "max": {"live_pair_capacity": 4},
        "unavailable": ["survivors"],
        "physical_shape": "candidate_pairs",
        "consumer_kind": "count",
        "semantic_contract": {"predicate": "within"},
    }


def test_per_statement_hotpath_summaries_aggregate_without_raw_calls() -> None:
    summaries = (
        [
            {
                "name": "relation.reduce",
                "category": "filter",
                "calls": 2,
                "elapsed_seconds": 0.0,
                "timing_mode": "counter",
                "metadata": {
                    "work_amplification": {
                        "schema_version": 1,
                        "operation": "query_aggregate",
                        "metric_family": "relation",
                        "sum": {"candidate_pairs": 4},
                        "max": {"live_pair_capacity": 3},
                    }
                },
            }
        ],
        [
            {
                "name": "relation.reduce",
                "category": "filter",
                "calls": 1,
                "elapsed_seconds": 0.0,
                "timing_mode": "counter",
                "metadata": {
                    "work_amplification": {
                        "schema_version": 1,
                        "operation": "query_aggregate",
                        "metric_family": "relation",
                        "sum": {"candidate_pairs": 5},
                        "max": {"live_pair_capacity": 5},
                    }
                },
            }
        ],
    )

    combined = aggregate_hotpath_summaries(summaries)[0]
    packet = combined["metadata"]["work_amplification"]
    assert combined["calls"] == 3
    assert packet["sum"]["candidate_pairs"] == 9
    assert packet["max"]["live_pair_capacity"] == 5
