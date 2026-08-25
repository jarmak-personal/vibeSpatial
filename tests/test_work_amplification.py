from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest

from vibespatial.bench.amplification import (
    SCHEMA_VERSION,
    WORK_AMPLIFICATION_FIELD,
    Lineage,
    LineageAliaser,
    Metric,
    MetricStatus,
    Ratio,
    Record,
    make_ratio,
    ratio_from_metrics,
)


def _record() -> Record:
    metrics = (
        Metric(
            "coarse_candidates",
            12,
            MetricStatus.EXACT,
            source="spatial_index.query_count",
            unit="pairs",
        ),
        Metric(
            "exact_pairs",
            3,
            MetricStatus.EXACT,
            source="predicate.survivor_count",
            unit="pairs",
        ),
        Metric(
            "terminal_bytes",
            None,
            MetricStatus.UNAVAILABLE,
            unit="bytes",
            reason="allocator telemetry disabled",
        ),
    )
    return Record(
        operation="spatial-query",
        stage="exact-refinement",
        metric_family="relation",
        physical_shape="candidate_pairs",
        consumer_kind="pair_preserving",
        device="gpu",
        measurement_boundary="profile_stage",
        boundary_role="compute",
        instrumentation_level=0,
        source_lineage=(Lineage("source_0", role="left"), Lineage("source_1", role="right")),
        metrics=metrics,
        ratios=(
            ratio_from_metrics(
                "coarse_to_exact",
                numerator=metrics[0],
                denominator=metrics[1],
            ),
        ),
        semantic_contract={
            "predicate": "intersects",
            "preserves": ["multiplicity", "left_index", "right_index"],
            "nullable": True,
            "dimension": None,
            "nested": {"enabled": True},
        },
    )


@pytest.mark.parametrize(
    ("status", "value", "reason"),
    [
        (MetricStatus.EXACT, 1, None),
        (MetricStatus.SAMPLED, 1.5, None),
        (MetricStatus.DERIVED, 0, None),
        (MetricStatus.UNAVAILABLE, None, "not instrumented"),
        (MetricStatus.INVALID, None, "counter overflow"),
    ],
)
def test_metric_statuses_round_trip(
    status: MetricStatus,
    value: int | float | None,
    reason: str | None,
) -> None:
    source = "test.counter" if status in {
        MetricStatus.EXACT,
        MetricStatus.SAMPLED,
        MetricStatus.DERIVED,
    } else None
    metadata = {"sampling": {"method": "every_n", "stride": 8}} if status is MetricStatus.SAMPLED else {}
    metric = Metric(
        "physical_work",
        value,
        status,
        source=source,
        unit="items",
        reason=reason,
        metadata=metadata,
    )

    payload = metric.to_dict()

    assert payload["status"] == status.value
    assert Metric.from_dict(payload) == metric
    json.dumps(payload, allow_nan=False)


def test_metric_rejects_status_value_mismatches_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="value must be present"):
        Metric("candidate_pairs", None, MetricStatus.EXACT, source="test.counter")
    with pytest.raises(ValueError, match="value must be None"):
        Metric("candidate_pairs", 4, MetricStatus.UNAVAILABLE, reason="missing")
    with pytest.raises(ValueError, match="reason is required"):
        Metric("candidate_pairs", None, MetricStatus.INVALID)
    with pytest.raises(ValueError, match="must be finite"):
        Metric("candidate_pairs", float("nan"), MetricStatus.EXACT, source="test.counter")
    with pytest.raises(TypeError, match="must be an int, float, or None"):
        Metric("candidate_pairs", True, MetricStatus.EXACT, source="test.counter")
    with pytest.raises(ValueError, match="status must be one of"):
        Metric("candidate_pairs", 1, "estimated", source="test.counter")


def test_measured_metrics_require_sources_and_sampled_metrics_require_sampling_metadata() -> None:
    with pytest.raises(ValueError, match="source is required"):
        Metric("candidate_pairs", 4, MetricStatus.EXACT)
    with pytest.raises(ValueError, match=r"metadata\.sampling"):
        Metric("candidate_pairs", 4, MetricStatus.SAMPLED, source="sampled.counter")

    derived = Metric(
        "pair_bytes",
        64,
        MetricStatus.DERIVED,
        source="metric_product",
        metadata={"source_metrics": ["exact_pairs", "pair_width_bytes"]},
    )
    assert derived.to_dict()["metadata"]["source_metrics"] == [
        "exact_pairs",
        "pair_width_bytes",
    ]


def test_make_ratio_retains_raw_names_and_values() -> None:
    ratio = make_ratio(
        "coarse_to_exact",
        numerator_metric="coarse_candidates",
        numerator_value=12,
        denominator_metric="exact_pairs",
        denominator_value=3,
    )

    assert ratio.status is MetricStatus.DERIVED
    assert ratio.value == 4.0
    assert ratio.numerator_name == "coarse_candidates"
    assert ratio.numerator_value == 12
    assert ratio.denominator_name == "exact_pairs"
    assert ratio.denominator_value == 3
    assert Ratio.from_dict(ratio.to_dict()) == ratio


def test_make_ratio_marks_zero_denominator_invalid_without_finite_substitute() -> None:
    ratio = make_ratio(
        "relation_to_terminal",
        numerator_metric="exact_pairs",
        numerator_value=57,
        denominator_metric="terminal_rows",
        denominator_value=0,
    )

    assert ratio.to_dict() == {
        "name": "relation_to_terminal",
        "numerator_metric": "exact_pairs",
        "numerator_value": 57,
        "denominator_metric": "terminal_rows",
        "denominator_value": 0,
        "value": None,
        "status": "invalid",
        "reason": "denominator_zero",
        "denominator_zero": True,
    }


def test_ratio_rejects_fabricated_or_inconsistent_values() -> None:
    with pytest.raises(ValueError, match="zero denominator"):
        Ratio(
            "relation_to_terminal",
            numerator_metric="exact_pairs",
            numerator_value=57,
            denominator_metric="terminal_rows",
            denominator_value=0,
            value=57.0,
        )
    with pytest.raises(ValueError, match="does not match"):
        Ratio(
            "coarse_to_exact",
            numerator_metric="coarse_candidates",
            numerator_value=12,
            denominator_metric="exact_pairs",
            denominator_value=3,
            value=5.0,
        )
    with pytest.raises(TypeError, match="numerator_value must be an int or float"):
        make_ratio(
            "bad_ratio",
            numerator_metric="work",
            numerator_value=None,  # type: ignore[arg-type]
            denominator_metric="output",
            denominator_value=1,
        )


def test_ratio_from_unavailable_metric_is_explicitly_unavailable() -> None:
    numerator = Metric("pair_bytes", None, "unavailable", reason="allocator disabled")
    denominator = Metric("terminal_bytes", 8, "exact", source="allocator.used_bytes")

    ratio = ratio_from_metrics(
        "relation_byte_amplification",
        numerator=numerator,
        denominator=denominator,
    )

    assert ratio.status is MetricStatus.UNAVAILABLE
    assert ratio.value is None
    assert ratio.reason == "input_metric_unavailable"
    assert ratio.numerator_name == "pair_bytes"
    assert ratio.denominator_name == "terminal_bytes"


def test_denominator_zero_flag_is_only_set_for_zero_denominator_invalid_ratio() -> None:
    ratio = Ratio(
        "unavailable_ratio",
        numerator_metric="missing_work",
        numerator_value=None,
        denominator_metric="terminal_rows",
        denominator_value=0,
        value=None,
        status=MetricStatus.UNAVAILABLE,
        reason="input_metric_unavailable",
    )

    assert ratio.denominator_zero is False
    assert ratio.to_dict()["denominator_zero"] is False


def test_lineage_aliaser_is_trace_local_stable_and_never_serializes_raw_tokens() -> None:
    raw_left = UUID("12345678-1234-5678-1234-567812345678")
    raw_right = "c1185671d7214dd3a3caf420f1809b6f"
    aliases = LineageAliaser()

    left = aliases.alias(raw_left, role="left")
    same_left = aliases.alias(raw_left, role="probe")
    right = aliases.alias(raw_right, role="right")
    serialized = json.dumps([left.to_dict(), same_left.to_dict(), right.to_dict()])

    assert left == Lineage("source_0", role="left")
    assert same_left == Lineage("source_0", role="probe")
    assert right == Lineage("source_1", role="right")
    assert str(raw_left) not in serialized
    assert raw_left.hex not in serialized
    assert raw_right not in serialized
    assert raw_right not in repr(aliases)
    assert LineageAliaser().alias(raw_right) == Lineage("source_0")


def test_lineage_rejects_raw_tokens_and_aliaser_rejects_unhashable_tokens() -> None:
    with pytest.raises(ValueError, match=r"source_<number>"):
        Lineage("c1185671d7214dd3a3caf420f1809b6f")
    with pytest.raises(TypeError, match="hashable"):
        LineageAliaser().alias([])


def test_record_serializes_as_one_versioned_json_native_object() -> None:
    record = _record()

    payload = record.to_dict()

    assert payload["schema_version"] == SCHEMA_VERSION == 1
    assert WORK_AMPLIFICATION_FIELD == "work_amplification"
    assert set(payload) == {
        "schema_version",
        "operation",
        "stage",
        "metric_family",
        "physical_shape",
        "consumer_kind",
        "device",
        "measurement_boundary",
        "boundary_role",
        "instrumentation_level",
        "source_lineage",
        "metrics",
        "ratios",
        "semantic_contract",
    }
    assert payload["operation"] == "spatial-query"
    assert payload["source_lineage"] == [
        {"alias": "source_0", "role": "left"},
        {"alias": "source_1", "role": "right"},
    ]
    assert payload["semantic_contract"]["preserves"] == [
        "multiplicity",
        "left_index",
        "right_index",
    ]
    json.dumps({WORK_AMPLIFICATION_FIELD: [payload]}, allow_nan=False)
    assert Record.from_dict(payload) == record


def test_record_is_deeply_immutable_and_to_dict_is_detached() -> None:
    record = _record()

    with pytest.raises(FrozenInstanceError):
        record.stage = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        record.semantic_contract["extra"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        record.semantic_contract["nested"]["enabled"] = False  # type: ignore[index]

    detached = record.to_dict()
    detached["semantic_contract"]["preserves"].append("geometry")
    assert tuple(record.semantic_contract["preserves"]) == (
        "multiplicity",
        "left_index",
        "right_index",
    )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "error_type", "match"),
    [
        ("operation", "", ValueError, "operation must not be empty"),
        ("instrumentation_level", True, TypeError, "must be an integer"),
        ("instrumentation_level", 3, ValueError, "must be 0, 1, or 2"),
        ("source_lineage", (), ValueError, "source_lineage must not be empty"),
        ("metrics", (), ValueError, "metrics must not be empty"),
        ("ratios", {}, TypeError, "ratios must be a list or tuple"),
    ],
)
def test_record_rejects_invalid_fields(
    field_name: str,
    bad_value: object,
    error_type: type[Exception],
    match: str,
) -> None:
    kwargs = {
        name: getattr(_record(), name)
        for name in (
            "operation",
            "stage",
            "metric_family",
            "physical_shape",
            "consumer_kind",
            "device",
            "measurement_boundary",
            "boundary_role",
            "instrumentation_level",
            "source_lineage",
            "metrics",
            "ratios",
            "semantic_contract",
        )
    }
    kwargs["semantic_contract"] = _record().to_dict()["semantic_contract"]
    kwargs[field_name] = bad_value

    with pytest.raises(error_type, match=match):
        Record(**kwargs)


def test_record_rejects_non_json_semantic_contract_values() -> None:
    kwargs = _record().to_dict()
    kwargs.pop("schema_version")
    kwargs["source_lineage"] = tuple(Lineage.from_dict(item) for item in kwargs["source_lineage"])
    kwargs["metrics"] = tuple(Metric.from_dict(item) for item in kwargs["metrics"])
    kwargs["ratios"] = tuple(Ratio.from_dict(item) for item in kwargs["ratios"])
    kwargs["semantic_contract"] = {"bad": ("tuple",)}

    with pytest.raises(TypeError, match="JSON-native"):
        Record(**kwargs)


def test_record_rejects_duplicate_or_missing_metric_references() -> None:
    record = _record()
    kwargs = {
        field_name: getattr(record, field_name)
        for field_name in (
            "operation",
            "stage",
            "metric_family",
            "physical_shape",
            "consumer_kind",
            "device",
            "measurement_boundary",
            "boundary_role",
            "instrumentation_level",
            "source_lineage",
            "metrics",
            "ratios",
        )
    }
    kwargs["semantic_contract"] = record.to_dict()["semantic_contract"]

    with pytest.raises(ValueError, match="duplicate names"):
        Record(**(kwargs | {"metrics": (record.metrics[0], record.metrics[0])}))
    with pytest.raises(ValueError, match="missing metric"):
        Record(**(kwargs | {"metrics": record.metrics[1:]}))


def test_from_dict_rejects_unknown_version_fields_and_non_json_arrays() -> None:
    payload = _record().to_dict()
    with pytest.raises(ValueError, match="unsupported work-amplification schema_version"):
        Record.from_dict(payload | {"schema_version": 2})
    with pytest.raises(ValueError, match="unknown fields"):
        Record.from_dict(payload | {"future_field": True})
    with pytest.raises(TypeError, match="metrics must be a JSON array"):
        Record.from_dict(payload | {"metrics": tuple(payload["metrics"])})
    with pytest.raises(TypeError, match="payload must be a dict"):
        Record.from_dict([])
