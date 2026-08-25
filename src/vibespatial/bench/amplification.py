"""Versioned work-amplification evidence schema.

The types in this module describe profiling evidence only.  They deliberately
do not participate in runtime selection or production dispatch.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

SCHEMA_VERSION = 1
WORK_AMPLIFICATION_FIELD = "work_amplification"
_LINEAGE_ALIAS_PATTERN = re.compile(r"source_[0-9]+\Z")
_MEASURED_STATUSES = frozenset(("exact", "sampled", "derived"))
_NULL_STATUSES = frozenset(("unavailable", "invalid"))


class MetricStatus(StrEnum):
    """Provenance and availability state for an amplification measurement."""

    EXACT = "exact"
    SAMPLED = "sampled"
    DERIVED = "derived"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


def _nonempty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _status(value: object, *, field_name: str = "status") -> MetricStatus:
    if isinstance(value, MetricStatus):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a MetricStatus or string")
    try:
        return MetricStatus(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in MetricStatus)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def _number_or_none(value: object, *, field_name: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be an int, float, or None")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _validate_status_value(
    *,
    status: MetricStatus,
    value: int | float | None,
    reason: str | None,
    owner: str,
) -> None:
    if status.value in _MEASURED_STATUSES:
        if value is None:
            raise ValueError(f"{owner}.value must be present when status is {status.value}")
        if reason is not None:
            raise ValueError(f"{owner}.reason must be None when status is {status.value}")
        return
    if status.value in _NULL_STATUSES:
        if value is not None:
            raise ValueError(f"{owner}.value must be None when status is {status.value}")
        if reason is None:
            raise ValueError(f"{owner}.reason is required when status is {status.value}")


def _optional_reason(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, field_name=field_name)


def _require_object(payload: object, *, owner: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"{owner} payload must be a dict")
    for key in payload:
        if not isinstance(key, str):
            raise TypeError(f"{owner} payload keys must be strings")
    return payload


def _check_keys(
    payload: dict[str, Any],
    *,
    owner: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{owner} payload is missing fields: {', '.join(sorted(missing))}")
    unknown = payload.keys() - required - optional
    if unknown:
        raise ValueError(f"{owner} payload has unknown fields: {', '.join(sorted(unknown))}")


def _freeze_json(value: object, *, path: str) -> object:
    """Validate a JSON-native value and return an immutable representation."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, list):
        return tuple(_freeze_json(item, path=f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(value, dict):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} object keys must be strings")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    raise TypeError(
        f"{path} must contain only JSON-native dict, list, string, number, boolean, or null values"
    )


def _thaw_json(value: object) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class Metric:
    """A named physical-work measurement."""

    name: str
    value: int | float | None
    status: MetricStatus | str
    source: str | None = None
    unit: str | None = None
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_string(self.name, field_name="metric.name"))
        object.__setattr__(
            self,
            "value",
            _number_or_none(self.value, field_name=f"metric[{self.name}].value"),
        )
        normalized_status = _status(self.status, field_name=f"metric[{self.name}].status")
        object.__setattr__(self, "status", normalized_status)
        normalized_source = None
        if self.source is not None:
            normalized_source = _nonempty_string(
                self.source,
                field_name=f"metric[{self.name}].source",
            )
        if normalized_status.value in _MEASURED_STATUSES and normalized_source is None:
            raise ValueError(
                f"metric[{self.name}].source is required when status is {normalized_status.value}"
            )
        object.__setattr__(self, "source", normalized_source)
        if self.unit is not None:
            object.__setattr__(
                self,
                "unit",
                _nonempty_string(self.unit, field_name=f"metric[{self.name}].unit"),
            )
        normalized_reason = _optional_reason(
            self.reason,
            field_name=f"metric[{self.name}].reason",
        )
        object.__setattr__(self, "reason", normalized_reason)
        if not isinstance(self.metadata, dict):
            raise TypeError(f"metric[{self.name}].metadata must be a JSON-native dict")
        frozen_metadata = _freeze_json(
            self.metadata,
            path=f"metric[{self.name}].metadata",
        )
        object.__setattr__(self, "metadata", frozen_metadata)
        if normalized_status is MetricStatus.SAMPLED:
            sampling = frozen_metadata.get("sampling")
            if not isinstance(sampling, Mapping) or not sampling:
                raise ValueError(
                    f"metric[{self.name}].metadata.sampling must be a non-empty JSON object "
                    "when status is sampled"
                )
        _validate_status_value(
            status=normalized_status,
            value=self.value,
            reason=normalized_reason,
            owner=f"metric[{self.name}]",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "status": self.status.value,
            "source": self.source,
            "unit": self.unit,
            "reason": self.reason,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: object) -> Metric:
        data = _require_object(payload, owner="Metric")
        _check_keys(
            data,
            owner="Metric",
            required=frozenset(("name", "value", "status")),
            optional=frozenset(("source", "unit", "reason", "metadata")),
        )
        return cls(
            name=data["name"],
            value=data["value"],
            status=data["status"],
            source=data.get("source"),
            unit=data.get("unit"),
            reason=data.get("reason"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class Ratio:
    """A derived ratio with its auditable raw inputs."""

    name: str
    numerator_metric: str
    numerator_value: int | float | None
    denominator_metric: str
    denominator_value: int | float | None
    value: float | None
    status: MetricStatus | str = MetricStatus.DERIVED
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_string(self.name, field_name="ratio.name"))
        object.__setattr__(
            self,
            "numerator_metric",
            _nonempty_string(
                self.numerator_metric,
                field_name=f"ratio[{self.name}].numerator_metric",
            ),
        )
        object.__setattr__(
            self,
            "denominator_metric",
            _nonempty_string(
                self.denominator_metric,
                field_name=f"ratio[{self.name}].denominator_metric",
            ),
        )
        numerator = _number_or_none(
            self.numerator_value,
            field_name=f"ratio[{self.name}].numerator_value",
        )
        denominator = _number_or_none(
            self.denominator_value,
            field_name=f"ratio[{self.name}].denominator_value",
        )
        value = _number_or_none(self.value, field_name=f"ratio[{self.name}].value")
        object.__setattr__(self, "numerator_value", numerator)
        object.__setattr__(self, "denominator_value", denominator)
        object.__setattr__(self, "value", None if value is None else float(value))
        normalized_status = _status(self.status, field_name=f"ratio[{self.name}].status")
        object.__setattr__(self, "status", normalized_status)
        normalized_reason = _optional_reason(
            self.reason,
            field_name=f"ratio[{self.name}].reason",
        )
        object.__setattr__(self, "reason", normalized_reason)
        _validate_status_value(
            status=normalized_status,
            value=self.value,
            reason=normalized_reason,
            owner=f"ratio[{self.name}]",
        )
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.status is MetricStatus.UNAVAILABLE:
            return
        if self.numerator_value is None or self.denominator_value is None:
            raise ValueError(
                f"ratio[{self.name}] raw inputs are required unless status is unavailable"
            )
        if self.denominator_value == 0:
            if not (
                self.status is MetricStatus.INVALID
                and self.value is None
                and self.reason == "denominator_zero"
            ):
                raise ValueError(
                    f"ratio[{self.name}] with a zero denominator must be invalid with "
                    "reason denominator_zero"
                )
            return
        if self.status is MetricStatus.INVALID:
            return
        expected = self.numerator_value / self.denominator_value
        if not math.isclose(self.value, expected, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f"ratio[{self.name}].value does not match its raw inputs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "numerator_metric": self.numerator_metric,
            "numerator_value": self.numerator_value,
            "denominator_metric": self.denominator_metric,
            "denominator_value": self.denominator_value,
            "value": self.value,
            "status": self.status.value,
            "reason": self.reason,
            "denominator_zero": self.denominator_zero,
        }

    @property
    def denominator_zero(self) -> bool:
        return (
            self.status is MetricStatus.INVALID
            and self.denominator_value == 0
            and self.reason == "denominator_zero"
        )

    @property
    def numerator_name(self) -> str:
        """Compatibility spelling for callers that treat the reference as a name."""
        return self.numerator_metric

    @property
    def denominator_name(self) -> str:
        """Compatibility spelling for callers that treat the reference as a name."""
        return self.denominator_metric

    @classmethod
    def from_dict(cls, payload: object) -> Ratio:
        data = _require_object(payload, owner="Ratio")
        _check_keys(
            data,
            owner="Ratio",
            required=frozenset(
                (
                    "name",
                    "numerator_metric",
                    "numerator_value",
                    "denominator_metric",
                    "denominator_value",
                    "value",
                    "status",
                    "denominator_zero",
                )
            ),
            optional=frozenset(("reason",)),
        )
        denominator_zero = data["denominator_zero"]
        if not isinstance(denominator_zero, bool):
            raise TypeError("Ratio.denominator_zero must be a boolean")
        ratio = cls(
            name=data["name"],
            numerator_metric=data["numerator_metric"],
            numerator_value=data["numerator_value"],
            denominator_metric=data["denominator_metric"],
            denominator_value=data["denominator_value"],
            value=data["value"],
            status=data["status"],
            reason=data.get("reason"),
        )
        if denominator_zero is not ratio.denominator_zero:
            raise ValueError("Ratio.denominator_zero does not match denominator_value")
        return ratio


def make_ratio(
    name: str,
    *,
    numerator_metric: str,
    numerator_value: int | float,
    denominator_metric: str,
    denominator_value: int | float,
) -> Ratio:
    """Build a derived ratio without fabricating a value for zero denominators."""
    numerator = _number_or_none(numerator_value, field_name="numerator_value")
    denominator = _number_or_none(denominator_value, field_name="denominator_value")
    if numerator is None:
        raise TypeError("numerator_value must be an int or float")
    if denominator is None:
        raise TypeError("denominator_value must be an int or float")
    if denominator == 0:
        return Ratio(
            name=name,
            numerator_metric=numerator_metric,
            numerator_value=numerator,
            denominator_metric=denominator_metric,
            denominator_value=denominator,
            value=None,
            status=MetricStatus.INVALID,
            reason="denominator_zero",
        )
    return Ratio(
        name=name,
        numerator_metric=numerator_metric,
        numerator_value=numerator,
        denominator_metric=denominator_metric,
        denominator_value=denominator,
        value=numerator / denominator,
        status=MetricStatus.DERIVED,
    )


def ratio_from_metrics(name: str, *, numerator: Metric, denominator: Metric) -> Ratio:
    """Build a ratio from two available metrics, retaining their names and values."""
    if not isinstance(numerator, Metric) or not isinstance(denominator, Metric):
        raise TypeError("numerator and denominator must be Metric instances")
    if numerator.value is None or denominator.value is None:
        return Ratio(
            name=name,
            numerator_metric=numerator.name,
            numerator_value=numerator.value,
            denominator_metric=denominator.name,
            denominator_value=denominator.value,
            value=None,
            status=MetricStatus.UNAVAILABLE,
            reason="input_metric_unavailable",
        )
    return make_ratio(
        name,
        numerator_metric=numerator.name,
        numerator_value=numerator.value,
        denominator_metric=denominator.name,
        denominator_value=denominator.value,
    )


@dataclass(frozen=True, slots=True)
class Lineage:
    """A trace-local, serialization-safe alias for one source role."""

    alias: str
    role: str = "source"

    def __post_init__(self) -> None:
        alias = _nonempty_string(self.alias, field_name="lineage.alias")
        if _LINEAGE_ALIAS_PATTERN.fullmatch(alias) is None:
            raise ValueError("lineage.alias must use the trace-local source_<number> form")
        object.__setattr__(self, "alias", alias)
        object.__setattr__(
            self,
            "role",
            _nonempty_string(self.role, field_name="lineage.role"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"alias": self.alias, "role": self.role}

    @classmethod
    def from_dict(cls, payload: object) -> Lineage:
        data = _require_object(payload, owner="Lineage")
        _check_keys(
            data,
            owner="Lineage",
            required=frozenset(("alias",)),
            optional=frozenset(("role",)),
        )
        return cls(alias=data["alias"], role=data.get("role", "source"))


class LineageAliaser:
    """Assign deterministic aliases within one trace without exposing raw tokens."""

    __slots__ = ("_aliases",)

    def __init__(self) -> None:
        self._aliases: dict[tuple[type[object], object], str] = {}

    def alias(self, raw_token: object, *, role: str = "source") -> Lineage:
        if raw_token is None:
            raise TypeError("raw lineage token must not be None")
        try:
            hash(raw_token)
        except TypeError as exc:
            raise TypeError("raw lineage token must be hashable") from exc
        key = (type(raw_token), raw_token)
        alias = self._aliases.get(key)
        if alias is None:
            alias = f"source_{len(self._aliases)}"
            self._aliases[key] = alias
        return Lineage(alias=alias, role=role)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(aliases={len(self._aliases)})"


def _typed_tuple(
    value: object,
    *,
    item_type: type[Any],
    field_name: str,
    allow_empty: bool,
) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list or tuple")
    result = tuple(value)
    if not allow_empty and not result:
        raise ValueError(f"{field_name} must not be empty")
    if any(not isinstance(item, item_type) for item in result):
        raise TypeError(f"{field_name} items must be {item_type.__name__} instances")
    return result


@dataclass(frozen=True, slots=True)
class Record:
    """One immutable work-amplification metric-family record."""

    operation: str
    stage: str
    metric_family: str
    physical_shape: str
    consumer_kind: str
    device: str
    measurement_boundary: str
    boundary_role: str
    instrumentation_level: int
    source_lineage: tuple[Lineage, ...]
    metrics: tuple[Metric, ...]
    ratios: tuple[Ratio, ...]
    semantic_contract: Mapping[str, Any]
    schema_version: int = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "operation",
            "stage",
            "metric_family",
            "physical_shape",
            "consumer_kind",
            "device",
            "measurement_boundary",
            "boundary_role",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonempty_string(getattr(self, field_name), field_name=f"record.{field_name}"),
            )
        if isinstance(self.instrumentation_level, bool) or not isinstance(
            self.instrumentation_level, int
        ):
            raise TypeError("record.instrumentation_level must be an integer")
        if self.instrumentation_level not in (0, 1, 2):
            raise ValueError("record.instrumentation_level must be 0, 1, or 2")
        lineage = _typed_tuple(
            self.source_lineage,
            item_type=Lineage,
            field_name="record.source_lineage",
            allow_empty=False,
        )
        metrics = _typed_tuple(
            self.metrics,
            item_type=Metric,
            field_name="record.metrics",
            allow_empty=False,
        )
        ratios = _typed_tuple(
            self.ratios,
            item_type=Ratio,
            field_name="record.ratios",
            allow_empty=True,
        )
        object.__setattr__(self, "source_lineage", lineage)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "ratios", ratios)
        if not isinstance(self.semantic_contract, dict):
            raise TypeError("record.semantic_contract must be a JSON-native dict")
        object.__setattr__(
            self,
            "semantic_contract",
            _freeze_json(self.semantic_contract, path="record.semantic_contract"),
        )
        self._validate_names()

    def _validate_names(self) -> None:
        metric_by_name = {metric.name: metric for metric in self.metrics}
        if len(metric_by_name) != len(self.metrics):
            raise ValueError("record.metrics contains duplicate names")
        ratio_names = {ratio.name for ratio in self.ratios}
        if len(ratio_names) != len(self.ratios):
            raise ValueError("record.ratios contains duplicate names")
        for ratio in self.ratios:
            for input_name, input_value in (
                (ratio.numerator_metric, ratio.numerator_value),
                (ratio.denominator_metric, ratio.denominator_value),
            ):
                metric = metric_by_name.get(input_name)
                if metric is None:
                    raise ValueError(
                        f"record ratio[{ratio.name}] references missing metric {input_name!r}"
                    )
                if metric.value != input_value:
                    raise ValueError(
                        f"record ratio[{ratio.name}] raw value does not match metric {input_name!r}"
                    )

    def to_dict(self) -> dict[str, Any]:
        """Return one JSON-native record for ``metadata['work_amplification']``."""
        return {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "stage": self.stage,
            "metric_family": self.metric_family,
            "physical_shape": self.physical_shape,
            "consumer_kind": self.consumer_kind,
            "device": self.device,
            "measurement_boundary": self.measurement_boundary,
            "boundary_role": self.boundary_role,
            "instrumentation_level": self.instrumentation_level,
            "source_lineage": [item.to_dict() for item in self.source_lineage],
            "metrics": [item.to_dict() for item in self.metrics],
            "ratios": [item.to_dict() for item in self.ratios],
            "semantic_contract": _thaw_json(self.semantic_contract),
        }

    @classmethod
    def from_dict(cls, payload: object) -> Record:
        data = dict(_require_object(payload, owner="Record"))
        required = frozenset(
            (
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
            )
        )
        _check_keys(data, owner="Record", required=required)
        version = data["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise TypeError("Record.schema_version must be an integer")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported work-amplification schema_version {version}; expected {SCHEMA_VERSION}"
            )
        for field_name, item_type in (
            ("source_lineage", Lineage),
            ("metrics", Metric),
            ("ratios", Ratio),
        ):
            if not isinstance(data[field_name], list):
                raise TypeError(f"Record.{field_name} must be a JSON array")
            data[field_name] = tuple(item_type.from_dict(item) for item in data[field_name])
        return cls(
            operation=data["operation"],
            stage=data["stage"],
            metric_family=data["metric_family"],
            physical_shape=data["physical_shape"],
            consumer_kind=data["consumer_kind"],
            device=data["device"],
            measurement_boundary=data["measurement_boundary"],
            boundary_role=data["boundary_role"],
            instrumentation_level=data["instrumentation_level"],
            source_lineage=data["source_lineage"],
            metrics=data["metrics"],
            ratios=data["ratios"],
            semantic_contract=data["semantic_contract"],
        )


__all__ = [
    "SCHEMA_VERSION",
    "WORK_AMPLIFICATION_FIELD",
    "Lineage",
    "LineageAliaser",
    "Metric",
    "MetricStatus",
    "Ratio",
    "Record",
    "make_ratio",
    "ratio_from_metrics",
]
