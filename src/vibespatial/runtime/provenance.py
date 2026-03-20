"""Provenance tagging and rewrite system for spatial operation optimization.

Tags intermediate GeometryArray results with metadata about the operation that
created them, enabling downstream operations to recognize patterns and substitute
cheaper equivalents automatically.

See ADR-0039 for the design rationale.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import shapely

from ._runtime import ExecutionMode
from .dispatch import record_dispatch_event
from .event_log import append_event_record

if TYPE_CHECKING:
    from vibespatial.api.geometry_array import GeometryArray


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceTag:
    """Metadata about how a GeometryArray was created."""

    operation: str
    params: tuple[tuple[str, Any], ...]
    source_geom_types: frozenset[str] | None
    source_array: Any | None = field(default=None, repr=False, compare=False, hash=False)
    reason: str = ""

    def get_param(self, key: str, default: Any = None) -> Any:
        for k, v in self.params:
            if k == key:
                return v
        return default


@dataclass(frozen=True)
class RewriteRule:
    """Declarative rewrite rule specification."""

    name: str
    input_pattern: str
    consumer_operation: str
    preconditions: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class RewriteEvent:
    """Record of a provenance-based rewrite that fired."""

    rule_name: str
    surface: str
    original_operation: str
    rewritten_operation: str
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REWRITE_REGISTRY: dict[tuple[str, str], list[RewriteRule]] = defaultdict(list)


def register_rewrite(rule: RewriteRule) -> None:
    """Add a rewrite rule to the registry."""
    key = (rule.input_pattern, rule.consumer_operation)
    if rule not in REWRITE_REGISTRY[key]:
        REWRITE_REGISTRY[key].append(rule)


def get_rewrite_candidates(tag_operation: str, consumer_operation: str) -> tuple[RewriteRule, ...]:
    """Look up candidate rewrite rules for a (producer, consumer) pair."""
    return tuple(REWRITE_REGISTRY.get((tag_operation, consumer_operation), ()))


# ---------------------------------------------------------------------------
# Event logging (parallel to dispatch.py)
# ---------------------------------------------------------------------------

_REWRITE_EVENTS: deque[RewriteEvent] = deque(maxlen=512)


def record_rewrite_event(
    *,
    rule_name: str,
    surface: str,
    original_operation: str,
    rewritten_operation: str,
    reason: str,
    detail: str = "",
) -> RewriteEvent:
    event = RewriteEvent(
        rule_name=rule_name,
        surface=surface,
        original_operation=original_operation,
        rewritten_operation=rewritten_operation,
        reason=reason,
        detail=detail,
    )
    _REWRITE_EVENTS.append(event)
    append_event_record("rewrite", event.to_dict())
    return event


def get_rewrite_events(*, clear: bool = False) -> list[RewriteEvent]:
    events = list(_REWRITE_EVENTS)
    if clear:
        _REWRITE_EVENTS.clear()
    return events


def clear_rewrite_events() -> None:
    _REWRITE_EVENTS.clear()


# ---------------------------------------------------------------------------
# Geometry type inference
# ---------------------------------------------------------------------------

_POINT_FAMILIES = frozenset({"point", "multipoint"})


def infer_geom_types(ga: GeometryArray) -> frozenset[str] | None:
    """Infer geometry family names from a GeometryArray.

    Uses OwnedGeometryArray tags (zero-cost) when available, otherwise
    falls back to shapely.get_type_id.
    """
    from vibespatial.geometry.owned import TAG_FAMILIES

    if ga._owned is not None:
        tags = ga._owned.tags
        if hasattr(tags, "get"):
            # CuPy array -- pull to host
            tags = np.asarray(tags)
        unique_tags = np.unique(tags)
        return frozenset(
            TAG_FAMILIES[int(t)].value for t in unique_tags if int(t) >= 0
        )
    if ga._shapely_data is not None and len(ga._shapely_data) > 0:
        type_ids = shapely.get_type_id(ga._shapely_data)
        valid = type_ids[~np.isnan(type_ids.astype(float))] if type_ids.dtype == object else type_ids
        unique_ids = np.unique(valid[valid >= 0])
        _SHAPELY_TYPE_NAMES = {
            0: "point",
            1: "linestring",
            2: "linestring",  # LinearRing
            3: "polygon",
            4: "multipoint",
            5: "multilinestring",
            6: "multipolygon",
            7: "polygon",  # GeometryCollection -> treat as mixed
        }
        return frozenset(
            _SHAPELY_TYPE_NAMES.get(int(tid), "unknown") for tid in unique_ids
        )
    return None


def _is_point_only(geom_types: frozenset[str] | None) -> bool:
    """Check if geometry types are exclusively point-based."""
    if geom_types is None:
        return False
    return len(geom_types) > 0 and geom_types <= _POINT_FAMILIES


# ---------------------------------------------------------------------------
# Provenance tag construction helpers
# ---------------------------------------------------------------------------


def make_buffer_tag(
    source: GeometryArray,
    distance: float | int,
    cap_style: str,
    join_style: str,
    single_sided: bool,
    quad_segs: int,
) -> ProvenanceTag:
    """Create a provenance tag for a buffer operation."""
    geom_types = infer_geom_types(source)
    return ProvenanceTag(
        operation="buffer",
        params=(
            ("distance", distance),
            ("cap_style", cap_style),
            ("join_style", join_style),
            ("single_sided", single_sided),
            ("quad_segs", quad_segs),
        ),
        source_geom_types=geom_types,
        source_array=source,
        reason=f"buffer({distance}) on {sorted(geom_types) if geom_types else 'unknown'} geometries",
    )


# ---------------------------------------------------------------------------
# Rewrite rule definitions
# ---------------------------------------------------------------------------

R1_BUFFER_INTERSECTS = RewriteRule(
    name="R1_buffer_intersects_to_dwithin",
    input_pattern="buffer",
    consumer_operation="intersects",
    preconditions=("point_only", "positive_distance", "round_cap", "round_join", "not_single_sided"),
    reason="buffer(r).intersects(Y) == dwithin(Y, r) for isotropic point buffers",
)

R5_BUFFER_ZERO = RewriteRule(
    name="R5_buffer_zero_identity",
    input_pattern="buffer",
    consumer_operation="__self__",
    preconditions=("distance_zero",),
    reason="buffer(0) is the identity operation",
)

R6_BUFFER_CHAIN = RewriteRule(
    name="R6_buffer_chain_merge",
    input_pattern="buffer",
    consumer_operation="buffer",
    preconditions=("positive_radii", "same_style", "point_only"),
    reason="buffer(a).buffer(b) == buffer(a+b) for positive radii on point geometries",
)

register_rewrite(R1_BUFFER_INTERSECTS)
register_rewrite(R5_BUFFER_ZERO)
register_rewrite(R6_BUFFER_CHAIN)


# ---------------------------------------------------------------------------
# Precondition checks
# ---------------------------------------------------------------------------


def _r1_preconditions_met(tag: ProvenanceTag) -> bool:
    """Check R1 preconditions: point-only, positive distance, round cap/join, not single-sided."""
    if not _is_point_only(tag.source_geom_types):
        return False
    distance = tag.get_param("distance")
    if not isinstance(distance, int | float) or distance <= 0:
        return False
    if tag.get_param("cap_style") != "round":
        return False
    if tag.get_param("join_style") != "round":
        return False
    if tag.get_param("single_sided"):
        return False
    if tag.source_array is None:
        return False
    return True


def _r6_preconditions_met(tag: ProvenanceTag, new_distance: float | int, new_cap: str, new_join: str) -> bool:
    """Check R6 preconditions: positive radii, same style, point-only."""
    if not _is_point_only(tag.source_geom_types):
        return False
    prev_distance = tag.get_param("distance")
    if not isinstance(prev_distance, int | float) or prev_distance <= 0:
        return False
    if not isinstance(new_distance, int | float) or new_distance <= 0:
        return False
    if tag.get_param("cap_style") != new_cap:
        return False
    if tag.get_param("join_style") != new_join:
        return False
    if tag.get_param("single_sided"):
        return False
    if tag.source_array is None:
        return False
    return True


# ---------------------------------------------------------------------------
# Rewrite attempt functions
# ---------------------------------------------------------------------------


def attempt_rewrite_buffer_intersects(
    tag: ProvenanceTag,
    left: GeometryArray,
    right: Any,
    **kwargs: Any,
) -> np.ndarray | None:
    """Attempt R1: buffer(r).intersects(Y) -> dwithin(Y, r)."""
    if not _r1_preconditions_met(tag):
        return None

    distance = tag.get_param("distance")
    source = tag.source_array

    # Use the source (pre-buffer) geometry for the dwithin check
    from vibespatial.spatial.distance_owned import evaluate_geopandas_dwithin

    left_arg = source._owned if source._owned is not None else source._data
    if hasattr(right, "_owned") and right._owned is not None:
        right_arg = right._owned
    elif hasattr(right, "_data"):
        right_arg = right._data
    else:
        right_arg = right

    result = evaluate_geopandas_dwithin(left_arg, right_arg, distance)
    if result is not None:
        record_rewrite_event(
            rule_name="R1_buffer_intersects_to_dwithin",
            surface="geopandas.array.intersects",
            original_operation="intersects",
            rewritten_operation="dwithin",
            reason=R1_BUFFER_INTERSECTS.reason,
            detail=f"buffer_distance={distance}",
        )
        record_dispatch_event(
            surface="geopandas.array.intersects",
            operation="intersects",
            implementation="provenance_rewrite_R1_dwithin",
            reason="provenance rewrite: buffer(r).intersects -> dwithin(r)",
            detail=f"buffer_distance={distance}",
            selected=ExecutionMode.AUTO,
        )
        return result

    # GPU path unavailable, try Shapely dwithin fallback
    source_data = source._data
    right_data = right._data if hasattr(right, "_data") else right
    result = shapely.dwithin(source_data, right_data, distance=distance)
    if result is not None:
        record_rewrite_event(
            rule_name="R1_buffer_intersects_to_dwithin",
            surface="geopandas.array.intersects",
            original_operation="intersects",
            rewritten_operation="dwithin",
            reason=R1_BUFFER_INTERSECTS.reason,
            detail=f"buffer_distance={distance}, fallback=shapely",
        )
        record_dispatch_event(
            surface="geopandas.array.intersects",
            operation="intersects",
            implementation="provenance_rewrite_R1_dwithin_shapely",
            reason="provenance rewrite: buffer(r).intersects -> shapely.dwithin(r)",
            detail=f"buffer_distance={distance}",
            selected=ExecutionMode.CPU,
        )
        return result

    return None


# ---------------------------------------------------------------------------
# Central rewrite dispatcher (called from GeometryArray._binary_method)
# ---------------------------------------------------------------------------

# Map of (input_pattern, consumer_operation) -> attempt function
_REWRITE_ATTEMPTS: dict[str, Any] = {
    "R1_buffer_intersects_to_dwithin": attempt_rewrite_buffer_intersects,
}


def attempt_provenance_rewrite(
    op: str,
    left: GeometryArray,
    right: Any,
    **kwargs: Any,
) -> np.ndarray | None:
    """Check if the left operand's provenance enables a cheaper rewrite.

    Returns the rewritten result array, or None if no rewrite applies.
    """
    if not hasattr(left, "_provenance") or left._provenance is None:
        return None

    tag = left._provenance
    candidates = get_rewrite_candidates(tag.operation, op)
    if not candidates:
        return None

    for rule in candidates:
        attempt_fn = _REWRITE_ATTEMPTS.get(rule.name)
        if attempt_fn is None:
            continue
        result = attempt_fn(tag, left, right, **kwargs)
        if result is not None:
            return result

    return None
