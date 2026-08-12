from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ._runtime import ExecutionMode
from .precision import KernelClass


class WorkloadShape(StrEnum):
    """Classification of how left and right geometry arrays relate in size.

    PAIRWISE:        left and right have the same length; element-wise ops.
    BROADCAST_RIGHT: right has length 1, left has length > 1; the single
                     right geometry is broadcast against every left row.
    SCALAR_RIGHT:    right is a scalar (not an array); skips pandas index
                     alignment entirely.

    BROADCAST_LEFT is intentionally omitted — no consumer exists today.
    INDEXED is intentionally omitted — gather-evaluate-scatter is a
    different computation model, not a workload shape.
    """

    PAIRWISE = "pairwise"
    BROADCAST_RIGHT = "broadcast_right"
    SCALAR_RIGHT = "scalar_right"


def detect_workload_shape(
    left_count: int,
    right_count: int | None,
) -> WorkloadShape:
    """Classify the workload shape for a binary operation."""
    if right_count is None:
        return WorkloadShape.SCALAR_RIGHT
    if right_count == 1 and left_count > 1:
        return WorkloadShape.BROADCAST_RIGHT
    if left_count == right_count:
        return WorkloadShape.PAIRWISE
    raise ValueError(
        f"Incompatible lengths: left={left_count}, right={right_count}. "
        "Use gpd.sjoin() for many-to-many operations."
    )


@dataclass(frozen=True)
class PhysicalWorkEstimate:
    """Shape-level work estimate for GPU dispatch.

    ADR-0046 makes row count a bootstrap signal, not the physical execution
    contract.  This carrier lets callers expose the actual work units that
    dominate the selected shape while preserving row-only compatibility for
    older dispatch paths.
    """

    row_count: int
    coordinate_count: int = 0
    coordinate_pair_count: int = 0
    segment_count: int = 0
    segment_pair_count: int = 0
    part_count: int = 0
    part_pair_count: int = 0
    ring_count: int = 0
    candidate_pair_count: int = 0
    relation_pair_count: int = 0
    group_count: int = 0
    output_row_count: int = 0
    output_byte_count: int = 0
    temporary_byte_count: int = 0
    primary_unit_count: int | None = None
    primary_unit_name: str = "row"

    def __post_init__(self) -> None:
        for name in (
            "row_count",
            "coordinate_count",
            "coordinate_pair_count",
            "segment_count",
            "segment_pair_count",
            "part_count",
            "part_pair_count",
            "ring_count",
            "candidate_pair_count",
            "relation_pair_count",
            "group_count",
            "output_row_count",
            "output_byte_count",
            "temporary_byte_count",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"PhysicalWorkEstimate {name} must be non-negative")
        if self.primary_unit_count is not None and int(self.primary_unit_count) < 0:
            raise ValueError("PhysicalWorkEstimate primary_unit_count must be non-negative")
        if not self.primary_unit_name:
            raise ValueError("PhysicalWorkEstimate primary_unit_name must be non-empty")

    @classmethod
    def from_rows(cls, row_count: int) -> PhysicalWorkEstimate:
        return cls(row_count=int(row_count))

    @classmethod
    def for_candidate_pairs(
        cls,
        *,
        row_count: int,
        candidate_pair_count: int,
        output_row_count: int = 0,
        output_byte_count: int = 0,
        temporary_byte_count: int = 0,
        primary_unit_name: str = "candidate-pair",
    ) -> PhysicalWorkEstimate:
        row_count = int(row_count)
        candidate_pair_count = int(candidate_pair_count)
        output_row_count = int(output_row_count)
        output_byte_count = int(output_byte_count)
        temporary_byte_count = int(temporary_byte_count)
        return cls(
            row_count=row_count,
            candidate_pair_count=candidate_pair_count,
            output_row_count=output_row_count,
            output_byte_count=output_byte_count,
            temporary_byte_count=temporary_byte_count,
            primary_unit_count=max(
                row_count,
                candidate_pair_count,
                output_row_count,
                output_byte_count // 64,
                temporary_byte_count // 128,
            ),
            primary_unit_name=primary_unit_name,
        )

    @classmethod
    def for_relation_pairs(
        cls,
        *,
        row_count: int,
        relation_pair_count: int,
        output_row_count: int = 0,
        output_byte_count: int = 0,
        temporary_byte_count: int = 0,
        primary_unit_name: str = "relation-pair",
    ) -> PhysicalWorkEstimate:
        row_count = int(row_count)
        relation_pair_count = int(relation_pair_count)
        output_row_count = int(output_row_count)
        output_byte_count = int(output_byte_count)
        temporary_byte_count = int(temporary_byte_count)
        return cls(
            row_count=row_count,
            relation_pair_count=relation_pair_count,
            output_row_count=output_row_count,
            output_byte_count=output_byte_count,
            temporary_byte_count=temporary_byte_count,
            primary_unit_count=max(
                row_count,
                relation_pair_count,
                output_row_count,
                output_byte_count // 64,
                temporary_byte_count // 128,
            ),
            primary_unit_name=primary_unit_name,
        )

    @property
    def is_row_only(self) -> bool:
        return (
            self.primary_unit_count is None
            and self.coordinate_count == 0
            and self.coordinate_pair_count == 0
            and self.segment_count == 0
            and self.segment_pair_count == 0
            and self.part_count == 0
            and self.part_pair_count == 0
            and self.ring_count == 0
            and self.candidate_pair_count == 0
            and self.relation_pair_count == 0
            and self.group_count == 0
            and self.output_row_count == 0
            and self.output_byte_count == 0
            and self.temporary_byte_count == 0
        )

    def dispatch_unit_count(self) -> int:
        if self.primary_unit_count is not None:
            return int(self.primary_unit_count)
        byte_units = max(
            int(self.output_byte_count) // 64,
            int(self.temporary_byte_count) // 128,
        )
        return max(
            int(self.row_count),
            int(self.coordinate_count),
            int(self.coordinate_pair_count),
            int(self.segment_count),
            int(self.segment_pair_count),
            int(self.part_count),
            int(self.part_pair_count),
            int(self.ring_count),
            int(self.candidate_pair_count),
            int(self.relation_pair_count),
            int(self.group_count),
            int(self.output_row_count),
            byte_units,
        )

    def dispatch_unit_name(self) -> str:
        if self.primary_unit_count is not None:
            return self.primary_unit_name
        return "row" if self.is_row_only else "work-unit"

    def live_device_byte_count(self) -> int:
        """Return the estimated simultaneously live output and scratch bytes."""
        return int(self.output_byte_count) + int(self.temporary_byte_count)

    def is_device_memory_admissible(
        self,
        available_device_bytes: int,
        *,
        budget_numerator: int = 1,
        budget_denominator: int = 2,
    ) -> bool:
        """Return whether this shape fits its reserved share of free device memory.

        A physical plan cannot consume all currently free memory: downstream
        output carriers, allocator fragmentation, and concurrently resident
        native inputs remain live. Callers may tune the reserved share for a
        shape, while the default admits at most half of currently available
        memory.
        """
        available = int(available_device_bytes)
        numerator = int(budget_numerator)
        denominator = int(budget_denominator)
        if available < 0:
            raise ValueError("available_device_bytes must be non-negative")
        if numerator <= 0 or denominator <= 0 or numerator > denominator:
            raise ValueError("device memory budget must be a fraction in (0, 1]")
        budget = available * numerator // denominator
        return self.live_device_byte_count() <= budget

    def telemetry_detail(self) -> str:
        parts = [
            f"rows={int(self.row_count)}",
            f"dispatch_units={self.dispatch_unit_count()}",
            f"dispatch_unit={self.dispatch_unit_name()}",
        ]
        for name, value in (
            ("coordinates", self.coordinate_count),
            ("coordinate_pairs", self.coordinate_pair_count),
            ("segments", self.segment_count),
            ("segment_pairs", self.segment_pair_count),
            ("parts", self.part_count),
            ("part_pairs", self.part_pair_count),
            ("rings", self.ring_count),
            ("candidate_pairs", self.candidate_pair_count),
            ("relation_pairs", self.relation_pair_count),
            ("groups", self.group_count),
            ("output_rows", self.output_row_count),
            ("output_bytes", self.output_byte_count),
            ("temporary_bytes", self.temporary_byte_count),
        ):
            if int(value) != 0:
                parts.append(f"{name}={int(value)}")
        return ", ".join(parts)


def _array_length(value: object) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(value)  # type: ignore[arg-type]


def estimate_physical_work_from_owned(
    owned: object,
    *,
    candidate_pair_count: int = 0,
    relation_pair_count: int = 0,
    group_count: int = 0,
    output_row_count: int = 0,
    output_byte_count: int = 0,
    temporary_byte_count: int = 0,
    primary_unit_count: int | None = None,
    primary_unit_name: str = "work-unit",
) -> PhysicalWorkEstimate:
    """Build a physical work estimate from an owned geometry carrier.

    The helper intentionally inspects only stable columnar shape metadata:
    row count and coordinate-buffer lengths.  Indexed views scale the compact
    base-buffer shape to their logical row count so dispatch estimates describe
    work executed after row indirection rather than merely the referenced
    storage.  It does not materialize geometry objects or host rows, so it is
    safe to call from dispatch planning.
    """
    row_count = int(getattr(owned, "row_count", 0))
    shape_owned = (
        getattr(owned, "_source_owned", owned)
        if bool(getattr(owned, "_is_lazy_grouped_union_owned", False))
        else owned
    )
    device_state = getattr(shape_owned, "device_state", None)
    device_families = getattr(device_state, "families", None)
    families = device_families or getattr(shape_owned, "families", None) or {}
    coordinate_count = 0
    ring_count = 0
    part_count = 0
    segment_count = 0
    for buffer in families.values():
        x = getattr(buffer, "x", None)
        if x is not None:
            n_coords = _array_length(x)
            coordinate_count += n_coords
            segment_count += n_coords
        ring_offsets = getattr(buffer, "ring_offsets", None)
        if ring_offsets is not None:
            ring_count += max(_array_length(ring_offsets) - 1, 0)
        part_offsets = getattr(buffer, "part_offsets", None)
        if part_offsets is not None:
            part_count += max(_array_length(part_offsets) - 1, 0)
    if bool(getattr(shape_owned, "is_indexed_view", False)):
        base = getattr(shape_owned, "_base", None)
        base_row_count = int(getattr(base, "row_count", 0))
        if row_count == 0:
            coordinate_count = 0
            segment_count = 0
            part_count = 0
            ring_count = 0
        elif base_row_count > 0:
            coordinate_count = (coordinate_count * row_count + base_row_count - 1) // base_row_count
            segment_count = (segment_count * row_count + base_row_count - 1) // base_row_count
            part_count = (part_count * row_count + base_row_count - 1) // base_row_count
            ring_count = (ring_count * row_count + base_row_count - 1) // base_row_count
    return PhysicalWorkEstimate(
        row_count=row_count,
        coordinate_count=coordinate_count,
        segment_count=segment_count,
        part_count=part_count,
        ring_count=ring_count,
        candidate_pair_count=int(candidate_pair_count),
        relation_pair_count=int(relation_pair_count),
        group_count=int(group_count),
        output_row_count=int(output_row_count),
        output_byte_count=int(output_byte_count),
        temporary_byte_count=int(temporary_byte_count),
        primary_unit_count=primary_unit_count,
        primary_unit_name=primary_unit_name,
    )


def estimate_spatial_index_work_from_owned(
    owned: object,
    *,
    output_row_count: int = 0,
    output_byte_count: int = 0,
    temporary_byte_count: int = 0,
    primary_unit_name: str = "spatial-index-unit",
) -> PhysicalWorkEstimate:
    """Estimate spatial-index build work from owned columnar geometry shape."""
    estimate = estimate_physical_work_from_owned(
        owned,
        output_row_count=output_row_count,
        output_byte_count=output_byte_count,
        temporary_byte_count=temporary_byte_count,
        primary_unit_name=primary_unit_name,
    )
    return PhysicalWorkEstimate(
        row_count=estimate.row_count,
        coordinate_count=estimate.coordinate_count,
        coordinate_pair_count=estimate.coordinate_pair_count,
        segment_count=estimate.segment_count,
        segment_pair_count=estimate.segment_pair_count,
        part_count=estimate.part_count,
        part_pair_count=estimate.part_pair_count,
        ring_count=estimate.ring_count,
        candidate_pair_count=estimate.candidate_pair_count,
        relation_pair_count=estimate.relation_pair_count,
        group_count=estimate.group_count,
        output_row_count=estimate.output_row_count,
        output_byte_count=estimate.output_byte_count,
        temporary_byte_count=estimate.temporary_byte_count,
        primary_unit_count=max(
            estimate.row_count,
            estimate.coordinate_count,
            estimate.segment_count,
            estimate.part_count,
            estimate.ring_count,
            estimate.output_row_count,
        ),
        primary_unit_name=primary_unit_name,
    )


def estimate_grouped_work_from_owned(
    owned: object,
    *,
    grouped: object | None = None,
    group_count: int | None = None,
    output_row_count: int | None = None,
    output_byte_count: int = 0,
    temporary_byte_count: int = 0,
    primary_unit_name: str = "grouped-segment",
) -> PhysicalWorkEstimate:
    """Estimate segmented grouped geometry work from native carriers.

    The estimate uses only owned columnar metadata and grouped carrier shape
    fields.  It does not inspect group labels or geometry objects on host, so
    callers can use it for dispatch telemetry before admissibility checks.
    """
    if group_count is not None:
        resolved_group_count = int(group_count)
    elif grouped is not None:
        try:
            resolved_group_count = int(grouped.resolved_group_count)  # type: ignore[attr-defined]
        except Exception:
            raw_group_count = getattr(grouped, "group_count", 0)
            resolved_group_count = 0 if raw_group_count is None else int(raw_group_count)
    else:
        resolved_group_count = 0

    output_rows = resolved_group_count if output_row_count is None else int(output_row_count)
    base = estimate_physical_work_from_owned(
        owned,
        group_count=resolved_group_count,
        output_row_count=output_rows,
        output_byte_count=int(output_byte_count),
        temporary_byte_count=int(temporary_byte_count),
        primary_unit_name=primary_unit_name,
    )
    inferred_output_bytes = int(output_byte_count)
    if inferred_output_bytes == 0:
        inferred_output_bytes = (
            int(base.coordinate_count) * 16 + int(base.ring_count) * 8 + int(output_rows) * 32
        )
    inferred_temporary_bytes = int(temporary_byte_count)
    if inferred_temporary_bytes == 0:
        inferred_temporary_bytes = (
            int(base.row_count) * 12 + max(int(resolved_group_count), int(output_rows), 1) * 8
        )
    return PhysicalWorkEstimate(
        row_count=base.row_count,
        coordinate_count=base.coordinate_count,
        coordinate_pair_count=base.coordinate_pair_count,
        segment_count=base.segment_count,
        segment_pair_count=base.segment_pair_count,
        part_count=base.part_count,
        part_pair_count=base.part_pair_count,
        ring_count=base.ring_count,
        group_count=resolved_group_count,
        output_row_count=output_rows,
        output_byte_count=inferred_output_bytes,
        temporary_byte_count=inferred_temporary_bytes,
        primary_unit_count=max(
            int(base.row_count),
            int(base.coordinate_count),
            int(base.segment_count),
            int(base.part_count),
            int(base.ring_count),
            int(resolved_group_count),
            int(output_rows),
            inferred_output_bytes // 64,
            inferred_temporary_bytes // 128,
        ),
        primary_unit_name=primary_unit_name,
    )


def estimate_pairwise_work_from_owned(
    left: object,
    right: object,
    *,
    workload: WorkloadShape | None = None,
    output_row_count: int | None = None,
    primary_unit_name: str = "pair-coordinate",
) -> PhysicalWorkEstimate:
    """Estimate aligned binary work from left/right owned columnar shape."""
    row_count = int(getattr(left, "row_count", 0))
    left_estimate = estimate_physical_work_from_owned(left)
    right_estimate = estimate_physical_work_from_owned(right)
    right_multiplier = (
        max(row_count, 1)
        if workload in (WorkloadShape.BROADCAST_RIGHT, WorkloadShape.SCALAR_RIGHT)
        else 1
    )
    coordinate_count = (
        int(left_estimate.coordinate_count)
        + int(right_estimate.coordinate_count) * right_multiplier
    )
    segment_count = (
        int(left_estimate.segment_count) + int(right_estimate.segment_count) * right_multiplier
    )
    ring_count = int(left_estimate.ring_count) + int(right_estimate.ring_count) * right_multiplier
    part_count = int(left_estimate.part_count) + int(right_estimate.part_count) * right_multiplier
    output_rows = row_count if output_row_count is None else int(output_row_count)
    return PhysicalWorkEstimate(
        row_count=row_count,
        coordinate_count=coordinate_count,
        segment_count=segment_count,
        part_count=part_count,
        ring_count=ring_count,
        output_row_count=output_rows,
        primary_unit_count=max(row_count, coordinate_count, segment_count, output_rows),
        primary_unit_name=primary_unit_name,
    )


def estimate_relation_pair_work_from_owned(
    left: object,
    right: object,
    *,
    pair_count: int,
    output_byte_count: int | None = None,
    temporary_byte_count: int | None = None,
    temporary_bytes_per_segment: int = 0,
    primary_unit_name: str = "relation-segment",
) -> PhysicalWorkEstimate:
    """Estimate many-to-many relation work before candidate rows are gathered."""
    pair_count = int(pair_count)

    def _expanded(owned: object) -> PhysicalWorkEstimate:
        if owned is None:
            return PhysicalWorkEstimate(
                row_count=pair_count,
                coordinate_count=pair_count,
                segment_count=pair_count,
            )
        source = estimate_physical_work_from_owned(owned)
        source_rows = max(source.row_count, 1)

        def _count(value: int) -> int:
            return pair_count * ((int(value) + source_rows - 1) // source_rows)

        return PhysicalWorkEstimate(
            row_count=pair_count,
            coordinate_count=_count(source.coordinate_count),
            segment_count=_count(source.segment_count),
            part_count=_count(source.part_count),
            ring_count=_count(source.ring_count),
        )

    left_work = _expanded(left)
    right_work = _expanded(right)
    coordinate_count = left_work.coordinate_count + right_work.coordinate_count
    segment_count = left_work.segment_count + right_work.segment_count
    part_count = left_work.part_count + right_work.part_count
    ring_count = left_work.ring_count + right_work.ring_count
    output_bytes = (
        coordinate_count * 16 + ring_count * 8 + pair_count * 32
        if output_byte_count is None
        else int(output_byte_count)
    )
    temporary_bytes = (
        segment_count * int(temporary_bytes_per_segment)
        if temporary_byte_count is None
        else int(temporary_byte_count)
    )
    return PhysicalWorkEstimate(
        row_count=pair_count,
        coordinate_count=coordinate_count,
        segment_count=segment_count,
        part_count=part_count,
        ring_count=ring_count,
        candidate_pair_count=pair_count,
        output_row_count=pair_count,
        output_byte_count=output_bytes,
        temporary_byte_count=temporary_bytes,
        primary_unit_count=max(
            pair_count,
            coordinate_count,
            segment_count,
            part_count,
            ring_count,
            output_bytes // 64,
            temporary_bytes // 128,
        ),
        primary_unit_name=primary_unit_name,
    )


def estimate_segment_pair_work_from_owned(
    owned: object,
    *,
    selected_row_count: int | None = None,
    output_row_count: int = 0,
    output_byte_count: int = 0,
    temporary_byte_count: int = 0,
    primary_unit_name: str = "segment-pair",
) -> PhysicalWorkEstimate:
    """Estimate within-geometry quadratic segment comparison work.

    The exact sum of per-geometry segment pairs lives behind device offsets
    for arbitrary-width geometry.  Dispatch planning must not synchronize to
    recover it, so this helper computes the balanced-distribution estimate
    from authoritative buffer lengths and family row counts.  Fixed-width
    carriers make that estimate exact; variable-width carriers retain the
    correct quadratic physical shape without a host metadata crossover.
    """
    base = estimate_physical_work_from_owned(
        owned,
        output_row_count=output_row_count,
        output_byte_count=output_byte_count,
        temporary_byte_count=temporary_byte_count,
        primary_unit_name=primary_unit_name,
    )

    shape_owned = (
        getattr(owned, "_source_owned", owned)
        if bool(getattr(owned, "_is_lazy_grouped_union_owned", False))
        else owned
    )
    device_state = getattr(shape_owned, "device_state", None)
    device_families = getattr(device_state, "families", None)
    families = device_families or getattr(shape_owned, "families", None) or {}
    segment_pair_count = 0
    for buffer in families.values():
        x = getattr(buffer, "x", None)
        geometry_offsets = getattr(buffer, "geometry_offsets", None)
        if x is None or geometry_offsets is None:
            continue
        coordinate_count = _array_length(x)
        geometry_count = max(_array_length(geometry_offsets) - 1, 0)
        if coordinate_count < 2 or geometry_count == 0:
            continue
        ring_offsets = getattr(buffer, "ring_offsets", None)
        part_offsets = getattr(buffer, "part_offsets", None)
        span_offsets = (
            ring_offsets
            if ring_offsets is not None
            else part_offsets
            if part_offsets is not None
            else geometry_offsets
        )
        span_count = max(_array_length(span_offsets) - 1, 1)
        segment_count = max(coordinate_count - span_count, 0)
        quotient, remainder = divmod(segment_count, geometry_count)
        segment_pair_count += (
            remainder * quotient * (quotient + 1) // 2
            + (geometry_count - remainder) * quotient * (quotient - 1) // 2
        )
    if bool(getattr(shape_owned, "is_indexed_view", False)):
        indexed_base = getattr(shape_owned, "_base", None)
        base_row_count = int(getattr(indexed_base, "row_count", 0))
        if base_row_count > 0:
            segment_pair_count = (
                segment_pair_count * base.row_count + base_row_count - 1
            ) // base_row_count
    selected_rows = base.row_count if selected_row_count is None else int(selected_row_count)
    if selected_rows < 0 or selected_rows > base.row_count:
        raise ValueError("selected_row_count must be within the owned row count")

    def _selected_count(value: int) -> int:
        if base.row_count == 0 or selected_rows == 0:
            return 0
        return (int(value) * selected_rows + base.row_count - 1) // base.row_count

    coordinate_count = _selected_count(base.coordinate_count)
    segment_count = _selected_count(base.segment_count)
    ring_count = _selected_count(base.ring_count)
    segment_pair_count = _selected_count(segment_pair_count)
    return PhysicalWorkEstimate(
        row_count=selected_rows,
        coordinate_count=coordinate_count,
        segment_count=segment_count,
        segment_pair_count=segment_pair_count,
        part_count=_selected_count(base.part_count),
        ring_count=ring_count,
        output_row_count=base.output_row_count,
        output_byte_count=base.output_byte_count,
        temporary_byte_count=base.temporary_byte_count,
        primary_unit_count=max(
            selected_rows,
            coordinate_count,
            segment_count,
            segment_pair_count,
            base.output_row_count,
            base.output_byte_count // 64,
            base.temporary_byte_count // 128,
        ),
        primary_unit_name=primary_unit_name,
    )


def estimate_pairwise_product_work_from_owned(
    left: object,
    right: object,
    *,
    pair_unit: str,
    output_row_count: int | None = None,
    output_byte_count: int = 0,
    temporary_byte_count: int = 0,
    primary_unit_name: str | None = None,
) -> PhysicalWorkEstimate:
    """Estimate aligned binary vertex- or segment-product work.

    Exact per-row products require reducing device offsets.  The balanced
    product preserves the quadratic shape from authoritative aggregate
    lengths without introducing a planning synchronization.  A one-row right
    carrier naturally models broadcast reuse; an already tiled right carrier
    is divided across the aligned row count.
    """
    if pair_unit not in {"coordinate", "segment"}:
        raise ValueError("pair_unit must be 'coordinate' or 'segment'")
    left_shape = estimate_physical_work_from_owned(left)
    right_shape = estimate_physical_work_from_owned(right)
    base = estimate_pairwise_work_from_owned(
        left,
        right,
        output_row_count=output_row_count,
        primary_unit_name=primary_unit_name or f"{pair_unit}-pair",
    )
    right_rows = int(getattr(right, "row_count", 0))
    divisor = max(base.row_count, 1) if right_rows == base.row_count else 1
    if pair_unit == "coordinate":
        left_count = left_shape.coordinate_count
        right_count = right_shape.coordinate_count
    else:
        left_count = left_shape.segment_count
        right_count = right_shape.segment_count
    pair_count = (left_count * right_count + divisor - 1) // divisor
    coordinate_pair_count = pair_count if pair_unit == "coordinate" else 0
    segment_pair_count = pair_count if pair_unit == "segment" else 0
    return PhysicalWorkEstimate(
        row_count=base.row_count,
        coordinate_count=base.coordinate_count,
        coordinate_pair_count=coordinate_pair_count,
        segment_count=base.segment_count,
        segment_pair_count=segment_pair_count,
        part_count=base.part_count,
        ring_count=base.ring_count,
        output_row_count=base.output_row_count,
        output_byte_count=int(output_byte_count),
        temporary_byte_count=int(temporary_byte_count),
        primary_unit_count=max(
            base.row_count,
            base.coordinate_count,
            base.segment_count,
            pair_count,
            base.output_row_count,
            int(output_byte_count) // 64,
            int(temporary_byte_count) // 128,
        ),
        primary_unit_name=primary_unit_name or f"{pair_unit}-pair",
    )


def estimate_part_pair_work_from_owned(
    owned: object,
    *,
    output_row_count: int = 0,
    output_byte_count: int = 0,
    temporary_byte_count: int = 0,
    primary_unit_name: str = "part-pair",
) -> PhysicalWorkEstimate:
    """Estimate within-geometry component endpoint graph work."""
    base = estimate_physical_work_from_owned(
        owned,
        output_row_count=output_row_count,
        output_byte_count=output_byte_count,
        temporary_byte_count=temporary_byte_count,
        primary_unit_name=primary_unit_name,
    )
    shape_owned = (
        getattr(owned, "_source_owned", owned)
        if bool(getattr(owned, "_is_lazy_grouped_union_owned", False))
        else owned
    )
    device_state = getattr(shape_owned, "device_state", None)
    device_families = getattr(device_state, "families", None)
    families = device_families or getattr(shape_owned, "families", None) or {}
    part_pair_count = 0
    for buffer in families.values():
        geometry_offsets = getattr(buffer, "geometry_offsets", None)
        part_offsets = getattr(buffer, "part_offsets", None)
        if geometry_offsets is None or part_offsets is None:
            continue
        geometry_count = max(_array_length(geometry_offsets) - 1, 0)
        part_count = max(_array_length(part_offsets) - 1, 0)
        if geometry_count == 0 or part_count < 2:
            continue
        quotient, remainder = divmod(part_count, geometry_count)
        part_pair_count += (
            remainder * quotient * (quotient + 1) // 2
            + (geometry_count - remainder) * quotient * (quotient - 1) // 2
        )
    if bool(getattr(shape_owned, "is_indexed_view", False)):
        indexed_base = getattr(shape_owned, "_base", None)
        base_row_count = int(getattr(indexed_base, "row_count", 0))
        if base_row_count > 0:
            part_pair_count = (
                part_pair_count * base.row_count + base_row_count - 1
            ) // base_row_count
    return PhysicalWorkEstimate(
        row_count=base.row_count,
        coordinate_count=base.coordinate_count,
        segment_count=base.segment_count,
        part_count=base.part_count,
        part_pair_count=part_pair_count,
        ring_count=base.ring_count,
        output_row_count=base.output_row_count,
        output_byte_count=base.output_byte_count,
        temporary_byte_count=base.temporary_byte_count,
        primary_unit_count=max(
            base.row_count,
            base.coordinate_count,
            base.part_count,
            part_pair_count,
            base.output_row_count,
            base.output_byte_count // 64,
            base.temporary_byte_count // 128,
        ),
        primary_unit_name=primary_unit_name,
    )


class DispatchDecision(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True)
class CrossoverPolicy:
    """Per-kernel crossover thresholds for AUTO dispatch.

    ``auto_min_rows`` is the pairwise threshold (left and right have the
    same length).  ``broadcast_min_rows`` is an optional lower threshold
    for broadcast workload shapes (BROADCAST_RIGHT / SCALAR_RIGHT) where
    the right-side geometry fits in L1 cache and is reused N times,
    making GPU profitable at much smaller N.
    """

    kernel_name: str
    kernel_class: KernelClass
    auto_min_rows: int
    reason: str
    broadcast_min_rows: int | None = None


# Pairwise thresholds by kernel class.
DEFAULT_CROSSOVER_POLICIES: dict[KernelClass, int] = {
    KernelClass.COARSE: 1_000,
    KernelClass.METRIC: 5_000,
    KernelClass.PREDICATE: 10_000,
    KernelClass.CONSTRUCTIVE: 50_000,
}

# Broadcast thresholds by kernel class.  These are lower than the
# pairwise thresholds because broadcast-right has perfect right-side
# data locality: one geometry, read once, reused N times from L1 cache.
DEFAULT_BROADCAST_CROSSOVER_POLICIES: dict[KernelClass, int] = {
    KernelClass.COARSE: 256,
    KernelClass.METRIC: 500,
    KernelClass.PREDICATE: 1_000,
    KernelClass.CONSTRUCTIVE: 500,
}

_KERNEL_CROSSOVER_OVERRIDES: dict[str, int] = {
    "normalize": 500,
    "point_clip": 10_000,
    "point_buffer": 0,
    "linestring_buffer": 5_000,
    "segment_classify": 4_096,
    "flat_index_build": 0,
    "bbox_overlap_candidates": 0,
    "point_regular_grid_candidates": 0,
    "point_box_query": 0,
    "spatial_index_knn": 0,
    "clip_scalar_mask_bounds_filter": 50_000,
    "make_valid_repair": 2_000,
    "polygon_centroid": 500,
    "geometry_area": 500,
    "geometry_length": 500,
    "intersects": 0,
    "contains": 0,
    "within": 0,
    "touches": 0,
    "covers": 0,
    "covered_by": 0,
    "overlaps": 0,
    "disjoint": 0,
    "contains_properly": 0,
}

_BROADCAST_SHAPES = frozenset({WorkloadShape.BROADCAST_RIGHT, WorkloadShape.SCALAR_RIGHT})


def effective_crossover_threshold(
    policy: CrossoverPolicy,
    workload_shape: WorkloadShape | None = None,
) -> int:
    if workload_shape is not None and workload_shape in _BROADCAST_SHAPES:
        return (
            policy.broadcast_min_rows
            if policy.broadcast_min_rows is not None
            else policy.auto_min_rows // 10
        )
    return policy.auto_min_rows


def default_crossover_policy(
    kernel_name: str,
    kernel_class: KernelClass | str,
) -> CrossoverPolicy:
    normalized_class = (
        kernel_class if isinstance(kernel_class, KernelClass) else KernelClass(kernel_class)
    )
    override = _KERNEL_CROSSOVER_OVERRIDES.get(kernel_name)
    broadcast_threshold = DEFAULT_BROADCAST_CROSSOVER_POLICIES[normalized_class]
    if override is not None:
        return CrossoverPolicy(
            kernel_name=kernel_name,
            kernel_class=normalized_class,
            auto_min_rows=override,
            reason=f"kernel-specific crossover override for {kernel_name} is {override} rows",
            broadcast_min_rows=broadcast_threshold,
        )
    threshold = DEFAULT_CROSSOVER_POLICIES[normalized_class]
    return CrossoverPolicy(
        kernel_name=kernel_name,
        kernel_class=normalized_class,
        auto_min_rows=threshold,
        reason=f"provisional auto crossover for {normalized_class.value} kernels is {threshold} rows",
        broadcast_min_rows=broadcast_threshold,
    )


def select_dispatch_for_rows(
    *,
    requested_mode: ExecutionMode | str,
    row_count: int,
    policy: CrossoverPolicy,
    gpu_available: bool,
    workload_shape: WorkloadShape | None = None,
) -> DispatchDecision:
    """Select CPU or GPU execution based on row count and crossover policy.

    When *workload_shape* is ``BROADCAST_RIGHT`` or ``SCALAR_RIGHT``, the
    effective threshold is ``policy.broadcast_min_rows`` (or
    ``policy.auto_min_rows // 10`` if the policy does not set a broadcast
    threshold).  This reflects the fact that broadcast workloads have
    perfect right-side data locality and benefit from GPU execution at
    much smaller N than pairwise workloads.
    """
    return select_dispatch_for_estimate(
        requested_mode=requested_mode,
        work_estimate=PhysicalWorkEstimate.from_rows(row_count),
        policy=policy,
        gpu_available=gpu_available,
        workload_shape=workload_shape,
    )


def select_dispatch_for_estimate(
    *,
    requested_mode: ExecutionMode | str,
    work_estimate: PhysicalWorkEstimate,
    policy: CrossoverPolicy,
    gpu_available: bool,
    workload_shape: WorkloadShape | None = None,
) -> DispatchDecision:
    """Select CPU or GPU execution from an ADR-0046 physical work estimate."""
    mode = (
        requested_mode
        if isinstance(requested_mode, ExecutionMode)
        else ExecutionMode(requested_mode)
    )

    if mode is ExecutionMode.CPU:
        return DispatchDecision.CPU

    if mode is ExecutionMode.GPU:
        if not gpu_available:
            raise RuntimeError("GPU execution was requested, but no GPU runtime is available")
        return DispatchDecision.GPU

    if not gpu_available:
        return DispatchDecision.CPU

    threshold = effective_crossover_threshold(policy, workload_shape)

    if work_estimate.dispatch_unit_count() < threshold:
        return DispatchDecision.CPU

    return DispatchDecision.GPU
