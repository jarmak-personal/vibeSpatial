"""Overlay workload-family planning and strategy detection.

The public overlay surface and the GPU-owned overlay executor both need the
same answer to the same question: what is the true constructive workload we
are about to run? This module turns row counts, candidate-pair evidence, and
surface semantics into one immutable planning object so execution-family
selection happens at the public boundary or chunk boundary instead of drifting
into ad hoc mid-pipeline branches.

Uses the shared ``WorkloadShape`` enum from ``vibespatial.runtime.crossover``
for broadcast detection (nsf.5), falling back to overlay-specific detection
for ``broadcast_left`` (which the shared enum intentionally omits).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from vibespatial.runtime.fusion import PipelineStep, StepKind, plan_fusion

if TYPE_CHECKING:
    from vibespatial.geometry.owned import OwnedGeometryArray
    from vibespatial.runtime.crossover import WorkloadShape
    from vibespatial.runtime.fusion import FusionPlan


class OverlayExecutionFamily(StrEnum):
    CLIP_REWRITE = "clip_rewrite"
    BROADCAST_RIGHT_INTERSECTION = "broadcast_right_intersection"
    BROADCAST_RIGHT_DIFFERENCE = "broadcast_right_difference"
    COVERAGE_UNION = "coverage_union"
    GROUPED_UNION = "grouped_union"
    GENERIC_RECONSTRUCTION = "generic_reconstruction"


class OverlayTopologyClass(StrEnum):
    MASK_CLIP = "mask_clip"
    BROADCAST_MASK = "broadcast_mask"
    COVERAGE = "coverage"
    GROUPED_SET = "grouped_set"
    RECONSTRUCTION = "reconstruction"


class OverlayResultShape(StrEnum):
    LEFT_ROWS = "left_rows"
    PAIRWISE_ROWS = "pairwise_rows"
    GROUPED_LEFT_ROWS = "grouped_left_rows"
    SURFACE_PARTS = "surface_parts"


@dataclass(frozen=True)
class OverlayStrategy:
    """Immutable constructive planning object for overlay execution."""

    name: str  # e.g. "broadcast_right", "per_group", "clip_rewrite"
    many_side: str  # "left", "right", or "both" (for N-vs-M)
    reason: str  # human-readable explanation for provenance
    workload_shape: WorkloadShape | None = None  # shared enum, None for overlay-only shapes
    operation: str = "intersection"
    topology_class: OverlayTopologyClass = OverlayTopologyClass.RECONSTRUCTION
    semantics_flags: frozenset[str] = frozenset()
    result_shape: OverlayResultShape = OverlayResultShape.SURFACE_PARTS
    execution_family: OverlayExecutionFamily = OverlayExecutionFamily.GENERIC_RECONSTRUCTION
    fusion_plan: FusionPlan | None = None

    @property
    def workload_shape_label(self) -> str:
        return self.workload_shape.value if self.workload_shape is not None else self.name

    @property
    def semantics_label(self) -> str:
        if not self.semantics_flags:
            return "none"
        return ",".join(sorted(self.semantics_flags))

    @property
    def fusion_stage_labels(self) -> str:
        if self.fusion_plan is None or not self.fusion_plan.stages:
            return "none"
        labels: list[str] = []
        for stage in self.fusion_plan.stages:
            step_names = "+".join(step.name for step in stage.steps)
            labels.append(f"{stage.disposition.value}:{step_names}")
        return ";".join(labels)

    def telemetry_detail(
        self,
        *,
        left_rows: int,
        right_rows: int,
        candidate_pair_count: int,
    ) -> str:
        return (
            f"left={left_rows}, right={right_rows}, "
            f"pairs={candidate_pair_count}, strategy={self.name}, "
            f"workload_shape={self.workload_shape_label}, "
            f"execution_family={self.execution_family.value}, "
            f"topology_class={self.topology_class.value}, "
            f"result_shape={self.result_shape.value}, "
            f"semantics={self.semantics_label}, "
            f"fusion_stages={self.fusion_stage_labels}"
        )


def _detect_overlay_workload_shape(left_rows: int, right_rows: int):
    from vibespatial.runtime.crossover import detect_workload_shape

    try:
        return detect_workload_shape(left_rows, right_rows)
    except ValueError:
        return None


def _overlay_fusion_steps(
    family: OverlayExecutionFamily,
    *,
    clip_rewrite: bool,
) -> tuple[PipelineStep, ...]:
    if clip_rewrite or family is OverlayExecutionFamily.CLIP_REWRITE:
        return (
            PipelineStep(
                name="mask_bounds_filter",
                kind=StepKind.FILTER,
                output_name="mask_rows",
                reusable_output=True,
            ),
            PipelineStep(
                name="emit_clip_slice",
                kind=StepKind.GEOMETRY,
                output_name="clipped_geometry",
            ),
            PipelineStep(
                name="constructive_export",
                kind=StepKind.MATERIALIZATION,
                output_name="geodataframe",
                materializes_host_output=True,
            ),
        )

    if family is OverlayExecutionFamily.BROADCAST_RIGHT_INTERSECTION:
        return (
            PipelineStep(
                name="candidate_pairs",
                kind=StepKind.INDEX,
                output_name="candidate_pairs",
                reusable_output=True,
            ),
            PipelineStep(
                name="containment_bypass",
                kind=StepKind.FILTER,
                output_name="broadcast_partition",
            ),
            PipelineStep(
                name="batched_sh_clip",
                kind=StepKind.GEOMETRY,
                output_name="simple_remainder",
            ),
            PipelineStep(
                name="row_isolated_overlay",
                kind=StepKind.GEOMETRY,
                output_name="overlay_fragments",
            ),
            PipelineStep(
                name="constructive_export",
                kind=StepKind.MATERIALIZATION,
                output_name="native_tabular",
                materializes_host_output=True,
            ),
        )

    if family is OverlayExecutionFamily.BROADCAST_RIGHT_DIFFERENCE:
        return (
            PipelineStep(
                name="candidate_pairs",
                kind=StepKind.INDEX,
                output_name="candidate_pairs",
                reusable_output=True,
            ),
            PipelineStep(
                name="containment_bypass",
                kind=StepKind.FILTER,
                output_name="broadcast_partition",
            ),
            PipelineStep(
                name="row_isolated_overlay",
                kind=StepKind.GEOMETRY,
                output_name="overlay_fragments",
            ),
            PipelineStep(
                name="constructive_export",
                kind=StepKind.MATERIALIZATION,
                output_name="native_tabular",
                materializes_host_output=True,
            ),
        )

    if family is OverlayExecutionFamily.COVERAGE_UNION:
        return (
            PipelineStep(
                name="candidate_pairs",
                kind=StepKind.INDEX,
                output_name="candidate_pairs",
                reusable_output=True,
            ),
            PipelineStep(
                name="row_isolated_overlay",
                kind=StepKind.GEOMETRY,
                output_name="coverage_union_rows",
                requires_stable_row_order=True,
            ),
            PipelineStep(
                name="constructive_export",
                kind=StepKind.MATERIALIZATION,
                output_name="native_tabular",
                materializes_host_output=True,
            ),
        )

    if family is OverlayExecutionFamily.GROUPED_UNION:
        return (
            PipelineStep(
                name="candidate_pairs",
                kind=StepKind.INDEX,
                output_name="candidate_pairs",
                reusable_output=True,
            ),
            PipelineStep(
                name="group_offsets",
                kind=StepKind.ORDERING,
                output_name="group_offsets",
                reusable_output=True,
            ),
            PipelineStep(
                name="segmented_union",
                kind=StepKind.GEOMETRY,
                output_name="grouped_right_union",
                requires_stable_row_order=True,
            ),
            PipelineStep(
                name="row_isolated_overlay",
                kind=StepKind.GEOMETRY,
                output_name="grouped_overlay_rows",
            ),
            PipelineStep(
                name="constructive_export",
                kind=StepKind.MATERIALIZATION,
                output_name="native_tabular",
                materializes_host_output=True,
            ),
        )

    return (
        PipelineStep(
            name="candidate_pairs",
            kind=StepKind.INDEX,
            output_name="candidate_pairs",
            reusable_output=True,
        ),
        PipelineStep(
            name="row_isolated_overlay",
            kind=StepKind.GEOMETRY,
            output_name="overlay_rows",
        ),
        PipelineStep(
            name="constructive_export",
            kind=StepKind.MATERIALIZATION,
            output_name="native_tabular",
            materializes_host_output=True,
        ),
    )


def _overlay_semantics_flags(
    *,
    clip_rewrite: bool,
    keep_geom_type: bool | None,
    prefer_exact_polygon_gpu: bool,
    preserve_lower_dim_results: bool,
) -> frozenset[str]:
    flags: set[str] = set()
    if clip_rewrite:
        flags.add("geometry_only_mask")
    if keep_geom_type is True:
        flags.add("keep_geom_type")
    elif keep_geom_type is False:
        flags.add("keep_all_geometry_types")
    if prefer_exact_polygon_gpu:
        flags.add("prefer_exact_polygon_gpu")
    if preserve_lower_dim_results:
        flags.add("preserve_lower_dim_results")
    return frozenset(flags)


def plan_overlay_operation(
    *,
    left_rows: int,
    right_rows: int,
    how: str,
    candidate_pair_count: int = 0,
    clip_rewrite: bool = False,
    keep_geom_type: bool | None = None,
    prefer_exact_polygon_gpu: bool = False,
    preserve_lower_dim_results: bool = False,
) -> OverlayStrategy:
    """Plan the constructive workload family for an overlay operation."""
    if left_rows < 0 or right_rows < 0:
        raise ValueError("overlay row counts must be non-negative")

    workload_shape = _detect_overlay_workload_shape(left_rows, right_rows)
    semantics_flags = _overlay_semantics_flags(
        clip_rewrite=clip_rewrite,
        keep_geom_type=keep_geom_type,
        prefer_exact_polygon_gpu=prefer_exact_polygon_gpu,
        preserve_lower_dim_results=preserve_lower_dim_results,
    )

    if clip_rewrite:
        family = OverlayExecutionFamily.CLIP_REWRITE
        topology_class = OverlayTopologyClass.MASK_CLIP
        result_shape = OverlayResultShape.LEFT_ROWS
        name = "clip_rewrite"
        many_side = "left"
        reason = (
            f"single-mask clip rewrite: {left_rows} left rows vs {right_rows} right rows, "
            f"how={how}"
        )
    elif workload_shape is not None and workload_shape.value == "broadcast_right" and how == "intersection":
        family = OverlayExecutionFamily.BROADCAST_RIGHT_INTERSECTION
        topology_class = OverlayTopologyClass.BROADCAST_MASK
        result_shape = OverlayResultShape.LEFT_ROWS
        name = "broadcast_right"
        many_side = "left"
        reason = (
            f"N-vs-1 broadcast-right intersection: {left_rows} left rows vs 1 right row, "
            f"how={how}, pairs={candidate_pair_count}"
        )
    elif workload_shape is not None and workload_shape.value == "broadcast_right" and how == "difference":
        family = OverlayExecutionFamily.BROADCAST_RIGHT_DIFFERENCE
        topology_class = OverlayTopologyClass.BROADCAST_MASK
        result_shape = OverlayResultShape.LEFT_ROWS
        name = "broadcast_right"
        many_side = "left"
        reason = (
            f"N-vs-1 broadcast-right difference: {left_rows} left rows vs 1 right row, "
            f"how={how}, pairs={candidate_pair_count}"
        )
    elif how == "union" and workload_shape is not None and workload_shape.value == "pairwise":
        family = OverlayExecutionFamily.COVERAGE_UNION
        topology_class = OverlayTopologyClass.COVERAGE
        result_shape = OverlayResultShape.PAIRWISE_ROWS
        name = "per_group"
        many_side = "both"
        reason = (
            f"pairwise coverage-style union: {left_rows} left rows vs {right_rows} right rows, "
            f"pairs={candidate_pair_count}"
        )
    elif how in ("difference", "symmetric_difference") and right_rows > 1 and candidate_pair_count != 0:
        family = OverlayExecutionFamily.GROUPED_UNION
        topology_class = OverlayTopologyClass.GROUPED_SET
        result_shape = OverlayResultShape.GROUPED_LEFT_ROWS
        if left_rows == 1 and right_rows > 1:
            name = "broadcast_left"
            many_side = "right"
        else:
            name = "per_group"
            many_side = "both"
        reason = (
            f"grouped right-neighbour union before {how}: {left_rows} left rows vs "
            f"{right_rows} right rows, pairs={candidate_pair_count}"
        )
    else:
        family = OverlayExecutionFamily.GENERIC_RECONSTRUCTION
        topology_class = OverlayTopologyClass.RECONSTRUCTION
        result_shape = OverlayResultShape.SURFACE_PARTS
        if left_rows == 1 and right_rows > 1:
            name = "broadcast_left"
            many_side = "right"
            reason = (
                f"1-vs-N reconstruction: 1 left row vs {right_rows} right rows, "
                f"how={how}, pairs={candidate_pair_count}"
            )
        else:
            name = "per_group"
            many_side = "both"
            reason = (
                f"generic reconstruction: {left_rows} left rows vs {right_rows} right rows, "
                f"how={how}, pairs={candidate_pair_count}"
            )

    fusion_plan = plan_fusion(_overlay_fusion_steps(family, clip_rewrite=clip_rewrite))
    return OverlayStrategy(
        name=name,
        many_side=many_side,
        reason=reason,
        workload_shape=workload_shape,
        operation=how,
        topology_class=topology_class,
        semantics_flags=semantics_flags,
        result_shape=result_shape,
        execution_family=family,
        fusion_plan=fusion_plan,
    )


def select_overlay_strategy(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    how: str,
    *,
    candidate_pair_count: int = 0,
) -> OverlayStrategy:
    """Detect workload shape and select the overlay execution family."""
    return plan_overlay_operation(
        left_rows=left.row_count,
        right_rows=right.row_count,
        how=how,
        candidate_pair_count=candidate_pair_count,
    )
