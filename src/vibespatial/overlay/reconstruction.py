from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from vibespatial.runtime.fusion import IntermediateDisposition, PipelineStep, StepKind, plan_fusion


class OverlayOperation(StrEnum):
    INTERSECTION = "intersection"
    UNION = "union"
    DIFFERENCE = "difference"
    SYMMETRIC_DIFFERENCE = "symmetric_difference"
    IDENTITY = "identity"


class ReconstructionPrimitive(StrEnum):
    SEGMENT_CLASSIFICATION = "segment_classification"
    NODE_EMIT = "node_emit"
    SEGMENT_SPLIT = "segment_split"
    STABLE_SORT = "stable_sort"
    RUN_LENGTH_ENCODE = "run_length_encode"
    REDUCE_BY_KEY = "reduce_by_key"
    PREFIX_SUM = "prefix_sum"
    COMPACT = "compact"
    SCATTER = "scatter"
    GATHER = "gather"


@dataclass(frozen=True)
class ReconstructionStage:
    name: str
    primitive: ReconstructionPrimitive
    purpose: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    cccl_mapping: tuple[str, ...]
    disposition: IntermediateDisposition
    requires_exact_signs: bool = False
    preserves_stable_order: bool = False


@dataclass(frozen=True)
class OverlayReconstructionPlan:
    operation: OverlayOperation
    stages: tuple[ReconstructionStage, ...]
    fusion_steps: tuple[PipelineStep, ...]
    reason: str


def _shared_prefix() -> tuple[ReconstructionStage, ...]:
    return (
        ReconstructionStage(
            name="classify_segments",
            primitive=ReconstructionPrimitive.SEGMENT_CLASSIFICATION,
            purpose="Classify candidate segment pairs with exact fallback before reconstruction.",
            inputs=("candidate_segment_pairs",),
            outputs=("segment_classes", "intersection_points"),
            cccl_mapping=("transform", "DeviceSelect", "scatter"),
            disposition=IntermediateDisposition.EPHEMERAL,
            requires_exact_signs=True,
        ),
        ReconstructionStage(
            name="emit_nodes",
            primitive=ReconstructionPrimitive.NODE_EMIT,
            purpose="Emit segment endpoints and newly created intersection nodes into one event stream.",
            inputs=("segment_classes", "intersection_points"),
            outputs=("node_events",),
            cccl_mapping=("gather", "scatter"),
            disposition=IntermediateDisposition.EPHEMERAL,
        ),
        ReconstructionStage(
            name="split_segments",
            primitive=ReconstructionPrimitive.SEGMENT_SPLIT,
            purpose="Split original segments at emitted nodes to produce atomic directed edges.",
            inputs=("node_events",),
            outputs=("directed_edges",),
            cccl_mapping=("stable_sort", "prefix_sum", "scatter"),
            disposition=IntermediateDisposition.EPHEMERAL,
        ),
        ReconstructionStage(
            name="group_half_edges",
            primitive=ReconstructionPrimitive.STABLE_SORT,
            purpose="Stable-sort directed edges by source node and turning order for deterministic traversal.",
            inputs=("directed_edges",),
            outputs=("sorted_half_edges",),
            cccl_mapping=("stable_sort", "reduce_by_key"),
            disposition=IntermediateDisposition.EPHEMERAL,
            preserves_stable_order=True,
        ),
        ReconstructionStage(
            name="walk_rings",
            primitive=ReconstructionPrimitive.RUN_LENGTH_ENCODE,
            purpose="Traverse sorted half-edges into candidate rings and open chains.",
            inputs=("sorted_half_edges",),
            outputs=("candidate_rings", "open_chains"),
            cccl_mapping=("prefix_sum", "reduce_by_key", "scatter"),
            disposition=IntermediateDisposition.EPHEMERAL,
        ),
        ReconstructionStage(
            name="label_faces",
            primitive=ReconstructionPrimitive.REDUCE_BY_KEY,
            purpose="Label each candidate face or chain with left/right source coverage for overlay semantics.",
            inputs=("candidate_rings", "open_chains"),
            outputs=("labeled_faces", "labeled_chains"),
            cccl_mapping=("gather", "reduce_by_key"),
            disposition=IntermediateDisposition.EPHEMERAL,
            requires_exact_signs=True,
        ),
    )


def _selection_stage(operation: OverlayOperation) -> ReconstructionStage:
    selectors = {
        OverlayOperation.INTERSECTION: ("faces with left=1 and right=1", "selected_faces"),
        OverlayOperation.UNION: ("faces with left=1 or right=1", "selected_faces"),
        OverlayOperation.DIFFERENCE: ("faces with left=1 and right=0", "selected_faces"),
        OverlayOperation.SYMMETRIC_DIFFERENCE: ("faces with left!=right", "selected_faces"),
        OverlayOperation.IDENTITY: ("left faces split by right coverage", "selected_faces"),
    }
    purpose, output = selectors[operation]
    return ReconstructionStage(
        name="select_overlay_faces",
        primitive=ReconstructionPrimitive.COMPACT,
        purpose=f"Select {purpose} according to the public overlay operation.",
        inputs=("labeled_faces", "labeled_chains"),
        outputs=(output,),
        cccl_mapping=("DeviceSelect", "scatter"),
        disposition=IntermediateDisposition.EPHEMERAL,
    )


def _emit_stage(operation: OverlayOperation) -> ReconstructionStage:
    if operation in {OverlayOperation.INTERSECTION, OverlayOperation.UNION, OverlayOperation.IDENTITY}:
        purpose = "Emit final polygon, line, and point geometry buffers in deterministic row order."
    else:
        purpose = "Emit final face and chain buffers for geometry-producing difference-style outputs."
    return ReconstructionStage(
        name="emit_geometry",
        primitive=ReconstructionPrimitive.SCATTER,
        purpose=purpose,
        inputs=("selected_faces",),
        outputs=("geometry_buffers",),
        cccl_mapping=("prefix_sum", "scatter", "gather"),
        disposition=IntermediateDisposition.PERSIST,
        preserves_stable_order=True,
    )


def plan_overlay_reconstruction(
    operation: OverlayOperation | str,
) -> OverlayReconstructionPlan:
    normalized = operation if isinstance(operation, OverlayOperation) else OverlayOperation(operation)
    stages = (*_shared_prefix(), _selection_stage(normalized), _emit_stage(normalized))
    fusion_steps = (
        PipelineStep(name="segment_classes", kind=StepKind.GEOMETRY, output_name="segment_classes"),
        PipelineStep(name="node_events", kind=StepKind.DERIVED, output_name="node_events"),
        PipelineStep(name="directed_edges", kind=StepKind.DERIVED, output_name="directed_edges"),
        PipelineStep(name="sorted_half_edges", kind=StepKind.ORDERING, output_name="sorted_half_edges"),
        PipelineStep(name="candidate_rings", kind=StepKind.GEOMETRY, output_name="candidate_rings"),
        PipelineStep(name="labeled_faces", kind=StepKind.FILTER, output_name="labeled_faces"),
        PipelineStep(name="selected_faces", kind=StepKind.FILTER, output_name="selected_faces"),
        PipelineStep(
            name="geometry_buffers",
            kind=StepKind.GEOMETRY,
            output_name="geometry_buffers",
            reusable_output=True,
        ),
    )
    return OverlayReconstructionPlan(
        operation=normalized,
        stages=stages,
        fusion_steps=fusion_steps,
        reason=(
            "Reconstruct overlay outputs with staged node emission, edge splitting, stable sorting, "
            "face labeling, and operation-specific selection so the GPU path can map onto CCCL compaction, "
            "prefix-sum, and reduce-by-key primitives."
        ),
    )


def fusion_plan_for_overlay(operation: OverlayOperation | str):
    return plan_fusion(plan_overlay_reconstruction(operation).fusion_steps)
