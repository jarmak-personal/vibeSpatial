from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StepKind(StrEnum):
    GEOMETRY = "geometry"
    DERIVED = "derived"
    FILTER = "filter"
    ORDERING = "ordering"
    INDEX = "index"
    MATERIALIZATION = "materialization"
    RASTER = "raster"


class IntermediateDisposition(StrEnum):
    EPHEMERAL = "ephemeral"
    PERSIST = "persist"
    BOUNDARY = "boundary"


@dataclass(frozen=True)
class PipelineStep:
    name: str
    kind: StepKind
    output_name: str
    output_rows_follow_input: bool = True
    reusable_output: bool = False
    materializes_host_output: bool = False
    requires_stable_row_order: bool = False


@dataclass(frozen=True)
class FusionStage:
    steps: tuple[PipelineStep, ...]
    disposition: IntermediateDisposition
    reason: str


@dataclass(frozen=True)
class FusionPlan:
    stages: tuple[FusionStage, ...]
    peak_memory_target_ratio: float
    reason: str


def plan_fusion(steps: tuple[PipelineStep, ...] | list[PipelineStep]) -> FusionPlan:
    normalized = tuple(steps)
    if not normalized:
        return FusionPlan(stages=(), peak_memory_target_ratio=1.0, reason="empty pipeline")

    stages: list[FusionStage] = []
    current: list[PipelineStep] = []

    for step in normalized:
        if step.materializes_host_output:
            if current:
                stages.append(
                    FusionStage(
                        steps=tuple(current),
                        disposition=IntermediateDisposition.EPHEMERAL,
                        reason="preceding device-local chain is fusible before host materialization",
                    )
                )
                current = []
            stages.append(
                FusionStage(
                    steps=(step,),
                    disposition=IntermediateDisposition.BOUNDARY,
                    reason="explicit host materialization is a hard fusion boundary",
                )
            )
            continue

        if step.reusable_output:
            if current:
                stages.append(
                    FusionStage(
                        steps=tuple(current),
                        disposition=IntermediateDisposition.EPHEMERAL,
                        reason="device-local chain can stay fused until a reusable structure is emitted",
                    )
                )
                current = []
            stages.append(
                FusionStage(
                    steps=(step,),
                    disposition=IntermediateDisposition.PERSIST,
                    reason="reusable structures such as indexes should persist rather than be fused away",
                )
            )
            continue

        if current and current[-1].requires_stable_row_order and not step.output_rows_follow_input:
            stages.append(
                FusionStage(
                    steps=tuple(current),
                    disposition=IntermediateDisposition.EPHEMERAL,
                    reason="row-order-sensitive consumer must finish before a reordering step",
                )
            )
            current = [step]
            continue

        current.append(step)

    if current:
        stages.append(
            FusionStage(
                steps=tuple(current),
                disposition=IntermediateDisposition.EPHEMERAL,
                reason="remaining device-local chain is fusible",
            )
        )

    return FusionPlan(
        stages=tuple(stages),
        peak_memory_target_ratio=1.5,
        reason="use a lightweight staged DAG: fuse ephemeral device-local chains, persist reusable structures, and stop at explicit materialization boundaries",
    )
