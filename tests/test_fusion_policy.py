from __future__ import annotations

from vibespatial import (
    IntermediateDisposition,
    PipelineStep,
    StepKind,
    default_fusible_sequences,
    plan_fusion,
)


def test_bounds_key_sort_stays_in_one_ephemeral_stage() -> None:
    plan = plan_fusion(default_fusible_sequences()["bounds_sfc_sort"])

    assert len(plan.stages) == 1
    assert plan.stages[0].disposition is IntermediateDisposition.EPHEMERAL
    assert [step.name for step in plan.stages[0].steps] == ["bounds", "morton_keys", "sort"]


def test_materialization_breaks_fusion_chain() -> None:
    plan = plan_fusion(default_fusible_sequences()["predicate_filter_materialize"])

    assert len(plan.stages) == 2
    assert plan.stages[0].disposition is IntermediateDisposition.EPHEMERAL
    assert plan.stages[1].disposition is IntermediateDisposition.BOUNDARY
    assert plan.stages[1].steps[0].name == "to_pandas"


def test_reusable_index_persists_instead_of_being_fused_away() -> None:
    plan = plan_fusion(default_fusible_sequences()["build_and_query_index"])

    assert len(plan.stages) == 2
    assert plan.stages[0].disposition is IntermediateDisposition.PERSIST
    assert plan.stages[0].steps[0].name == "build_index"
    assert plan.stages[1].disposition is IntermediateDisposition.EPHEMERAL


def test_row_order_sensitive_step_stops_before_reordering() -> None:
    plan = plan_fusion(
        (
            PipelineStep(
                name="mask_consumer",
                kind=StepKind.FILTER,
                output_name="mask_consumer",
                requires_stable_row_order=True,
            ),
            PipelineStep(
                name="sort",
                kind=StepKind.ORDERING,
                output_name="permutation",
                output_rows_follow_input=False,
            ),
        )
    )

    assert len(plan.stages) == 2
    assert [step.name for step in plan.stages[0].steps] == ["mask_consumer"]
    assert [step.name for step in plan.stages[1].steps] == ["sort"]
