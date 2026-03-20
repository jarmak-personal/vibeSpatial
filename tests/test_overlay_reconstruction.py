from __future__ import annotations

from vibespatial import OverlayOperation, fusion_plan_for_overlay, plan_overlay_reconstruction
from vibespatial.runtime.fusion import IntermediateDisposition


def test_overlay_reconstruction_plan_has_shared_topology_prefix() -> None:
    plan = plan_overlay_reconstruction("union")

    assert [stage.name for stage in plan.stages[:6]] == [
        "classify_segments",
        "emit_nodes",
        "split_segments",
        "group_half_edges",
        "walk_rings",
        "label_faces",
    ]
    assert plan.stages[0].requires_exact_signs is True
    assert plan.stages[5].requires_exact_signs is True


def test_overlay_reconstruction_selection_stage_changes_by_operation() -> None:
    union = plan_overlay_reconstruction(OverlayOperation.UNION)
    difference = plan_overlay_reconstruction(OverlayOperation.DIFFERENCE)
    symmetric = plan_overlay_reconstruction(OverlayOperation.SYMMETRIC_DIFFERENCE)

    assert "left=1 or right=1" in union.stages[-2].purpose
    assert "left=1 and right=0" in difference.stages[-2].purpose
    assert "left!=right" in symmetric.stages[-2].purpose


def test_overlay_reconstruction_emit_stage_persists_geometry_buffers() -> None:
    plan = plan_overlay_reconstruction("intersection")

    emit_stage = plan.stages[-1]
    assert emit_stage.disposition is IntermediateDisposition.PERSIST
    assert emit_stage.outputs == ("geometry_buffers",)
    assert emit_stage.preserves_stable_order is True


def test_overlay_fusion_plan_keeps_reconstruction_ephemeral_until_geometry_emit() -> None:
    plan = fusion_plan_for_overlay("symmetric_difference")

    assert len(plan.stages) >= 2
    assert plan.stages[-1].disposition is IntermediateDisposition.PERSIST
    assert plan.stages[-1].steps[-1].output_name == "geometry_buffers"
    assert plan.reason
