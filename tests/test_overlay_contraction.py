from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import shapely
from shapely.geometry import box

from vibespatial import has_gpu_runtime
from vibespatial.api import read_file
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.overlay.contraction import (
    overlay_contraction_owned,
    summarize_overlay_contraction_canary,
)

contract_module = importlib.import_module("vibespatial.overlay.contract")
DATA_ROOT = Path(__file__).resolve().parents[1] / "tests" / "upstream" / "geopandas" / "tests" / "data"
LEFT_NYBB = DATA_ROOT / "nybb_16a.zip"
RIGHT_NYBB = DATA_ROOT / "overlay" / "nybb_qgis" / "polydf2.shp"


def _load_nybb_owned():
    left = read_file(f"zip://{LEFT_NYBB}")
    right = read_file(str(RIGHT_NYBB))
    return left.geometry.values.to_owned(), right.geometry.values.to_owned()


def test_overlay_contraction_nybb_pair_alignment_shape_is_stable() -> None:
    left_owned, right_owned = _load_nybb_owned()
    summary = summarize_overlay_contraction_canary(left_owned, right_owned, dispatch_mode="cpu")

    assert summary["pair_count"] == 15
    assert summary["left_indices"] == [0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    assert summary["right_indices"] == [0, 1, 2, 3, 6, 7, 4, 5, 6, 4, 5, 6, 7, 8, 9]
    assert summary["left_aligned_segment_count"] == 207657
    assert summary["right_aligned_segment_count"] == 975


@pytest.mark.gpu
def test_overlay_contraction_nybb_exact_refine_shape_is_stable() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left_owned, right_owned = _load_nybb_owned()
    summary = summarize_overlay_contraction_canary(left_owned, right_owned)

    assert summary["candidate_pairs"] == 1044
    assert summary["point_intersection_count"] == 94
    assert summary["row_point_intersection_counts"] == [2, 6, 0, 2, 8, 10, 4, 2, 2, 6, 10, 4, 10, 20, 8]
    assert summary["max_exact_events"] == 29276


@pytest.mark.gpu
def test_overlay_contraction_nybb_microcell_bounds_are_tractable() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left_owned, right_owned = _load_nybb_owned()
    summary = summarize_overlay_contraction_canary(left_owned, right_owned)
    microcells = summary["microcells"]

    assert microcells["max_row_microcell_upper_bound"] == 346593
    assert microcells["total_microcell_upper_bound"] == 2233636
    assert microcells["max_active_segment_count"] == 48
    assert len(microcells["row_microcell_upper_bounds"]) == 15


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("intersection", box(1.0, 0.0, 2.0, 2.0)),
        ("union", box(0.0, 0.0, 3.0, 2.0)),
        ("difference", box(0.0, 0.0, 1.0, 2.0)),
    ],
)
def test_overlay_contraction_owned_matches_simple_rectangles(operation, expected) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)])
    right = from_shapely_geometries([box(1.0, 0.0, 3.0, 2.0)])

    result = overlay_contraction_owned(left, right, operation=operation)
    got = result.to_shapely()[0]

    assert shapely.equals(got, expected)


@pytest.mark.gpu
def test_overlay_contraction_gpu_path_bypasses_host_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    monkeypatch.setattr(
        contract_module,
        "_contract_overlay_microcells_host",
        lambda *args, **kwargs: pytest.fail("GPU contraction should not need the host helper on simple rectangles"),
    )

    left = from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)])
    right = from_shapely_geometries([box(1.0, 0.0, 3.0, 2.0)])

    result = overlay_contraction_owned(left, right, operation="intersection")

    assert shapely.equals(result.to_shapely()[0], box(1.0, 0.0, 2.0, 2.0))
