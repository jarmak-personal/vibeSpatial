from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box

from vibespatial import from_shapely_geometries, has_gpu_runtime
from vibespatial.overlay.contract import contract_overlay_microcells
from vibespatial.overlay.contraction_reconstruct import (
    _coalesce_selected_microcells,
    reconstruct_overlay_from_microcells,
)
from vibespatial.overlay.microcells import (
    OverlayMicrocellBands,
    OverlayMicrocellLabels,
    build_and_label_overlay_microcells,
    build_overlay_microcell_bands,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None


def _to_host_bool(arr) -> np.ndarray:
    if cp is not None and hasattr(arr, "__cuda_array_interface__"):
        return cp.asnumpy(arr).astype(bool, copy=False)
    return np.asarray(arr, dtype=bool)


def _to_host_array(arr) -> np.ndarray:
    if cp is not None and hasattr(arr, "__cuda_array_interface__"):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def test_overlay_microcell_bands_partition_simple_rectangle_overlap() -> None:
    left = from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)])
    right = from_shapely_geometries([box(1.0, 0.0, 3.0, 2.0)])

    bands = build_overlay_microcell_bands(left, right, dispatch_mode="cpu")

    assert bands.count == 3
    assert bands.row_indices.tolist() == [0, 0, 0]
    assert bands.interval_indices.tolist() == [0, 1, 2]
    assert np.allclose(bands.representative_x, [0.5, 1.5, 2.5])
    assert np.allclose(bands.representative_y, [1.0, 1.0, 1.0])
    assert np.allclose(bands.x_left, [0.0, 1.0, 2.0])
    assert np.allclose(bands.x_right, [1.0, 2.0, 3.0])


@pytest.mark.gpu
def test_overlay_microcell_contraction_keeps_distinct_overlap_regions() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)])
    right = from_shapely_geometries([box(1.0, 0.0, 3.0, 2.0)])

    labels = build_and_label_overlay_microcells(left, right)
    components = contract_overlay_microcells(labels)

    assert components.component_count == 3
    assert components.component_ids.tolist() == [0, 1, 2]


@pytest.mark.gpu
def test_overlay_microcell_labels_match_simple_rectangle_overlap() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)])
    right = from_shapely_geometries([box(1.0, 0.0, 3.0, 2.0)])

    labels = build_and_label_overlay_microcells(left, right)

    assert labels.count == 3
    assert _to_host_bool(labels.left_inside).tolist() == [True, True, False]
    assert _to_host_bool(labels.right_inside).tolist() == [False, True, True]


@pytest.mark.gpu
def test_overlay_microcell_contraction_merges_same_label_bands_across_intervals() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0.0, 0.0, 4.0, 2.0)])
    right = from_shapely_geometries([
        Polygon(
            [
                (0.0, 0.0),
                (1.0, 0.0),
                (2.0, 0.0),
                (4.0, 0.0),
                (4.0, 2.0),
                (3.0, 2.0),
                (2.0, 2.0),
                (0.0, 2.0),
                (0.0, 0.0),
            ]
        )
    ])

    labels = build_and_label_overlay_microcells(left, right)
    components = contract_overlay_microcells(labels)

    assert labels.count >= 3
    assert all(_to_host_bool(labels.left_inside))
    assert all(_to_host_bool(labels.right_inside))
    assert components.component_count == 1


@pytest.mark.gpu
def test_overlay_microcell_reconstruction_coalesces_consecutive_selected_bands() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    labels = OverlayMicrocellLabels(
        bands=OverlayMicrocellBands(
            row_indices=np.asarray([0, 0, 0], dtype=np.int32),
            interval_indices=np.asarray([0, 1, 2], dtype=np.int32),
            lower_segment_ids=np.asarray([5, 5, 5], dtype=np.int32),
            upper_segment_ids=np.asarray([9, 9, 9], dtype=np.int32),
            x_left=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
            x_right=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            y_lower_left=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            y_lower_right=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            y_upper_left=np.asarray([2.0, 2.0, 2.0], dtype=np.float64),
            y_upper_right=np.asarray([2.0, 2.0, 2.0], dtype=np.float64),
            representative_x=np.asarray([0.5, 1.5, 2.5], dtype=np.float64),
            representative_y=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        ),
        left_inside=cp.asarray([True, True, True], dtype=cp.bool_),
        right_inside=cp.asarray([True, True, True], dtype=cp.bool_),
    )
    selected_ids = cp.asarray([0, 1, 2], dtype=cp.int64)
    coalesced = _coalesce_selected_microcells(labels, selected_ids)

    assert int(coalesced["row_indices"].size) == 1
    assert np.allclose(cp.asnumpy(coalesced["x_left"]), [0.0])
    assert np.allclose(cp.asnumpy(coalesced["x_right"]), [3.0])


@pytest.mark.gpu
def test_overlay_microcells_support_row_isolated_batches() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries(
        [box(0.0, 0.0, 2.0, 2.0), box(0.0, 0.0, 3.0, 3.0)]
    )
    right = from_shapely_geometries(
        [box(1.0, 0.0, 3.0, 2.0), box(1.0, 1.0, 2.0, 2.0)]
    )

    labels = build_and_label_overlay_microcells(left, right)

    assert labels.count > 3
    assert labels.bands.row_count == 2
    assert set(_to_host_array(labels.bands.row_indices).tolist()) == {0, 1}
    both_inside = _to_host_bool(labels.left_inside) & _to_host_bool(labels.right_inside)
    assert int(np.count_nonzero(both_inside)) >= 2


@pytest.mark.gpu
def test_overlay_microcells_can_emit_operation_selected_bands() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)])
    right = from_shapely_geometries([box(1.0, 0.0, 3.0, 2.0)])

    labels = build_and_label_overlay_microcells(
        left,
        right,
        selection_operation="intersection",
    )

    assert labels.count == 1
    assert _to_host_bool(labels.left_inside).tolist() == [True]
    assert _to_host_bool(labels.right_inside).tolist() == [True]
    assert np.allclose(_to_host_array(labels.bands.x_left), [1.0])
    assert np.allclose(_to_host_array(labels.bands.x_right), [2.0])


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("operation", "expected_count", "expected_left", "expected_right", "expected_x"),
    [
        ("intersection", 1, [True], [True], [(1.0, 2.0)]),
        ("union", 3, [True, True, False], [False, True, True], [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]),
        ("difference", 1, [True], [False], [(0.0, 1.0)]),
        ("symmetric_difference", 2, [True, False], [False, True], [(0.0, 1.0), (2.0, 3.0)]),
        ("identity", 2, [True, True], [False, True], [(0.0, 1.0), (1.0, 2.0)]),
    ],
)
def test_overlay_microcells_selected_build_matches_simple_rectangle_operation(
    operation,
    expected_count,
    expected_left,
    expected_right,
    expected_x,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)])
    right = from_shapely_geometries([box(1.0, 0.0, 3.0, 2.0)])

    labels = build_and_label_overlay_microcells(
        left,
        right,
        selection_operation=operation,
    )

    assert labels.count == expected_count
    assert _to_host_bool(labels.left_inside).tolist() == expected_left
    assert _to_host_bool(labels.right_inside).tolist() == expected_right
    got_x = list(
        zip(
            _to_host_array(labels.bands.x_left).tolist(),
            _to_host_array(labels.bands.x_right).tolist(),
            strict=False,
        )
    )
    assert got_x == expected_x


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("intersection", box(1.0, 0.0, 2.0, 2.0)),
        ("union", box(0.0, 0.0, 3.0, 2.0)),
        ("difference", box(0.0, 0.0, 1.0, 2.0)),
        ("identity", box(0.0, 0.0, 2.0, 2.0)),
    ],
)
def test_overlay_microcell_reconstruction_matches_simple_rectangles(operation, expected) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)])
    right = from_shapely_geometries([box(1.0, 0.0, 3.0, 2.0)])

    labels = build_and_label_overlay_microcells(left, right)
    components = contract_overlay_microcells(labels)
    result = reconstruct_overlay_from_microcells(
        labels,
        operation,
        components=components,
        row_count=1,
    )
    got = result.to_shapely()[0]

    assert shapely.equals(got, expected)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("intersection", box(1.0, 0.0, 2.0, 2.0)),
        ("union", box(0.0, 0.0, 3.0, 2.0)),
        ("difference", box(0.0, 0.0, 1.0, 2.0)),
        ("identity", box(0.0, 0.0, 2.0, 2.0)),
    ],
)
def test_overlay_microcell_selected_build_reconstruction_matches_simple_rectangles(
    operation,
    expected,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)])
    right = from_shapely_geometries([box(1.0, 0.0, 3.0, 2.0)])

    labels = build_and_label_overlay_microcells(
        left,
        right,
        selection_operation=operation,
    )
    components = contract_overlay_microcells(labels)
    result = reconstruct_overlay_from_microcells(
        labels,
        operation,
        components=components,
        row_count=1,
    )
    got = result.to_shapely()[0]

    assert shapely.equals(got, expected)


@pytest.mark.gpu
def test_overlay_microcell_reconstruction_symdiff_matches_disjoint_strips() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)])
    right = from_shapely_geometries([box(1.0, 0.0, 3.0, 2.0)])

    labels = build_and_label_overlay_microcells(left, right)
    components = contract_overlay_microcells(labels)
    result = reconstruct_overlay_from_microcells(
        labels,
        "symmetric_difference",
        components=components,
        row_count=1,
    )

    got = result.to_shapely()[0]
    expected = shapely.union_all([box(0.0, 0.0, 1.0, 2.0), box(2.0, 0.0, 3.0, 2.0)])
    assert shapely.equals(got, expected)


@pytest.mark.gpu
def test_overlay_microcell_selected_build_reconstruction_symdiff_matches_disjoint_strips() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)])
    right = from_shapely_geometries([box(1.0, 0.0, 3.0, 2.0)])

    labels = build_and_label_overlay_microcells(
        left,
        right,
        selection_operation="symmetric_difference",
    )
    components = contract_overlay_microcells(labels)
    result = reconstruct_overlay_from_microcells(
        labels,
        "symmetric_difference",
        components=components,
        row_count=1,
    )

    got = result.to_shapely()[0]
    expected = shapely.union_all([box(0.0, 0.0, 1.0, 2.0), box(2.0, 0.0, 3.0, 2.0)])
    assert shapely.equals(got, expected)
