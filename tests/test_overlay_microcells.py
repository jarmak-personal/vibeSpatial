from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box

from vibespatial import from_shapely_geometries, has_gpu_runtime
from vibespatial.overlay.contraction_reconstruct import reconstruct_overlay_from_microcells
from vibespatial.overlay.microcells import (
    OverlayMicrocellBands,
    OverlayMicrocellLabels,
    build_and_label_overlay_microcells,
    build_overlay_microcell_bands,
)

microcells_module = importlib.import_module("vibespatial.overlay.microcells")

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


def test_overlay_microcell_labeling_is_one_row_indirected_device_pass() -> None:
    source = inspect.getsource(microcells_module.label_overlay_microcells)

    assert source.count("point_in_polygon(") == 2
    assert source.count("._device_indexed_take(") == 2
    assert "overlay_device_to_host" not in source
    assert "cp.unique(" not in source
    assert "for row_id" not in source
    assert "_broadcast_right_owned" not in source


def _selected_rectangular_microcell_labels(
    bands: list[tuple[int, float, float, float, float]],
) -> OverlayMicrocellLabels:
    interval_indices = cp.asarray([band[0] for band in bands], dtype=cp.int32)
    x_left = cp.asarray([band[1] for band in bands], dtype=cp.float64)
    x_right = cp.asarray([band[2] for band in bands], dtype=cp.float64)
    y_lower = cp.asarray([band[3] for band in bands], dtype=cp.float64)
    y_upper = cp.asarray([band[4] for band in bands], dtype=cp.float64)
    count = len(bands)
    return OverlayMicrocellLabels(
        bands=OverlayMicrocellBands(
            row_indices=cp.zeros(count, dtype=cp.int32),
            interval_indices=interval_indices,
            lower_segment_ids=cp.arange(count, dtype=cp.int32),
            upper_segment_ids=cp.arange(count, dtype=cp.int32) + count,
            x_left=x_left,
            x_right=x_right,
            y_lower_left=y_lower,
            y_lower_right=y_lower,
            y_upper_left=y_upper,
            y_upper_right=y_upper,
            representative_x=0.5 * (x_left + x_right),
            representative_y=0.5 * (y_lower + y_upper),
        ),
        left_inside=cp.ones(count, dtype=cp.bool_),
        right_inside=cp.zeros(count, dtype=cp.bool_),
    )


def test_selected_microcell_builder_uses_bounded_interval_pages() -> None:
    source = inspect.getsource(
        microcells_module._build_selected_row_microcell_arrays_device
    )

    assert "memberships_per_page" in source
    assert "_stable_radix_order_pass" in source
    assert "cp.lexsort(cp.stack((y_mid, interval_ids)))" not in source
    assert "overlay microcells selected-row membership-count allocation fence" not in source
    assert "selected-row valid-band admission fence" not in source
    assert "selected-row adjacent-band admission fence" not in source
    assert "selected-row kept-band admission fence" not in source


def test_selected_microcell_ingress_is_device_only_for_all_row_shapes() -> None:
    source = inspect.getsource(
        microcells_module._build_and_label_selected_overlay_microcells_device
    )

    assert "_same_row_span_summary=(left_device.count, right_device.count, 0)" in source
    assert "_build_selected_segmented_microcell_labels_device" in source
    assert "overlay_device_to_host" not in source
    assert "_segment_row_spans" not in source
    assert "for row" not in source


def test_segmented_microcell_carrier_uses_device_row_indirection() -> None:
    source = inspect.getsource(
        microcells_module._build_selected_segmented_microcell_labels_device
    )

    assert "segments_per_row = cp.bincount" in source
    assert "membership_capacity = interval_count * max_row_segments" in source
    assert "membership_interval_ids = cp.searchsorted" in source
    assert "cp.repeat(" not in source
    assert "segment_row_offsets" in source
    assert "sorted_interval_groups" in source
    assert "overlay_device_to_host" not in source
    assert "for row" not in source


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
def test_overlay_microcell_boundary_graph_merges_bands_across_intervals() -> None:
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
    result = reconstruct_overlay_from_microcells(labels, "identity", row_count=1)

    assert labels.count >= 3
    assert all(_to_host_bool(labels.left_inside))
    assert all(_to_host_bool(labels.right_inside))
    assert shapely.equals(result.to_shapely()[0], box(0.0, 0.0, 4.0, 2.0))


@pytest.mark.gpu
def test_overlay_microcell_boundary_graph_atomizes_partial_vertical_hole_seams() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    labels = _selected_rectangular_microcell_labels(
        [
            (0, 0.0, 1.0, 0.0, 4.0),
            (1, 1.0, 3.0, 0.0, 1.0),
            (1, 1.0, 3.0, 3.0, 4.0),
            (2, 3.0, 4.0, 0.0, 4.0),
        ]
    )
    result = reconstruct_overlay_from_microcells(labels, "identity", row_count=1)
    expected = Polygon(
        box(0.0, 0.0, 4.0, 4.0).exterior.coords,
        [box(1.0, 1.0, 3.0, 3.0).exterior.coords],
    )

    assert shapely.equals(result.to_shapely()[0], expected)


@pytest.mark.gpu
def test_overlay_microcell_boundary_graph_preserves_nested_island() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    labels = _selected_rectangular_microcell_labels(
        [
            (0, 0.0, 1.0, 0.0, 6.0),
            (1, 1.0, 2.0, 0.0, 1.0),
            (1, 1.0, 2.0, 5.0, 6.0),
            (2, 2.0, 4.0, 0.0, 1.0),
            (2, 2.0, 4.0, 2.0, 4.0),
            (2, 2.0, 4.0, 5.0, 6.0),
            (3, 4.0, 5.0, 0.0, 1.0),
            (3, 4.0, 5.0, 5.0, 6.0),
            (4, 5.0, 6.0, 0.0, 6.0),
        ]
    )
    result = reconstruct_overlay_from_microcells(labels, "identity", row_count=1)
    outer = Polygon(
        box(0.0, 0.0, 6.0, 6.0).exterior.coords,
        [box(1.0, 1.0, 5.0, 5.0).exterior.coords],
    )
    expected = shapely.union_all([outer, box(2.0, 2.0, 4.0, 4.0)])

    assert shapely.equals(result.to_shapely()[0], expected)


@pytest.mark.gpu
def test_overlay_microcell_boundary_graph_preserves_exact_nonzero_sliver() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    labels = _selected_rectangular_microcell_labels(
        [(0, 0.0, 1.0, 0.0, 1.0e-14)]
    )
    result = reconstruct_overlay_from_microcells(labels, "identity", row_count=1)

    assert shapely.equals(result.to_shapely()[0], box(0.0, 0.0, 1.0, 1.0e-14))


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
def test_overlay_microcells_gpu_generic_path_bypasses_host_row_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    monkeypatch.setattr(
        microcells_module,
        "_row_microcell_bands",
        lambda *args, **kwargs: pytest.fail("GPU generic microcell labeling should not use host row bands"),
    )

    left = from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)])
    right = from_shapely_geometries([box(1.0, 0.0, 3.0, 2.0)])

    labels = build_and_label_overlay_microcells(left, right)

    assert labels.count == 3
    assert _to_host_bool(labels.left_inside).tolist() == [True, True, False]
    assert _to_host_bool(labels.right_inside).tolist() == [False, True, True]


@pytest.mark.gpu
def test_overlay_microcells_selected_path_reuses_segment_span_metadata() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    left = from_shapely_geometries(
        [box(0.0, 0.0, 2.0, 2.0), box(0.0, 0.0, 3.0, 3.0)]
    )
    right = from_shapely_geometries(
        [box(1.0, 0.0, 3.0, 2.0), box(1.0, 1.0, 2.0, 2.0)]
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    labels = build_and_label_overlay_microcells(
        left,
        right,
        selection_operation="intersection",
    )
    events = get_d2h_transfer_events(clear=True)

    assert labels.count >= 2
    reasons = {event.reason for event in events}
    assert "segment same-row span summary scalar fence" not in reasons
    assert not any("overlay microcells left row-span" in reason for reason in reasons)
    assert not any("overlay microcells right row-span" in reason for reason in reasons)


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
    result = reconstruct_overlay_from_microcells(
        labels,
        operation,
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
    result = reconstruct_overlay_from_microcells(
        labels,
        operation,
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
    result = reconstruct_overlay_from_microcells(
        labels,
        "symmetric_difference",
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
    result = reconstruct_overlay_from_microcells(
        labels,
        "symmetric_difference",
        row_count=1,
    )

    got = result.to_shapely()[0]
    expected = shapely.union_all([box(0.0, 0.0, 1.0, 2.0), box(2.0, 0.0, 3.0, 2.0)])
    assert shapely.equals(got, expected)
