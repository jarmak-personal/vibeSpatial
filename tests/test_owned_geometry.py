from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon

from vibespatial import (
    BufferSharingMode,
    DiagnosticKind,
    ExecutionMode,
    Residency,
    RuntimeSelection,
    TransferTrigger,
    compute_geometry_bounds,
    compute_morton_keys,
    compute_offset_spans,
    compute_total_bounds,
    from_geoarrow,
    from_shapely_geometries,
    from_wkb,
    has_gpu_runtime,
)


def _sample_geometries() -> list[object | None]:
    return [
        Point(1, 2),
        None,
        Point(),
        LineString([(0, 0), (2, 4)]),
        Polygon([(0, 0), (3, 0), (3, 3), (0, 0)]),
        MultiPolygon(
            [
                Polygon([(10, 10), (12, 10), (12, 12), (10, 10)]),
                Polygon([(20, 20), (21, 20), (21, 21), (20, 20)]),
            ]
        ),
    ]


def test_shapely_round_trip_preserves_null_and_empty() -> None:
    owned = from_shapely_geometries(_sample_geometries())
    restored = owned.to_shapely()

    assert restored[0].equals(Point(1, 2))
    assert restored[1] is None
    assert restored[2].is_empty
    assert restored[3].equals(LineString([(0, 0), (2, 4)]))


def test_wkb_round_trip_matches_shapely_path() -> None:
    baseline = from_shapely_geometries(_sample_geometries())
    wkb = baseline.to_wkb()
    restored = from_wkb(wkb)

    restored_shapes = restored.to_shapely()
    baseline_shapes = baseline.to_shapely()
    for left, right in zip(restored_shapes, baseline_shapes, strict=True):
        if left is None or right is None:
            assert left is right
            continue
        assert left.equals(right)


def test_geoarrow_style_round_trip_preserves_family_buffers() -> None:
    baseline = from_shapely_geometries(_sample_geometries())
    restored = from_geoarrow(baseline.to_geoarrow())

    assert restored.row_count == baseline.row_count
    assert np.array_equal(restored.validity, baseline.validity)
    assert np.array_equal(restored.tags, baseline.tags)
    assert np.array_equal(restored.family_row_offsets, baseline.family_row_offsets)


def test_geoarrow_share_mode_reuses_buffers_and_delays_host_geometry_materialization() -> None:
    owned = from_shapely_geometries(_sample_geometries())
    shared_view = owned.to_geoarrow(sharing=BufferSharingMode.SHARE)
    adopted = from_geoarrow(shared_view, sharing=BufferSharingMode.AUTO)

    point_family = next(iter(adopted.families))
    assert np.shares_memory(shared_view.validity, adopted.validity)
    assert np.shares_memory(shared_view.families[point_family].x, adopted.families[point_family].x)
    assert adopted.geoarrow_backed is True
    assert adopted.shares_geoarrow_memory is True
    assert not any(event.kind is DiagnosticKind.MATERIALIZATION for event in adopted.diagnostics)

    adopted.to_shapely()

    assert any(event.kind is DiagnosticKind.MATERIALIZATION for event in adopted.diagnostics)


def test_bounds_and_total_bounds_ignore_nulls_and_empty() -> None:
    owned = from_shapely_geometries(_sample_geometries())

    bounds = compute_geometry_bounds(owned)
    total = compute_total_bounds(owned)

    assert np.allclose(bounds[0], np.asarray([1.0, 2.0, 1.0, 2.0]))
    assert np.isnan(bounds[1]).all()
    assert np.isnan(bounds[2]).all()
    assert total == (0.0, 0.0, 21.0, 21.0)


def test_offset_spans_expose_payload_hierarchy() -> None:
    owned = from_shapely_geometries(_sample_geometries())

    geometry_spans = compute_offset_spans(owned, level="geometry")
    part_spans = compute_offset_spans(owned, level="part")

    assert geometry_spans
    assert any(span.size > 0 for span in geometry_spans.values())
    assert any(span.size > 0 for span in part_spans.values())


def test_morton_keys_place_null_and_empty_at_end() -> None:
    owned = from_shapely_geometries(_sample_geometries())

    keys = compute_morton_keys(owned)

    assert keys.shape == (owned.row_count,)
    assert keys[1] == np.iinfo(np.uint64).max
    assert keys[2] == np.iinfo(np.uint64).max


@pytest.mark.gpu
def test_diagnostics_capture_residency_and_runtime_changes() -> None:
    owned = from_shapely_geometries(_sample_geometries())
    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="explicit gpu execution requested",
    )
    owned.record_runtime_selection(
        RuntimeSelection(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            reason="GPU runtime unavailable; using explicit CPU fallback",
        )
    )
    owned.to_shapely()
    report = owned.diagnostics_report()

    assert report["residency"] == Residency.HOST.value
    assert any(event["kind"] == DiagnosticKind.TRANSFER.value for event in report["events"])
    assert any("CPU fallback" in reason for reason in report["runtime_history"])


def test_move_to_device_allocates_device_mirrors_when_gpu_is_available() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = from_shapely_geometries(_sample_geometries())
    assert owned.device_state is None

    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="explicit gpu execution requested",
    )

    report = owned.diagnostics_report()
    assert owned.device_state is not None
    assert report["device_buffers_allocated"] is True
    assert report["residency"] == Residency.DEVICE.value


def test_take_by_indices_preserves_geometries() -> None:
    owned = from_shapely_geometries(_sample_geometries())
    full_shapely = owned.to_shapely()

    subset = owned.take(np.array([0, 3, 4, 5]))
    subset_shapely = subset.to_shapely()

    assert subset.row_count == 4
    assert subset_shapely[0].equals(full_shapely[0])  # Point
    assert subset_shapely[1].equals(full_shapely[3])  # LineString
    assert subset_shapely[2].equals(full_shapely[4])  # Polygon
    assert subset_shapely[3].equals(full_shapely[5])  # MultiPolygon


def test_take_by_boolean_mask_preserves_geometries() -> None:
    owned = from_shapely_geometries(_sample_geometries())
    full_shapely = owned.to_shapely()

    mask = np.array([True, False, False, False, True, True])
    subset = owned.take(mask)
    subset_shapely = subset.to_shapely()

    assert subset.row_count == 3
    assert subset_shapely[0].equals(full_shapely[0])  # Point
    assert subset_shapely[1].equals(full_shapely[4])  # Polygon
    assert subset_shapely[2].equals(full_shapely[5])  # MultiPolygon


def test_take_preserves_null_and_empty() -> None:
    owned = from_shapely_geometries(_sample_geometries())

    subset = owned.take(np.array([1, 2]))
    subset_shapely = subset.to_shapely()

    assert subset.row_count == 2
    assert subset_shapely[0] is None       # null
    assert subset_shapely[1].is_empty      # empty Point


def test_take_single_row() -> None:
    owned = from_shapely_geometries(_sample_geometries())
    full_shapely = owned.to_shapely()

    for i in range(len(full_shapely)):
        subset = owned.take(np.array([i]))
        result = subset.to_shapely()
        assert subset.row_count == 1
        if full_shapely[i] is None:
            assert result[0] is None
        else:
            assert result[0].equals(full_shapely[i])


def test_take_polygon_with_holes() -> None:
    poly_with_hole = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        [[(2, 2), (4, 2), (4, 4), (2, 2)]],
    )
    geoms = [Point(1, 1), poly_with_hole, Point(5, 5)]
    owned = from_shapely_geometries(geoms)
    subset = owned.take(np.array([1]))
    result = subset.to_shapely()

    assert subset.row_count == 1
    assert result[0].equals(poly_with_hole)
    assert len(list(result[0].interiors)) == 1


def test_take_all_families() -> None:
    geoms = [
        Point(1, 2),
        LineString([(0, 0), (1, 1), (2, 0)]),
        Polygon([(0, 0), (3, 0), (3, 3), (0, 0)]),
        MultiPoint([(0, 0), (1, 1)]),
        MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
        MultiPolygon([
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 2)]),
        ]),
    ]
    owned = from_shapely_geometries(geoms)
    for i, geom in enumerate(geoms):
        subset = owned.take(np.array([i]))
        result = subset.to_shapely()
        assert result[0].equals(geom), f"family {geom.geom_type} at index {i} failed round-trip"


def test_take_empty_indices() -> None:
    owned = from_shapely_geometries(_sample_geometries())

    subset = owned.take(np.array([], dtype=np.int64))
    assert subset.row_count == 0
    assert subset.to_shapely() == []


@pytest.mark.gpu
def test_compute_geometry_bounds_gpu_matches_cpu_reference() -> None:
    owned = from_shapely_geometries(_sample_geometries())

    cpu_bounds = compute_geometry_bounds(owned, dispatch_mode=ExecutionMode.CPU)
    gpu_bounds = compute_geometry_bounds(owned, dispatch_mode=ExecutionMode.GPU)

    assert np.allclose(cpu_bounds, gpu_bounds, equal_nan=True)
