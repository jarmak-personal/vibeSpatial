from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon

if TYPE_CHECKING:
    from vibespatial.geometry.owned import OwnedGeometryArray

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


def test_owned_to_shapely_does_not_route_through_wkb_bridge(monkeypatch) -> None:
    from vibespatial.io import wkb as wkb_module

    def _fail_encode(*args, **kwargs):
        raise AssertionError("owned->Shapely materialization should not use WKB bridge")

    monkeypatch.setattr(wkb_module, "encode_wkb_owned", _fail_encode)

    owned = from_shapely_geometries(_sample_geometries())
    restored = owned.to_shapely()

    assert restored[0].equals(Point(1, 2))
    assert restored[4].equals(Polygon([(0, 0), (3, 0), (3, 3), (0, 0)]))


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


def test_geoarrow_share_mode_reuses_cached_view_object() -> None:
    owned = from_shapely_geometries(_sample_geometries())

    first = owned.to_geoarrow(sharing=BufferSharingMode.SHARE)
    second = owned.to_geoarrow(sharing=BufferSharingMode.SHARE)

    assert first is second


def test_geoarrow_share_mode_reuses_cached_family_wrappers_on_import() -> None:
    owned = from_shapely_geometries(_sample_geometries())
    shared_view = owned.to_geoarrow(sharing=BufferSharingMode.SHARE)

    first = from_geoarrow(shared_view, sharing=BufferSharingMode.AUTO)
    second = from_geoarrow(shared_view, sharing=BufferSharingMode.AUTO)

    point_family = next(iter(first.families))
    assert first is not second
    assert first.families is not second.families
    assert first.families[point_family] is second.families[point_family]
    assert np.shares_memory(first.validity, second.validity)


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

    assert report["residency"] == Residency.DEVICE.value
    assert any(event["kind"] == DiagnosticKind.TRANSFER.value for event in report["events"])
    assert any(event["kind"] == DiagnosticKind.MATERIALIZATION.value for event in report["events"])
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


# ---------------------------------------------------------------------------
# Device-resident concat (lyy.29)
# ---------------------------------------------------------------------------


def _make_device_resident(geoms: list[object | None]) -> OwnedGeometryArray:
    """Create a device-resident OwnedGeometryArray with host stubs cleared."""
    from vibespatial.geometry.owned import FamilyGeometryBuffer

    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    # Clear host family buffers to simulate true device-only arrays
    owned.families = {
        family: FamilyGeometryBuffer(
            family=buffer.family,
            schema=buffer.schema,
            row_count=buffer.row_count,
            x=np.empty(0, dtype=np.float64),
            y=np.empty(0, dtype=np.float64),
            geometry_offsets=np.empty(0, dtype=np.int32),
            empty_mask=np.empty(0, dtype=np.bool_),
            host_materialized=False,
        )
        for family, buffer in owned.families.items()
    }
    return owned


@pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
class TestDeviceResidentConcat:
    """Verify that OwnedGeometryArray.concat() stays device-resident when all
    inputs are device-resident, with no D->H transfer."""

    def test_concat_points_stays_device_resident(self) -> None:
        """Concatenating device-resident point arrays produces a device-resident result."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        owned1 = _make_device_resident([Point(0, 0), Point(1, 1)])
        owned2 = _make_device_resident([Point(2, 2), Point(3, 3)])

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.device_state is not None
        assert result.row_count == 4
        # Verify host metadata is not materialized (stays None)
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None

        # Verify round-trip correctness after host materialization
        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(Point(0, 0))
        assert shapely_result[1].equals(Point(1, 1))
        assert shapely_result[2].equals(Point(2, 2))
        assert shapely_result[3].equals(Point(3, 3))

    def test_concat_polygons_stays_device_resident(self) -> None:
        """Concatenating device-resident polygon arrays preserves ring offsets."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        p2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])

        owned1 = _make_device_resident([p1])
        owned2 = _make_device_resident([p2])

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 2

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(p1)
        assert shapely_result[1].equals(p2)

    def test_concat_multipolygons_stays_device_resident(self) -> None:
        """Concatenating device-resident multipolygon arrays preserves 3-level offsets."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        mp1 = MultiPolygon([
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 2)]),
        ])
        mp2 = MultiPolygon([
            Polygon([(10, 10), (11, 10), (11, 11), (10, 10)]),
        ])

        owned1 = _make_device_resident([mp1])
        owned2 = _make_device_resident([mp2])

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 2

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(mp1)
        assert shapely_result[1].equals(mp2)

    def test_concat_linestrings_stays_device_resident(self) -> None:
        """Concatenating device-resident linestring arrays."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        ls1 = LineString([(0, 0), (1, 1), (2, 0)])
        ls2 = LineString([(5, 5), (6, 6)])

        owned1 = _make_device_resident([ls1])
        owned2 = _make_device_resident([ls2])

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 2

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(ls1)
        assert shapely_result[1].equals(ls2)

    def test_concat_multilinestrings_stays_device_resident(self) -> None:
        """Concatenating device-resident multilinestring arrays preserves part offsets."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        mls1 = MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
        mls2 = MultiLineString([[(10, 10), (11, 11)]])

        owned1 = _make_device_resident([mls1])
        owned2 = _make_device_resident([mls2])

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 2

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(mls1)
        assert shapely_result[1].equals(mls2)

    def test_concat_with_nulls_stays_device_resident(self) -> None:
        """Null rows are correctly handled in device-resident concat."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        owned1 = _make_device_resident([Point(0, 0), None])
        owned2 = _make_device_resident([None, Point(1, 1)])

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 4

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(Point(0, 0))
        assert shapely_result[1] is None
        assert shapely_result[2] is None
        assert shapely_result[3].equals(Point(1, 1))

    def test_concat_mixed_families_stays_device_resident(self) -> None:
        """Concatenating arrays with different geometry families."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        owned1 = _make_device_resident([
            Point(0, 0),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
        ])
        owned2 = _make_device_resident([
            LineString([(2, 2), (3, 3)]),
            Point(4, 4),
        ])

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 4

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(Point(0, 0))
        assert shapely_result[1].equals(Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]))
        assert shapely_result[2].equals(LineString([(2, 2), (3, 3)]))
        assert shapely_result[3].equals(Point(4, 4))

    def test_concat_no_d2h_transfer(self, monkeypatch) -> None:
        """Device-resident concat must not call _ensure_host_state."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        owned1 = _make_device_resident([Point(0, 0), Point(1, 1)])
        owned2 = _make_device_resident([Point(2, 2)])

        calls = []

        def _spy_host_state(self_inner):
            calls.append("_ensure_host_state")

        monkeypatch.setattr(
            OwnedGeometryArray, "_ensure_host_state", _spy_host_state,
        )

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert len(calls) == 0, (
            "_ensure_host_state was called during device-resident concat"
        )

    def test_concat_host_fallback_when_mixed_residency(self) -> None:
        """When some inputs are host-resident, falls back to host concat."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        device_owned = _make_device_resident([Point(0, 0)])
        host_owned = from_shapely_geometries([Point(1, 1)])

        result = OwnedGeometryArray.concat([device_owned, host_owned])

        # Should fall through to host path and produce correct result
        assert result.residency is Residency.HOST
        assert result.row_count == 2

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(Point(0, 0))
        assert shapely_result[1].equals(Point(1, 1))

    def test_concat_three_arrays_stays_device_resident(self) -> None:
        """Concatenating 3+ device-resident arrays."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        owned1 = _make_device_resident([Point(0, 0)])
        owned2 = _make_device_resident([Point(1, 1)])
        owned3 = _make_device_resident([Point(2, 2)])

        result = OwnedGeometryArray.concat([owned1, owned2, owned3])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 3

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(Point(0, 0))
        assert shapely_result[1].equals(Point(1, 1))
        assert shapely_result[2].equals(Point(2, 2))

    def test_concat_polygon_with_hole_stays_device_resident(self) -> None:
        """Polygons with holes preserve interior rings through device concat."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        poly_hole = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            [[(2, 2), (4, 2), (4, 4), (2, 2)]],
        )
        poly_simple = Polygon([(20, 20), (21, 20), (21, 21), (20, 20)])

        owned1 = _make_device_resident([poly_hole])
        owned2 = _make_device_resident([poly_simple])

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(poly_hole)
        assert len(list(shapely_result[0].interiors)) == 1
        assert shapely_result[1].equals(poly_simple)

    def test_concat_all_families_device_resident(self) -> None:
        """Concatenating arrays covering all 6 geometry families."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        geoms1 = [
            Point(1, 2),
            LineString([(0, 0), (1, 1)]),
            Polygon([(0, 0), (3, 0), (3, 3), (0, 0)]),
        ]
        geoms2 = [
            MultiPoint([(0, 0), (1, 1)]),
            MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
            MultiPolygon([
                Polygon([(10, 10), (12, 10), (12, 12), (10, 10)]),
            ]),
        ]

        owned1 = _make_device_resident(geoms1)
        owned2 = _make_device_resident(geoms2)

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 6

        shapely_result = result.to_shapely()
        for i, expected in enumerate(geoms1 + geoms2):
            assert shapely_result[i].equals(expected), (
                f"Mismatch at index {i}: {expected.geom_type}"
            )

    def test_concat_disjoint_families_device_resident(self) -> None:
        """One array has only polygons, the other has only points."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        owned1 = _make_device_resident([
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
        ])
        owned2 = _make_device_resident([
            Point(5, 5),
        ])

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 2

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]))
        assert shapely_result[1].equals(Point(5, 5))


class TestEnsureDeviceStateSafetyCheck:
    """Verify that _ensure_device_state refuses to upload unmaterialised stubs.

    When an OwnedGeometryArray has host_materialized=False stubs (empty
    x/y arrays) AND no device_state, uploading those stubs creates
    zero-length device buffers while metadata references rows that should
    contain coordinates.  Kernels then read garbage from uninitialized GPU
    memory, producing denormalized-double coordinates (e.g. 8e-309) and
    downstream TopologyException crashes.
    """

    @pytest.mark.skipif(not has_gpu_runtime(), reason="GPU not available")
    def test_ensure_device_state_rejects_unmaterialised_stubs(self) -> None:
        """_ensure_device_state raises RuntimeError for empty stubs without device_state."""
        from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
        from vibespatial.geometry.owned import FamilyGeometryBuffer, OwnedGeometryArray

        # Build an OGA that simulates the bug pattern:
        # - residency=HOST (or DEVICE)
        # - host families have host_materialized=False stubs
        # - device_state is None (lost during incorrect construction)
        polygon_stub = FamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
            row_count=1,
            x=np.empty(0, dtype=np.float64),
            y=np.empty(0, dtype=np.float64),
            geometry_offsets=np.empty(0, dtype=np.int32),
            empty_mask=np.empty(0, dtype=np.bool_),
            host_materialized=False,
        )
        from vibespatial.geometry.owned import FAMILY_TAGS

        oga = OwnedGeometryArray(
            validity=np.array([True], dtype=bool),
            tags=np.array([FAMILY_TAGS[GeometryFamily.POLYGON]], dtype=np.int8),
            family_row_offsets=np.array([0], dtype=np.int32),
            families={GeometryFamily.POLYGON: polygon_stub},
            residency=Residency.HOST,
            device_state=None,
        )

        with pytest.raises(RuntimeError, match="unmaterialised stubs"):
            oga._ensure_device_state()

    @pytest.mark.skipif(not has_gpu_runtime(), reason="GPU not available")
    def test_ensure_device_state_succeeds_for_materialised_host(self) -> None:
        """_ensure_device_state succeeds when host families are properly materialised."""
        from vibespatial.geometry.buffers import GeometryFamily as GF

        owned = from_shapely_geometries(
            [Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])],
            residency=Residency.HOST,
        )
        # This should succeed -- host families have real data
        d_state = owned._ensure_device_state()
        assert d_state is not None
        assert GF.POLYGON in d_state.families

    @pytest.mark.skipif(not has_gpu_runtime(), reason="GPU not available")
    def test_ensure_device_state_shortcircuits_when_device_state_exists(self) -> None:
        """_ensure_device_state returns existing device_state without re-uploading."""
        owned = _make_device_resident([
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
        ])
        assert owned.device_state is not None

        # Families are unmaterialised stubs, but device_state exists
        # so _ensure_device_state should short-circuit and not hit the check
        d_state = owned._ensure_device_state()
        assert d_state is owned.device_state
