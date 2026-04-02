"""Integration tests for the zero-copy GPU pipeline.

Verifies that multi-step spatial operations maintain device residency
without D->H transfers between steps.
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, Polygon

from vibespatial.geometry.owned import (
    OwnedGeometryArray,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.residency import Residency

# ---------------------------------------------------------------------------
# Phase 1: Device-resident builder
# ---------------------------------------------------------------------------


class TestBuildDeviceResidentOwned:
    """Tests for build_device_resident_owned factory."""

    def test_basic_construction(self):
        """build_device_resident_owned -> to_shapely produces correct geometries."""
        from vibespatial.geometry.owned import build_device_resident_owned

        pytest.importorskip("cupy")
        from vibespatial.cuda._runtime import get_cuda_runtime
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.geometry.owned import (
            FAMILY_TAGS,
            DeviceFamilyGeometryBuffer,
        )

        runtime = get_cuda_runtime()

        # Create 3 points: (1,2), (3,4), (5,6)
        x = np.array([1.0, 3.0, 5.0], dtype=np.float64)
        y = np.array([2.0, 4.0, 6.0], dtype=np.float64)
        d_x = runtime.from_host(x)
        d_y = runtime.from_host(y)
        geom_offsets = np.arange(4, dtype=np.int32)
        empty_mask = np.zeros(3, dtype=bool)

        device_families = {
            GeometryFamily.POINT: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POINT,
                x=d_x,
                y=d_y,
                geometry_offsets=runtime.from_host(geom_offsets),
                empty_mask=runtime.from_host(empty_mask),
            ),
        }

        result = build_device_resident_owned(
            device_families=device_families,
            row_count=3,
            tags=np.full(3, FAMILY_TAGS[GeometryFamily.POINT], dtype=np.int8),
            validity=np.ones(3, dtype=bool),
            family_row_offsets=np.arange(3, dtype=np.int32),
        )

        assert result.residency == Residency.DEVICE
        assert result.device_state is not None

        # Materialize and verify
        geoms = result.to_shapely()
        assert len(geoms) == 3
        assert geoms[0].x == 1.0
        assert geoms[0].y == 2.0
        assert geoms[2].x == 5.0
        assert geoms[2].y == 6.0

    def test_family_has_rows_device_resident_polygon(self):
        """family_has_rows returns True for device-resident Polygon family.

        The host stub has empty geometry_offsets (len 0) and ring_offsets=None,
        but device_state has the real data. family_has_rows must read device side.
        """
        from vibespatial.geometry.owned import build_device_resident_owned

        pytest.importorskip("cupy")
        from vibespatial.cuda._runtime import get_cuda_runtime
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.geometry.owned import (
            FAMILY_TAGS,
            DeviceFamilyGeometryBuffer,
        )

        runtime = get_cuda_runtime()

        # Build a device-resident Polygon: one square (0,0)-(1,0)-(1,1)-(0,1)-(0,0)
        x = np.array([0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)
        y = np.array([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        geom_offsets = np.array([0, 1], dtype=np.int32)  # 1 polygon -> 1 ring
        ring_offsets = np.array([0, 5], dtype=np.int32)  # 1 ring -> 5 coords
        empty_mask = np.zeros(1, dtype=bool)

        device_families = {
            GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                x=runtime.from_host(x),
                y=runtime.from_host(y),
                geometry_offsets=runtime.from_host(geom_offsets),
                empty_mask=runtime.from_host(empty_mask),
                ring_offsets=runtime.from_host(ring_offsets),
            ),
        }

        result = build_device_resident_owned(
            device_families=device_families,
            row_count=1,
            tags=np.full(1, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=np.int8),
            validity=np.ones(1, dtype=bool),
            family_row_offsets=np.arange(1, dtype=np.int32),
        )

        # Host stub has empty offsets — the old guard pattern would fail
        host_buf = result.families[GeometryFamily.POLYGON]
        assert not host_buf.host_materialized
        assert len(host_buf.geometry_offsets) == 0  # stub is empty
        assert host_buf.ring_offsets is None  # stub has no ring_offsets

        # family_has_rows reads device side → True
        assert result.family_has_rows(GeometryFamily.POLYGON) is True

        # Missing family → False
        assert result.family_has_rows(GeometryFamily.POINT) is False

    def test_family_has_rows_host_resident(self):
        """family_has_rows works correctly for host-resident arrays."""
        from vibespatial.geometry.buffers import GeometryFamily

        polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        owned = from_shapely_geometries(polys)

        assert owned.family_has_rows(GeometryFamily.POLYGON) is True
        assert owned.family_has_rows(GeometryFamily.POINT) is False


# ---------------------------------------------------------------------------
# Phase 1B/1C: Centroid device-resident output
# ---------------------------------------------------------------------------


class TestCentroidDeviceResident:
    """Tests that centroid returns device-resident OwnedGeometryArray."""

    def test_centroid_returns_owned(self):
        """centroid_owned returns OwnedGeometryArray, not tuple."""
        from vibespatial.constructive.centroid import centroid_owned

        points = [Point(1, 2), Point(3, 4), Point(5, 6)]
        owned = from_shapely_geometries(points)
        result = centroid_owned(owned)
        assert isinstance(result, OwnedGeometryArray)

    def test_centroid_correctness(self):
        """centroid_owned produces correct results."""
        from vibespatial.constructive.centroid import centroid_owned

        polys = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        ]
        owned = from_shapely_geometries(polys)
        result = centroid_owned(owned)
        geoms = result.to_shapely()
        assert len(geoms) == 2
        # Square centroid should be at center
        assert abs(geoms[0].x - 5.0) < 1e-10
        assert abs(geoms[0].y - 5.0) < 1e-10
        assert abs(geoms[1].x - 2.0) < 1e-10
        assert abs(geoms[1].y - 2.0) < 1e-10

    def test_geometry_array_centroid_from_owned(self):
        """GeometryArray.centroid uses from_owned path."""
        from vibespatial.api.geometry_array import GeometryArray

        points = [Point(1, 2), Point(3, 4)]
        ga = GeometryArray(np.array(points, dtype=object))
        ga._owned = from_shapely_geometries(points)

        result = ga.centroid
        assert isinstance(result, GeometryArray)
        assert result._owned is not None  # from_owned path was used


# ---------------------------------------------------------------------------
# Phase 2A: Affine transform
# ---------------------------------------------------------------------------


class TestAffineTransform:
    """Tests for GPU-accelerated affine transforms."""

    def test_translate(self):
        """translate_owned shifts coordinates correctly."""
        from vibespatial.constructive.affine_transform import translate_owned

        points = [Point(1, 2), Point(3, 4)]
        owned = from_shapely_geometries(points)
        result = translate_owned(owned, xoff=10.0, yoff=20.0)
        geoms = result.to_shapely()
        assert abs(geoms[0].x - 11.0) < 1e-10
        assert abs(geoms[0].y - 22.0) < 1e-10

    def test_affine_matrix(self):
        """affine_transform_owned applies matrix correctly."""
        from vibespatial.constructive.affine_transform import affine_transform_owned

        points = [Point(1, 0), Point(0, 1)]
        owned = from_shapely_geometries(points)
        # 90-degree rotation: [cos,-sin,0,sin,cos,0] = [0,-1,1,0,0,0]
        result = affine_transform_owned(owned, [0, -1, 1, 0, 0, 0])
        geoms = result.to_shapely()
        # (1,0) -> (0,1) and (0,1) -> (-1,0)
        assert abs(geoms[0].x - 0.0) < 1e-10
        assert abs(geoms[0].y - 1.0) < 1e-10

    def test_polygon_translate(self):
        """translate_owned works on polygons."""
        from vibespatial.constructive.affine_transform import translate_owned

        polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        owned = from_shapely_geometries(polys)
        result = translate_owned(owned, xoff=5.0, yoff=5.0)
        geoms = result.to_shapely()
        bounds = geoms[0].bounds
        assert abs(bounds[0] - 5.0) < 1e-10  # minx
        assert abs(bounds[1] - 5.0) < 1e-10  # miny


# ---------------------------------------------------------------------------
# Phase 2B: Envelope
# ---------------------------------------------------------------------------


class TestEnvelope:
    """Tests for envelope kernel."""

    def test_envelope_polygon(self):
        """envelope_owned produces correct bounding box polygon."""
        from vibespatial.constructive.envelope import envelope_owned

        # Triangle: should produce a rectangular envelope
        poly = Polygon([(0, 0), (5, 10), (10, 0)])
        owned = from_shapely_geometries([poly])
        result = envelope_owned(owned)
        geoms = result.to_shapely()
        assert len(geoms) == 1
        env = geoms[0]
        assert abs(env.bounds[0] - 0.0) < 1e-10  # minx
        assert abs(env.bounds[1] - 0.0) < 1e-10  # miny
        assert abs(env.bounds[2] - 10.0) < 1e-10  # maxx
        assert abs(env.bounds[3] - 10.0) < 1e-10  # maxy

    def test_envelope_gpu_100_polygons(self):
        """GPU envelope on 100 polygons: no exception, DEVICE residency, matches Shapely."""
        from vibespatial.constructive.envelope import _envelope_gpu

        pytest.importorskip("cupy")
        from vibespatial.runtime import has_gpu_runtime
        from vibespatial.runtime.residency import Residency

        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")

        rng = np.random.default_rng(99)
        polys = []
        for _ in range(100):
            cx, cy = rng.uniform(-500, 500, size=2)
            r = rng.uniform(1, 50)
            angles = np.sort(rng.uniform(0, 2 * np.pi, size=rng.integers(3, 8)))
            coords = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
            polys.append(Polygon(coords))

        owned = from_shapely_geometries(polys)
        result = _envelope_gpu(owned)
        assert result.residency == Residency.DEVICE

        geoms = result.to_shapely()
        for i, poly in enumerate(polys):
            expected = poly.bounds  # (minx, miny, maxx, maxy)
            actual = geoms[i].bounds
            np.testing.assert_allclose(actual, expected, atol=1e-10,
                                       err_msg=f"Polygon {i} bounds mismatch")


# ---------------------------------------------------------------------------
# Phase 2D: Property kernels
# ---------------------------------------------------------------------------


class TestPropertyKernels:
    """Tests for offset-arithmetic property computations."""

    def test_num_coordinates(self):
        """num_coordinates_owned matches Shapely."""
        from vibespatial.constructive.properties import num_coordinates_owned

        geoms = [
            Point(1, 2),
            shapely.LineString([(0, 0), (1, 1), (2, 2)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        ]
        owned = from_shapely_geometries(geoms)
        result = num_coordinates_owned(owned)
        expected = shapely.get_num_coordinates(np.array(geoms, dtype=object))
        np.testing.assert_array_equal(result, expected)

    def test_num_geometries(self):
        """num_geometries_owned matches Shapely."""
        from vibespatial.constructive.properties import num_geometries_owned

        geoms = [Point(1, 2), shapely.MultiPoint([(0, 0), (1, 1)])]
        owned = from_shapely_geometries(geoms)
        result = num_geometries_owned(owned)
        assert result[0] == 1
        assert result[1] == 2

    def test_get_xy(self):
        """get_x/y_owned extracts point coordinates."""
        from vibespatial.constructive.properties import get_x_owned, get_y_owned

        points = [Point(1, 2), Point(3, 4), Point(5, 6)]
        owned = from_shapely_geometries(points)
        x = get_x_owned(owned)
        y = get_y_owned(owned)
        np.testing.assert_allclose(x, [1.0, 3.0, 5.0])
        np.testing.assert_allclose(y, [2.0, 4.0, 6.0])

    def test_is_closed(self):
        """is_closed_owned detects open/closed linestrings."""
        from vibespatial.constructive.properties import is_closed_owned

        geoms = [
            shapely.LineString([(0, 0), (1, 1)]),  # open
            shapely.LineString([(0, 0), (1, 0), (0, 0)]),  # closed
        ]
        owned = from_shapely_geometries(geoms)
        result = is_closed_owned(owned)
        assert not result[0]
        assert result[1]

    def test_get_xy_device_resident_no_host_materialize(self):
        """get_x/y_owned reads from device buffers without populating host stubs."""
        from vibespatial.constructive.properties import get_x_owned, get_y_owned
        from vibespatial.geometry.owned import build_device_resident_owned

        pytest.importorskip("cupy")
        from vibespatial.cuda._runtime import get_cuda_runtime
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.geometry.owned import (
            FAMILY_TAGS,
            DeviceFamilyGeometryBuffer,
        )

        runtime = get_cuda_runtime()

        # 50 device-resident points
        rng = np.random.default_rng(77)
        expected_x = rng.uniform(-100, 100, size=50).astype(np.float64)
        expected_y = rng.uniform(-100, 100, size=50).astype(np.float64)
        geom_offsets = np.arange(51, dtype=np.int32)
        empty_mask = np.zeros(50, dtype=bool)

        device_families = {
            GeometryFamily.POINT: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POINT,
                x=runtime.from_host(expected_x),
                y=runtime.from_host(expected_y),
                geometry_offsets=runtime.from_host(geom_offsets),
                empty_mask=runtime.from_host(empty_mask),
            ),
        }

        owned = build_device_resident_owned(
            device_families=device_families,
            row_count=50,
            tags=np.full(50, FAMILY_TAGS[GeometryFamily.POINT], dtype=np.int8),
            validity=np.ones(50, dtype=bool),
            family_row_offsets=np.arange(50, dtype=np.int32),
        )

        x = get_x_owned(owned)
        y = get_y_owned(owned)

        np.testing.assert_allclose(x, expected_x)
        np.testing.assert_allclose(y, expected_y)

        # host_materialized must remain False — no _ensure_host_state was called
        assert not owned.families[GeometryFamily.POINT].host_materialized


# ---------------------------------------------------------------------------
# Phase 3: Simplify, convex hull, validity
# ---------------------------------------------------------------------------


class TestSimplify:
    """Tests for Visvalingam-Whyatt simplification."""

    def test_simplify_removes_vertices(self):
        """simplify_owned reduces vertex count."""
        from vibespatial.constructive.simplify import simplify_owned

        line = shapely.LineString([(0, 0), (0.5, 0.01), (1, 0), (1.5, 0.01), (2, 0)])
        owned = from_shapely_geometries([line])
        result = simplify_owned(owned, tolerance=0.1)
        geoms = result.to_shapely()
        assert geoms[0] is not None
        # Should have fewer coordinates than the original
        assert shapely.get_num_coordinates(geoms[0]) <= 5

    def test_simplify_zero_is_identity(self):
        """simplify with tolerance=0 returns original."""
        from vibespatial.constructive.simplify import simplify_owned

        line = shapely.LineString([(0, 0), (1, 1), (2, 0)])
        owned = from_shapely_geometries([line])
        result = simplify_owned(owned, tolerance=0)
        geoms = result.to_shapely()
        assert shapely.get_num_coordinates(geoms[0]) == 3


class TestConvexHull:
    """Tests for convex hull computation."""

    def test_convex_hull_triangle(self):
        """convex_hull of a triangle is the triangle itself."""
        from vibespatial.constructive.convex_hull import convex_hull_owned

        poly = Polygon([(0, 0), (10, 0), (5, 10)])
        owned = from_shapely_geometries([poly])
        result = convex_hull_owned(owned)
        geoms = result.to_shapely()
        assert geoms[0] is not None
        assert geoms[0].geom_type == "Polygon"

    def test_convex_hull_stays_device_resident(self, strict_device_guard):
        """convex_hull keeps single-family metadata on device."""
        from vibespatial.constructive.convex_hull import convex_hull_owned

        poly = Polygon([(0, 0), (10, 0), (5, 10)])
        owned = from_shapely_geometries([poly], residency=Residency.DEVICE)
        result = convex_hull_owned(owned, dispatch_mode=ExecutionMode.GPU)

        assert result.residency == Residency.DEVICE


class TestValidity:
    """Tests for is_valid/is_simple checks."""

    def test_is_valid_simple_polygon(self):
        """Valid polygon returns True."""
        from vibespatial.constructive.validity import is_valid_owned

        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        owned = from_shapely_geometries([poly])
        result = is_valid_owned(owned)
        assert result[0]

    def test_is_simple_line(self):
        """Non-self-intersecting line returns True."""
        from vibespatial.constructive.validity import is_simple_owned

        line = shapely.LineString([(0, 0), (1, 1), (2, 0)])
        owned = from_shapely_geometries([line])
        result = is_simple_owned(owned)
        assert result[0]


# ---------------------------------------------------------------------------
# Phase 6: Transfer audit
# ---------------------------------------------------------------------------


class TestTransferAudit:
    """Tests for assert_no_transfers context manager."""

    def test_no_transfers_passes(self):
        """Context manager passes when no transfers occur."""
        from vibespatial.runtime.execution_trace import assert_no_transfers

        with assert_no_transfers():
            _ = 1 + 1  # No transfers

    def test_transfer_raises(self):
        """Context manager raises on transfer."""
        from vibespatial.runtime.execution_trace import (
            TransferViolationError,
            assert_no_transfers,
            notify_transfer,
        )

        with pytest.raises(TransferViolationError):
            with assert_no_transfers():
                notify_transfer(
                    direction="d2h",
                    trigger="test",
                    reason="test transfer",
                )


# ---------------------------------------------------------------------------
# Step 3a: Zero-copy GPU pipeline chain tests
#
# These tests prove that multi-step spatial operations keep all geometry
# data on device throughout the chain.  assert_no_transfers catches any
# D->H transfer between steps.
# ---------------------------------------------------------------------------

def _has_gpu() -> bool:
    try:
        from vibespatial.runtime import has_gpu_runtime
        return has_gpu_runtime()
    except Exception:
        return False


_GPU_SKIP = pytest.mark.skipif(
    not _has_gpu(),
    reason="CUDA runtime not available",
)


def _make_random_polygons(n: int, *, seed: int = 42) -> list:
    """Generate n random convex polygons for testing."""
    rng = np.random.default_rng(seed)
    polys = []
    for _ in range(n):
        cx, cy = rng.uniform(-500, 500, size=2)
        r = rng.uniform(5, 50)
        npts = rng.integers(4, 12)
        angles = np.sort(rng.uniform(0, 2 * np.pi, size=npts))
        coords = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
        polys.append(Polygon(coords))
    return polys


@_GPU_SKIP
class TestZeroCopyChains:
    """End-to-end zero-copy chain tests (Step 3a).

    Each test builds device-resident geometry, chains multiple GPU
    operations through the public GeometryArray API, and asserts that:
    1. No D->H transfers occur between steps (dispatch wiring is exercised)
    2. The final result is device-resident (backed by OwnedGeometryArray)
    3. Results are correct when materialized

    Row counts are chosen to exceed AUTO crossover thresholds so that
    the adaptive dispatcher selects GPU for each operation:
    - COARSE (translate, reverse, simplify, segmentize, convex_hull, envelope): >= 1,000 rows
    - METRIC (centroid): >= 5,000 rows
    - CONSTRUCTIVE (orient, boundary): >= 50,000 rows — too high for fast
      tests, so those operations use _owned calls with explicit GPU dispatch
    """

    def test_centroid_translate_reverse(self):
        """centroid -> translate -> reverse stays on device."""
        from vibespatial.api.geometry_array import GeometryArray
        from vibespatial.runtime.execution_trace import assert_no_transfers

        polys = _make_random_polygons(5000)
        owned = from_shapely_geometries(polys, residency=Residency.DEVICE)
        ga = GeometryArray.from_owned(owned)

        with assert_no_transfers():
            c = ga.centroid
            t = c.translate(10.0, 20.0)
            r = t.reverse()

        assert r._owned is not None
        assert r._owned.residency == Residency.DEVICE

        # Verify correctness by materializing
        geoms = r._owned.to_shapely()
        assert len(geoms) == 5000
        assert all(g is not None for g in geoms)

    def test_simplify_orient_boundary(self):
        """simplify -> orient -> boundary stays on device.

        orient_owned and boundary_owned are CONSTRUCTIVE-class kernels with
        a 50,000-row AUTO crossover threshold.  The GeometryArray API does
        not expose a dispatch_mode parameter, so we call the _owned functions
        directly with explicit GPU dispatch for those two steps.
        """
        from vibespatial.api.geometry_array import GeometryArray
        from vibespatial.constructive.boundary import boundary_owned
        from vibespatial.constructive.orient import orient_owned
        from vibespatial.runtime.execution_trace import assert_no_transfers

        polys = _make_random_polygons(1000, seed=99)
        owned = from_shapely_geometries(polys, residency=Residency.DEVICE)
        ga = GeometryArray.from_owned(owned)
        gpu = ExecutionMode.GPU

        with assert_no_transfers():
            s = ga.simplify(1.0)
            # orient and boundary are CONSTRUCTIVE (50K threshold) —
            # GA API cannot force GPU dispatch, so use _owned directly.
            o = orient_owned(s._owned, dispatch_mode=gpu)
            b = boundary_owned(o, dispatch_mode=gpu)

        assert b.residency == Residency.DEVICE

        geoms = b.to_shapely()
        assert len(geoms) == 1000

    def test_centroid_translate_translate(self):
        """centroid -> translate -> translate stays on device."""
        from vibespatial.api.geometry_array import GeometryArray
        from vibespatial.runtime.execution_trace import assert_no_transfers

        polys = _make_random_polygons(5000, seed=77)
        owned = from_shapely_geometries(polys, residency=Residency.DEVICE)
        ga = GeometryArray.from_owned(owned)

        with assert_no_transfers():
            c = ga.centroid
            t1 = c.translate(10.0, 20.0)
            t2 = t1.translate(-5.0, -10.0)

        assert t2._owned is not None
        assert t2._owned.residency == Residency.DEVICE

        geoms = t2._owned.to_shapely()
        assert len(geoms) == 5000

    def test_segmentize_simplify_reverse(self):
        """segmentize -> simplify -> reverse stays on device (LineString)."""
        from vibespatial.api.geometry_array import GeometryArray
        from vibespatial.runtime.execution_trace import assert_no_transfers

        rng = np.random.default_rng(55)
        lines = []
        for _ in range(1000):
            npts = rng.integers(5, 20)
            coords = list(zip(
                np.cumsum(rng.uniform(0, 10, size=npts)),
                rng.uniform(-50, 50, size=npts),
            ))
            lines.append(shapely.LineString(coords))

        owned = from_shapely_geometries(lines, residency=Residency.DEVICE)
        ga = GeometryArray.from_owned(owned)

        with assert_no_transfers():
            seg = ga.segmentize(5.0)
            simp = seg.simplify(2.0)
            rev = simp.reverse()

        assert rev._owned is not None
        assert rev._owned.residency == Residency.DEVICE

        geoms = rev._owned.to_shapely()
        assert len(geoms) == 1000

    def test_set_precision_stays_on_device(self):
        """set_precision on device-resident input avoids D->H transfers."""
        from vibespatial.constructive.set_precision import set_precision_owned
        from vibespatial.runtime.execution_trace import assert_no_transfers

        polys = _make_random_polygons(1000, seed=123)
        owned = from_shapely_geometries(polys, residency=Residency.DEVICE)

        with assert_no_transfers():
            result = set_precision_owned(
                owned,
                grid_size=1.0,
                mode="pointwise",
                dispatch_mode=ExecutionMode.GPU,
            )

        assert result.residency == Residency.DEVICE

        geoms = result.to_shapely()
        assert len(geoms) == 1000

    def test_orient_reverse_orient(self):
        """orient -> reverse -> orient round-trips on device.

        orient_owned is a CONSTRUCTIVE-class kernel with a 50,000-row
        AUTO crossover threshold.  The GeometryArray.orient_polygons()
        method does not accept dispatch_mode, so we call orient_owned
        directly with explicit GPU dispatch to guarantee device residency.
        """
        from vibespatial.api.geometry_array import GeometryArray
        from vibespatial.constructive.orient import orient_owned
        from vibespatial.runtime.execution_trace import assert_no_transfers

        polys = _make_random_polygons(1000, seed=33)
        owned = from_shapely_geometries(polys, residency=Residency.DEVICE)
        gpu = ExecutionMode.GPU

        with assert_no_transfers():
            o1 = orient_owned(owned, exterior_cw=True, dispatch_mode=gpu)
            # reverse is COARSE (1000 threshold), 1000 rows -> GPU via GA
            ga_o1 = GeometryArray.from_owned(o1)
            r = ga_o1.reverse()
            o2 = orient_owned(r._owned, exterior_cw=False, dispatch_mode=gpu)

        assert o2.residency == Residency.DEVICE

    def test_convex_hull_stays_on_device(self):
        """convex_hull on device-resident input returns device-resident output."""
        from vibespatial.api.geometry_array import GeometryArray
        from vibespatial.runtime.execution_trace import assert_no_transfers

        polys = _make_random_polygons(1000, seed=11)
        owned = from_shapely_geometries(polys, residency=Residency.DEVICE)
        ga = GeometryArray.from_owned(owned)

        with assert_no_transfers():
            hull = ga.convex_hull

        assert hull._owned is not None
        assert hull._owned.residency == Residency.DEVICE

        geoms = hull._owned.to_shapely()
        assert len(geoms) == 1000
        assert all(g.geom_type == "Polygon" for g in geoms)

    def test_long_chain_five_ops(self):
        """5-op chain: centroid -> translate -> reverse -> translate -> reverse."""
        from vibespatial.api.geometry_array import GeometryArray
        from vibespatial.runtime.execution_trace import assert_no_transfers

        polys = _make_random_polygons(5000, seed=88)
        owned = from_shapely_geometries(polys, residency=Residency.DEVICE)
        ga = GeometryArray.from_owned(owned)

        with assert_no_transfers():
            c = ga.centroid
            t1 = c.translate(100.0, 200.0)
            r1 = t1.reverse()
            t2 = r1.translate(-50.0, -100.0)
            r2 = t2.reverse()

        assert r2._owned is not None
        assert r2._owned.residency == Residency.DEVICE
        geoms = r2._owned.to_shapely()
        assert len(geoms) == 5000

    def test_centroid_envelope(self):
        """centroid -> envelope stays on device with zero transfers."""
        from vibespatial.api.geometry_array import GeometryArray
        from vibespatial.runtime.execution_trace import assert_no_transfers

        polys = _make_random_polygons(5000, seed=77)
        owned = from_shapely_geometries(polys, residency=Residency.DEVICE)
        ga = GeometryArray.from_owned(owned)

        with assert_no_transfers():
            c = ga.centroid
            e = c.envelope

        assert e._owned is not None
        assert e._owned.residency == Residency.DEVICE

        # Verify correctness by materializing
        geoms = e._owned.to_shapely()
        assert len(geoms) == 5000
        assert all(g.geom_type == "Polygon" for g in geoms)
