"""Tests for GPU tree-reduction global set operations.

Covers:
  - union_all_gpu_owned (vibeSpatial-247.4.7)
  - coverage_union_all_gpu_owned (vibeSpatial-247.4.8)
  - intersection_all_gpu_owned (vibeSpatial-247.4.9)
  - unary_union_gpu_owned (vibeSpatial-247.4.10)
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box

import geopandas
import vibespatial
from tests.upstream.geopandas.tests.util import _NATURALEARTH_LOWRES
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime.fallbacks import STRICT_NATIVE_ENV_VAR, StrictNativeFallbackError
from vibespatial.runtime.residency import Residency

# ---------------------------------------------------------------------------
# GPU availability check
# ---------------------------------------------------------------------------

def _has_gpu() -> bool:
    try:
        import cupy as cp
        _ = cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="GPU not available")


def test_union_all_gpu_has_no_raw_cupy_scalar_syncs():
    """Device scalar branches must use named runtime fences."""
    source = Path("src/vibespatial/constructive/union_all.py").read_text()
    tree = ast.parse(source)
    failures: list[str] = []

    cupy_reductions = {
        "all",
        "any",
        "sum",
        "count_nonzero",
        "max",
        "min",
        "nanmax",
        "nanmin",
    }

    def _contains_cupy_reduction(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "cp"
            and child.func.attr in cupy_reductions
            for child in ast.walk(node)
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "item":
                failures.append(f"raw .item() at line {node.lineno}")
            continue
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"bool", "int", "float"}
            and node.args
        ):
            continue
        if _contains_cupy_reduction(node.args[0]):
            failures.append(f"raw {node.func.id}(cp reduction) at line {node.lineno}")

    assert failures == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_shapely(owned: OwnedGeometryArray):
    """Materialise a 1-row OGA to a single Shapely geometry."""
    geoms = owned.to_shapely()
    assert len(geoms) == 1, f"Expected 1-row OGA, got {len(geoms)}"
    return geoms[0]


def _geom_equiv(a, b, *, tolerance: float = 1e-6) -> bool:
    """Check topological equivalence with tolerance for floating-point noise."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a.is_empty and b.is_empty:
        return True
    # Symmetric difference area should be near-zero for equivalent geometries.
    try:
        sym_diff = a.symmetric_difference(b)
        return sym_diff.area < tolerance
    except Exception:
        return False


# ---------------------------------------------------------------------------
# union_all_gpu tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestUnionAllGPU:
    """Tests for union_all_gpu_owned."""

    def test_basic_polygon_union(self):
        """Union of overlapping polygons matches Shapely."""
        from vibespatial.constructive.union_all import union_all_gpu_owned

        polys = [
            box(0, 0, 2, 2),
            box(1, 1, 3, 3),
            box(2, 0, 4, 2),
        ]
        owned = from_shapely_geometries(polys)
        result = union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.union_all(arr)

        assert _geom_equiv(result_geom, expected), (
            f"GPU union_all != Shapely union_all\n"
            f"  GPU area={result_geom.area}, expected area={expected.area}"
        )

    def test_known_coverage_union_reuses_validity_cache_without_scalar_fences(self):
        """Known-coverage paths should trust seeded validity cache metadata."""
        from vibespatial.constructive import binary_constructive as binary_module
        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.geometry.owned import seed_all_validity_cache

        left = from_shapely_geometries([box(0, 0, 1, 1)], residency=Residency.DEVICE)
        right = from_shapely_geometries([box(1, 0, 2, 1)], residency=Residency.DEVICE)
        for owned in (left, right):
            seed_all_validity_cache(owned)
            owned._validity = None

        reset_d2h_transfer_count()
        result = binary_module._dispatch_single_row_polygon_known_coverage_union_gpu(
            left,
            right,
        )
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert result is not None
        assert "binary constructive row-validity scalar fence" not in reasons

        left_rows = from_shapely_geometries(
            [box(0, 0, 1, 1), box(10, 0, 11, 1)],
            residency=Residency.DEVICE,
        )
        right_rows = from_shapely_geometries(
            [box(1, 0, 2, 1), box(11, 0, 12, 1)],
            residency=Residency.DEVICE,
        )
        for owned in (left_rows, right_rows):
            seed_all_validity_cache(owned)
            owned._validity = None

        reset_d2h_transfer_count()
        result = binary_module._dispatch_row_aligned_polygon_known_coverage_union_gpu(
            left_rows,
            right_rows,
        )
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert result is not None
        assert result._current_cached_validity_mask() is not None
        assert "binary constructive all-validity scalar fence" not in reasons

        reset_d2h_transfer_count()
        assert binary_module._all_rows_valid(result)
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
        assert "binary constructive all-validity scalar fence" not in reasons

    def test_face_assembled_union_seeds_validity_cache_without_scalar_fences(self):
        """Exact face assembly should seed validity for immediate union consumers."""
        from vibespatial.constructive import binary_constructive as binary_module
        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.geometry.owned import seed_all_validity_cache

        left = from_shapely_geometries([box(0, 0, 2, 2)], residency=Residency.DEVICE)
        right = from_shapely_geometries([box(1, 0, 3, 2)], residency=Residency.DEVICE)
        for owned in (left, right):
            seed_all_validity_cache(owned)
            owned._validity = None

        reset_d2h_transfer_count()
        result = binary_module._dispatch_single_row_polygon_partition_union_gpu(
            left,
            right,
        )
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert result is not None
        assert result._current_cached_validity_mask() is not None
        assert "binary constructive row-validity scalar fence" not in reasons
        assert "binary constructive all-validity scalar fence" not in reasons

    def test_non_overlapping_polygons(self):
        """Union of non-overlapping polygons."""
        from vibespatial.constructive.union_all import union_all_gpu_owned

        polys = [box(0, 0, 1, 1), box(2, 2, 3, 3), box(4, 4, 5, 5)]
        owned = from_shapely_geometries(polys)
        result = union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.union_all(arr)

        assert abs(result_geom.area - expected.area) < 1e-6

    def test_empty_input(self):
        """Empty input returns an empty geometry."""
        from vibespatial.constructive.union_all import union_all_gpu_owned

        owned = from_shapely_geometries([])
        result = union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)
        assert result_geom.is_empty

    def test_single_row(self):
        """Single-row input returns the input geometry."""
        from vibespatial.constructive.union_all import union_all_gpu_owned

        poly = box(0, 0, 10, 10)
        owned = from_shapely_geometries([poly])
        result = union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)
        assert _geom_equiv(result_geom, poly)

    def test_with_null_rows(self):
        """Null rows are filtered out before union."""
        from vibespatial.constructive.union_all import union_all_gpu_owned

        polys = [box(0, 0, 2, 2), None, box(1, 1, 3, 3)]
        owned = from_shapely_geometries(polys)
        result = union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        # Expected: union of box(0,0,2,2) and box(1,1,3,3)
        valid_polys = [box(0, 0, 2, 2), box(1, 1, 3, 3)]
        arr = np.empty(len(valid_polys), dtype=object)
        arr[:] = valid_polys
        expected = shapely.union_all(arr)

        assert _geom_equiv(result_geom, expected)

    def test_with_grid_size(self):
        """grid_size parameter snaps coordinates before union."""
        from vibespatial.constructive.union_all import union_all_gpu_owned

        polys = [box(0, 0, 1.5, 1.5), box(1, 1, 2.5, 2.5)]
        owned = from_shapely_geometries(polys)
        result = union_all_gpu_owned(owned, grid_size=1.0)
        result_geom = _to_shapely(result)
        # With grid_size=1.0, coordinates snap to integers.
        # Just verify it's a valid geometry with non-zero area.
        assert result_geom.area > 0

    def test_many_polygons(self):
        """Tree reduction with many polygons (tests multiple levels)."""
        from vibespatial.constructive.union_all import union_all_gpu_owned

        # Create 8 overlapping polygons to exercise 3 levels of tree reduction.
        polys = [box(i, 0, i + 2, 2) for i in range(8)]
        owned = from_shapely_geometries(polys)
        result = union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.union_all(arr)

        assert _geom_equiv(result_geom, expected)

    def test_union_all_spatially_localizes_polygon_inputs(self, monkeypatch):
        """Polygon union should apply spatial-local ordering before tree reduction."""
        from vibespatial.constructive import union_all as union_all_module

        polys = [box(i, 0, i + 2, 2) for i in range(4)]
        owned = from_shapely_geometries(polys)
        called: dict[str, int] = {}

        original = union_all_module._spatially_localize_polygon_union_inputs

        def _wrapped(owned_input):
            called["rows"] = owned_input.row_count
            return original(owned_input)

        monkeypatch.setattr(
            union_all_module,
            "_spatially_localize_polygon_union_inputs",
            _wrapped,
        )

        result = union_all_module.union_all_gpu_owned(owned, dispatch_mode="gpu")

        assert called == {"rows": 4}
        assert result.row_count == 1

    def test_union_all_decomposes_disjoint_bbox_components_before_global_tree_reduce(self, monkeypatch):
        """Disjoint polygon clusters should take the exact component decomposition path."""
        from vibespatial.constructive import union_all as union_all_module

        polys = [
            box(0, 0, 1, 1),
            box(0.5, 0, 1.5, 1),
            box(10, 0, 11, 1),
            box(10.5, 0, 11.5, 1),
        ]
        owned = from_shapely_geometries(polys)
        called: dict[str, int] = {}
        original = union_all_module._try_exact_union_disjoint_bbox_components

        def _wrapped(owned_input, *, precision):
            called["rows"] = owned_input.row_count
            return original(owned_input, precision=precision)

        monkeypatch.setattr(
            union_all_module,
            "_try_exact_union_disjoint_bbox_components",
            _wrapped,
        )

        result = union_all_module.union_all_gpu_owned(owned, dispatch_mode="gpu")
        expected = shapely.union_all(np.asarray(polys, dtype=object))

        assert called == {"rows": 4}
        assert _geom_equiv(_to_shapely(result), expected)

    def test_union_all_decomposes_bbox_disjoint_color_subsets_before_global_tree_reduce(self, monkeypatch):
        """Dense overlap workloads should compress bbox-disjoint color classes first."""
        from vibespatial.constructive import union_all as union_all_module

        polys = [box(float(i), 0.0, float(i) + 10.0, 1.0) for i in range(64)]
        owned = from_shapely_geometries(polys)
        partial_calls: list[int] = []
        original = union_all_module.disjoint_subset_union_all_owned

        def _wrapped(owned_input, **kwargs):
            partial_calls.append(owned_input.row_count)
            return original(owned_input, **kwargs)

        monkeypatch.setattr(
            union_all_module,
            "disjoint_subset_union_all_owned",
            _wrapped,
        )

        result = union_all_module.union_all_gpu_owned(owned, dispatch_mode="gpu")
        expected = shapely.union_all(np.asarray(polys, dtype=object))

        assert len(partial_calls) > 1
        assert sum(partial_calls) == len(polys)
        assert max(partial_calls) < len(polys)
        assert _geom_equiv(_to_shapely(result), expected)

    def test_single_row_touching_country_union_is_valid(self):
        """Touch-only country unions should stay valid and exact on GPU."""
        from vibespatial.constructive.binary_constructive import binary_constructive_owned

        world = geopandas.read_file(_NATURALEARTH_LOWRES)
        brazil = world.loc[world["name"] == "Brazil", "geometry"].iloc[0]
        paraguay = world.loc[world["name"] == "Paraguay", "geometry"].iloc[0]

        left = from_shapely_geometries([brazil])
        right = from_shapely_geometries([paraguay])
        result = binary_constructive_owned("union", left, right, dispatch_mode="gpu")
        result_geom = _to_shapely(result)
        expected = shapely.union_all(np.asarray([brazil, paraguay], dtype=object))

        assert bool(shapely.is_valid(result_geom))
        assert _geom_equiv(result_geom, expected)

    def test_single_row_partial_shared_edge_union_is_valid(self):
        """Partial shared-edge unions must use the exact partition path."""
        from vibespatial.constructive.binary_constructive import binary_constructive_owned

        left_poly = box(0, 0, 2, 2)
        right_poly = box(2, 1, 3, 2)

        left = from_shapely_geometries([left_poly])
        right = from_shapely_geometries([right_poly])
        result = binary_constructive_owned("union", left, right, dispatch_mode="gpu")
        result_geom = _to_shapely(result)
        expected = shapely.union(left_poly, right_poly)

        assert bool(shapely.is_valid(result_geom))
        assert _geom_equiv(result_geom, expected)

    def test_union_all_filters_empty_device_rows_before_tree_reduce(self, monkeypatch):
        """Global union should drop structural empty rows without host geometry export."""
        from vibespatial.constructive import union_all as union_all_module
        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )

        owned = from_shapely_geometries(
            [Polygon(), box(0, 0, 1, 1)],
            residency=Residency.DEVICE,
        )
        owned._validity = None
        owned._tags = None
        owned._family_row_offsets = None

        def _fail_host_metadata(*_args, **_kwargs):
            raise AssertionError("union_all empty-row filter should stay device-native")

        def _fail_tree_reduce(*_args, **_kwargs):
            raise AssertionError("empty rows should filter before tree reduction")

        monkeypatch.setattr(type(owned), "_ensure_host_metadata", _fail_host_metadata)
        monkeypatch.setattr(union_all_module, "_tree_reduce_global", _fail_tree_reduce)
        reset_d2h_transfer_count()

        result = union_all_module.union_all_gpu_owned(owned, dispatch_mode="gpu")
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert result.row_count == 1
        assert result.residency is Residency.DEVICE
        assert reasons == [
            "union_all valid-nonempty row count scalar fence",
            "owned geometry device-take nested slice-size allocation fence",
        ]

    def test_coverage_union_all_filters_empty_device_rows_before_tree_reduce(self, monkeypatch):
        """Coverage union uses the same structural device empty-row filter."""
        from vibespatial.constructive import union_all as union_all_module
        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )

        owned = from_shapely_geometries(
            [Polygon(), box(0, 0, 1, 1)],
            residency=Residency.DEVICE,
        )
        owned._validity = None
        owned._tags = None
        owned._family_row_offsets = None

        def _fail_host_metadata(*_args, **_kwargs):
            raise AssertionError("coverage empty-row filter should stay device-native")

        def _fail_tree_reduce(*_args, **_kwargs):
            raise AssertionError("empty rows should filter before tree reduction")

        monkeypatch.setattr(type(owned), "_ensure_host_metadata", _fail_host_metadata)
        monkeypatch.setattr(union_all_module, "_tree_reduce_global", _fail_tree_reduce)
        reset_d2h_transfer_count()

        result = union_all_module.coverage_union_all_gpu_owned(owned, dispatch_mode="gpu")
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert result.row_count == 1
        assert result.residency is Residency.DEVICE
        assert reasons == [
            "coverage_union_all valid-nonempty row count scalar fence",
            "owned geometry device-take nested slice-size allocation fence",
        ]

    def test_global_union_all_empty_device_rows_return_device_empty(self, monkeypatch):
        """All-empty device inputs should not round-trip through host empties."""
        from vibespatial.constructive import union_all as union_all_module
        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )

        def _fail_host_metadata(*_args, **_kwargs):
            raise AssertionError("all-empty global union should stay device-native")

        def _fail_tree_reduce(*_args, **_kwargs):
            raise AssertionError("all-empty input should not enter tree reduction")

        monkeypatch.setattr(
            OwnedGeometryArray,
            "_ensure_host_metadata",
            _fail_host_metadata,
        )
        monkeypatch.setattr(union_all_module, "_tree_reduce_global", _fail_tree_reduce)

        for fn_name, reason in (
            ("union_all_gpu_owned", "union_all valid-nonempty row count scalar fence"),
            (
                "coverage_union_all_gpu_owned",
                "coverage_union_all valid-nonempty row count scalar fence",
            ),
        ):
            owned = from_shapely_geometries(
                [Polygon(), Polygon()],
                residency=Residency.DEVICE,
            )
            owned._validity = None
            owned._tags = None
            owned._family_row_offsets = None
            reset_d2h_transfer_count()

            result = getattr(union_all_module, fn_name)(owned, dispatch_mode="gpu")
            reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

            assert result.row_count == 1
            assert result.residency is Residency.DEVICE
            assert result.device_state is not None
            assert reasons == [reason]

    def test_single_row_union_preserves_enclosed_hole(self):
        """Single-row union must not fill holes created by coverage boundaries."""
        from vibespatial.constructive.binary_constructive import binary_constructive_owned

        parts = np.asarray(
            [
                box(-25.0, -25.0, 1025.0, 25.0),
                box(-25.0, -25.0, 25.0, 1025.0),
                box(-25.0, 175.0, 1025.0, 225.0),
                box(175.0, -25.0, 225.0, 1025.0),
            ],
            dtype=object,
        )
        left_poly = shapely.union(parts[0], parts[1])
        right_poly = shapely.union(parts[2], parts[3])

        result = binary_constructive_owned(
            "union",
            from_shapely_geometries([left_poly]),
            from_shapely_geometries([right_poly]),
            dispatch_mode="gpu",
        )
        result_geom = _to_shapely(result)
        expected = shapely.union_all(parts)

        assert bool(shapely.is_valid(result_geom))
        assert len(getattr(result_geom, "interiors", [])) == 1
        assert _geom_equiv(result_geom, expected)

    def test_single_row_tiny_degenerate_partner_union_preserves_dominant_polygon(self):
        """Near-collinear sliver partners should not invalidate GPU union."""
        from vibespatial.constructive.binary_constructive import binary_constructive_owned

        left_poly = shapely.from_wkt(
            "POLYGON ((360533.11793419765 3077767.725309121, "
            "360483.9865379098 3077624.8725980986, "
            "360531.4637954387 3077624.324575401, "
            "360533.11793419765 3077767.725309121))"
        )
        right_poly = shapely.from_wkt(
            "POLYGON ((360531.46379543876 3077624.3245754014, "
            "360531.4637954393 3077624.3245754014, "
            "360531.4637954401 3077624.324575401, "
            "360533.11793419905 3077767.725309125, "
            "360533.11793419765 3077767.7253091214, "
            "360531.46379543876 3077624.3245754014))"
        )

        left = from_shapely_geometries([left_poly])
        right = from_shapely_geometries([right_poly])
        vibespatial.clear_fallback_events()
        result = binary_constructive_owned("union", left, right, dispatch_mode="gpu")
        fallback_events = vibespatial.get_fallback_events(clear=True)
        result_geom = _to_shapely(result)
        expected = shapely.union(left_poly, right_poly)

        assert fallback_events == []
        assert bool(shapely.is_valid(result_geom))
        assert _geom_equiv(result_geom, expected, tolerance=1.0e-5)

    def test_tiny_union_rescue_rejects_intersecting_exterior_sliver(self):
        """The dominant rescue must not drop tiny area outside the dominant polygon."""
        from vibespatial.constructive.binary_constructive import (
            _dominant_tiny_area_polygon_union_rows_gpu,
        )

        dominant = box(0.0, 0.0, 10.0, 10.0)
        exterior_sliver = box(10.0 - 1.0e-9, 4.0, 10.0 + 1.0e-9, 4.0 + 1.0e-9)
        left = from_shapely_geometries([dominant], residency=Residency.DEVICE)
        right = from_shapely_geometries([exterior_sliver], residency=Residency.DEVICE)

        result = _dominant_tiny_area_polygon_union_rows_gpu(left, right)

        assert result is None

    def test_multi_row_union_uses_batched_partition_not_per_row_exact(self, monkeypatch):
        """Aligned batches must not fall back to one exact overlay graph per row."""
        from vibespatial.constructive import binary_constructive as binary_module

        def _fail_per_row_exact(*_args, **_kwargs):
            raise AssertionError("multi-row union used per-row exact fallback")

        monkeypatch.setattr(
            binary_module,
            "_dispatch_single_row_polygon_union_gpu",
            _fail_per_row_exact,
        )

        left_polys = [box(i * 10, 0, i * 10 + 2, 2) for i in range(3)]
        right_polys = [box(i * 10 + 2, 1, i * 10 + 3, 2) for i in range(3)]
        left = from_shapely_geometries(left_polys)
        right = from_shapely_geometries(right_polys)

        result = binary_module.binary_constructive_owned(
            "union",
            left,
            right,
            dispatch_mode="gpu",
        )
        actual = result.to_shapely()

        assert result.row_count == len(left_polys)
        for got, left_poly, right_poly in zip(actual, left_polys, right_polys, strict=True):
            assert bool(shapely.is_valid(got))
            assert _geom_equiv(got, shapely.union(left_poly, right_poly))

    def test_tree_reduce_batches_rounds(self, monkeypatch):
        """Tree reduction should issue one batched pairwise call per reduction round."""
        from vibespatial.constructive import union_all as union_all_module
        from vibespatial.constructive.binary_constructive import (
            binary_constructive_owned as original_binary_constructive_owned,
        )

        call_rows: list[tuple[int, int]] = []

        def _wrapped_binary_constructive_owned(op, left, right, **kwargs):
            call_rows.append((left.row_count, right.row_count))
            return original_binary_constructive_owned(op, left, right, **kwargs)

        monkeypatch.setattr(
            "vibespatial.constructive.binary_constructive.binary_constructive_owned",
            _wrapped_binary_constructive_owned,
        )

        polys = [box(i, 0, i + 2, 2) for i in range(8)]
        owned = from_shapely_geometries(polys)
        result = union_all_module.union_all_gpu_owned(owned, dispatch_mode="gpu")

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.union_all(arr)

        assert _geom_equiv(_to_shapely(result), expected)
        assert call_rows == [(4, 4), (2, 2), (1, 1)]

    def test_odd_count_polygons(self):
        """Odd number of polygons (tests carry-forward of unpaired element)."""
        from vibespatial.constructive.union_all import union_all_gpu_owned

        polys = [box(i, 0, i + 2, 2) for i in range(5)]
        owned = from_shapely_geometries(polys)
        result = union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.union_all(arr)

        assert _geom_equiv(result_geom, expected)

    def test_tree_reduce_carries_odd_row_between_rounds(self, monkeypatch):
        """Odd-count reduction should carry the tail row without exploding into per-row calls."""
        from vibespatial.constructive import union_all as union_all_module
        from vibespatial.constructive.binary_constructive import (
            binary_constructive_owned as original_binary_constructive_owned,
        )

        call_rows: list[tuple[int, int]] = []

        def _wrapped_binary_constructive_owned(op, left, right, **kwargs):
            call_rows.append((left.row_count, right.row_count))
            return original_binary_constructive_owned(op, left, right, **kwargs)

        monkeypatch.setattr(
            "vibespatial.constructive.binary_constructive.binary_constructive_owned",
            _wrapped_binary_constructive_owned,
        )

        polys = [box(i, 0, i + 2, 2) for i in range(5)]
        owned = from_shapely_geometries(polys)
        result = union_all_module.union_all_gpu_owned(owned, dispatch_mode="gpu")

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.union_all(arr)

        assert _geom_equiv(_to_shapely(result), expected)
        assert call_rows == [(2, 2), (1, 1), (1, 1)]

    def test_strict_native_tree_reduce_fallback_is_observable(self, monkeypatch):
        """Tree-reduction failures must hit strict-native before CPU reduce."""
        from vibespatial.constructive import union_all as union_all_module

        monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
        monkeypatch.setattr(
            union_all_module,
            "_spatially_localize_polygon_union_inputs",
            lambda owned: owned,
        )
        monkeypatch.setattr(
            union_all_module,
            "_try_exact_union_disjoint_bbox_components",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            union_all_module,
            "_try_exact_union_bbox_disjoint_color_subsets",
            lambda *_args, **_kwargs: None,
        )

        def _fail_tree_reduce(*_args, **_kwargs):
            raise RuntimeError("forced union tree failure")

        monkeypatch.setattr(union_all_module, "_tree_reduce_global", _fail_tree_reduce)
        owned = from_shapely_geometries(
            [box(0, 0, 2, 2), box(1, 0, 3, 2)],
            residency=Residency.DEVICE,
        )
        vibespatial.clear_fallback_events()

        with pytest.raises(StrictNativeFallbackError, match=r"constructive\.union_all_gpu"):
            union_all_module.union_all_gpu_owned(owned, dispatch_mode="gpu")

        events = vibespatial.get_fallback_events(clear=True)
        assert events
        assert events[-1].surface == "constructive.union_all_gpu"
        assert events[-1].reason == "GPU tree reduction failed"
        assert events[-1].d2h_transfer is True

    def test_strict_native_pairwise_decline_is_not_swallowed(self, monkeypatch):
        """Pairwise strict-native declines inside tree reduction must propagate."""
        from vibespatial.constructive import union_all as union_all_module

        monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
        monkeypatch.setattr(
            union_all_module,
            "_spatially_localize_polygon_union_inputs",
            lambda owned: owned,
        )
        monkeypatch.setattr(
            union_all_module,
            "_try_exact_union_disjoint_bbox_components",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            union_all_module,
            "_try_exact_union_bbox_disjoint_color_subsets",
            lambda *_args, **_kwargs: None,
        )

        def _strict_pairwise_decline(*_args, **_kwargs):
            raise StrictNativeFallbackError("forced pairwise strict decline")

        monkeypatch.setattr(
            "vibespatial.constructive.binary_constructive.binary_constructive_owned",
            _strict_pairwise_decline,
        )
        owned = from_shapely_geometries(
            [box(0, 0, 2, 2), box(1, 0, 3, 2)],
            residency=Residency.DEVICE,
        )

        with pytest.raises(StrictNativeFallbackError, match="forced pairwise strict decline"):
            union_all_module.union_all_gpu_owned(owned, dispatch_mode="gpu")

    def test_multipolygon_assembly_stays_device_resident(self, strict_device_guard):
        """Single-row union assembly helper should keep routing metadata on device."""
        from vibespatial.constructive.union_all import _assemble_multipolygon_gpu
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.runtime.residency import Residency

        polys = [box(0, 0, 2, 2), box(1, 1, 3, 3)]
        owned = from_shapely_geometries(polys, residency=Residency.DEVICE)
        result = _assemble_multipolygon_gpu(owned.device_state, {GeometryFamily.POLYGON})

        assert result.residency is Residency.DEVICE
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None


# ---------------------------------------------------------------------------
# coverage_union_all_gpu tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestCoverageUnionAllGPU:
    """Tests for coverage_union_all_gpu_owned."""

    def test_basic_coverage_union(self):
        """Coverage union of non-overlapping tiles matches Shapely."""
        from vibespatial.constructive.union_all import coverage_union_all_gpu_owned

        # Non-overlapping tiles (coverage property).
        polys = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2), box(1, 1, 2, 2)]
        owned = from_shapely_geometries(polys)
        result = coverage_union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.coverage_union_all(arr)

        assert _geom_equiv(result_geom, expected), (
            f"GPU coverage_union_all != Shapely\n"
            f"  GPU area={result_geom.area}, expected area={expected.area}"
        )

    def test_empty_input(self):
        """Empty input returns an empty geometry."""
        from vibespatial.constructive.union_all import coverage_union_all_gpu_owned

        owned = from_shapely_geometries([])
        result = coverage_union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)
        assert result_geom.is_empty

    def test_single_tile(self):
        """Single tile returns identity."""
        from vibespatial.constructive.union_all import coverage_union_all_gpu_owned

        poly = box(0, 0, 10, 10)
        owned = from_shapely_geometries([poly])
        result = coverage_union_all_gpu_owned(owned)
        result_geom = _to_shapely(result)
        assert _geom_equiv(result_geom, poly)

    def test_strict_native_tree_reduce_fallback_is_observable(self, monkeypatch):
        """Coverage reduction failures must hit strict-native before CPU reduce."""
        from vibespatial.constructive import union_all as union_all_module

        monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
        monkeypatch.setattr(
            union_all_module,
            "_spatially_localize_polygon_union_inputs",
            lambda owned: owned,
        )

        def _fail_tree_reduce(*_args, **_kwargs):
            raise RuntimeError("forced coverage tree failure")

        monkeypatch.setattr(union_all_module, "_tree_reduce_global", _fail_tree_reduce)
        owned = from_shapely_geometries(
            [box(0, 0, 1, 1), box(1, 0, 2, 1)],
            residency=Residency.DEVICE,
        )
        vibespatial.clear_fallback_events()

        with pytest.raises(
            StrictNativeFallbackError,
            match=r"constructive\.coverage_union_all_gpu",
        ):
            union_all_module.coverage_union_all_gpu_owned(owned, dispatch_mode="gpu")

        events = vibespatial.get_fallback_events(clear=True)
        assert events
        assert events[-1].surface == "constructive.coverage_union_all_gpu"
        assert events[-1].reason == "GPU tree reduction failed"
        assert events[-1].d2h_transfer is True


# ---------------------------------------------------------------------------
# intersection_all_gpu tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestIntersectionAllGPU:
    """Tests for intersection_all_gpu_owned."""

    def test_basic_intersection(self):
        """Intersection of overlapping polygons matches Shapely."""
        from vibespatial.constructive.union_all import intersection_all_gpu_owned

        polys = [box(0, 0, 3, 3), box(1, 1, 4, 4), box(2, 2, 5, 5)]
        owned = from_shapely_geometries(polys)
        result = intersection_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.intersection_all(arr)

        assert _geom_equiv(result_geom, expected), (
            f"GPU intersection_all != Shapely\n"
            f"  GPU area={result_geom.area}, expected area={expected.area}"
        )

    def test_no_common_region(self):
        """Intersection of non-overlapping polygons is empty."""
        from vibespatial.constructive.union_all import intersection_all_gpu_owned

        polys = [box(0, 0, 1, 1), box(2, 2, 3, 3)]
        owned = from_shapely_geometries(polys)
        result = intersection_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        assert result_geom.is_empty or result_geom.area < 1e-10

    def test_early_termination(self):
        """Early termination works when intersection becomes empty mid-way."""
        from vibespatial.constructive.union_all import intersection_all_gpu_owned

        # First two polygons are disjoint, so intersection is empty.
        # Third polygon is huge -- it should never be processed.
        polys = [box(0, 0, 1, 1), box(5, 5, 6, 6), box(-100, -100, 100, 100)]
        owned = from_shapely_geometries(polys)
        result = intersection_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        assert result_geom.is_empty or result_geom.area < 1e-10

    def test_empty_input(self):
        """Empty input returns an empty geometry."""
        from vibespatial.constructive.union_all import intersection_all_gpu_owned

        owned = from_shapely_geometries([])
        result = intersection_all_gpu_owned(owned)
        result_geom = _to_shapely(result)
        assert result_geom.is_empty

    def test_single_row(self):
        """Single-row input returns the input geometry."""
        from vibespatial.constructive.union_all import intersection_all_gpu_owned

        poly = box(0, 0, 10, 10)
        owned = from_shapely_geometries([poly])
        result = intersection_all_gpu_owned(owned)
        result_geom = _to_shapely(result)
        assert _geom_equiv(result_geom, poly)

    def test_with_null_rows(self):
        """Null rows are skipped in intersection."""
        from vibespatial.constructive.union_all import intersection_all_gpu_owned

        polys = [box(0, 0, 3, 3), None, box(1, 1, 4, 4)]
        owned = from_shapely_geometries(polys)
        result = intersection_all_gpu_owned(owned)
        result_geom = _to_shapely(result)

        # Expected: intersection(box(0,0,3,3), box(1,1,4,4)) = box(1,1,3,3)
        expected = shapely.intersection(box(0, 0, 3, 3), box(1, 1, 4, 4))
        assert _geom_equiv(result_geom, expected)

    def test_strict_native_tree_reduce_fallback_is_observable(self, monkeypatch):
        """Intersection reduction failures must hit strict-native before CPU reduce."""
        from vibespatial.constructive import union_all as union_all_module

        monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

        def _fail_tree_reduce(*_args, **_kwargs):
            raise RuntimeError("forced intersection tree failure")

        monkeypatch.setattr(union_all_module, "_tree_reduce_global", _fail_tree_reduce)
        owned = from_shapely_geometries(
            [box(0, 0, 3, 3), box(1, 1, 4, 4)],
            residency=Residency.DEVICE,
        )
        vibespatial.clear_fallback_events()

        with pytest.raises(
            StrictNativeFallbackError,
            match=r"constructive\.intersection_all_gpu",
        ):
            union_all_module.intersection_all_gpu_owned(owned, dispatch_mode="gpu")

        events = vibespatial.get_fallback_events(clear=True)
        assert events
        assert events[-1].surface == "constructive.intersection_all_gpu"
        assert events[-1].reason == "GPU tree reduction failed"
        assert events[-1].d2h_transfer is True


# ---------------------------------------------------------------------------
# unary_union_gpu tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestUnaryUnionGPU:
    """Tests for unary_union_gpu_owned."""

    def test_basic_unary_union(self):
        """Unary union delegates to union_all_gpu and matches Shapely."""
        from vibespatial.constructive.union_all import unary_union_gpu_owned

        polys = [box(0, 0, 2, 2), box(1, 1, 3, 3)]
        owned = from_shapely_geometries(polys)
        result = unary_union_gpu_owned(owned)
        result_geom = _to_shapely(result)

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.union_all(arr)

        assert _geom_equiv(result_geom, expected)

    def test_empty_input(self):
        """Empty input returns an empty geometry."""
        from vibespatial.constructive.union_all import unary_union_gpu_owned

        owned = from_shapely_geometries([])
        result = unary_union_gpu_owned(owned)
        result_geom = _to_shapely(result)
        assert result_geom.is_empty


# ---------------------------------------------------------------------------
# GeometryArray integration tests (dispatch wiring)
# ---------------------------------------------------------------------------


@requires_gpu
class TestGeometryArrayDispatch:
    """Verify that GeometryArray.union_all/intersection_all dispatch to GPU."""

    def test_union_all_unary_dispatch(self):
        """GeometryArray.union_all(method='unary') dispatches to GPU."""
        polys = [box(0, 0, 2, 2), box(1, 1, 3, 3)]
        owned = from_shapely_geometries(polys)
        ga = GeometryArray.from_owned(owned)
        result = ga.union_all(method="unary")

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.union_all(arr)

        assert _geom_equiv(result, expected)

    def test_union_all_coverage_dispatch(self):
        """GeometryArray.union_all(method='coverage') dispatches to GPU."""
        polys = [box(0, 0, 1, 1), box(1, 0, 2, 1)]
        owned = from_shapely_geometries(polys)
        ga = GeometryArray.from_owned(owned)
        result = ga.union_all(method="coverage")

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.coverage_union_all(arr)

        assert _geom_equiv(result, expected)

    def test_intersection_all_dispatch(self):
        """GeometryArray.intersection_all() dispatches to GPU."""
        polys = [box(0, 0, 3, 3), box(1, 1, 4, 4)]
        owned = from_shapely_geometries(polys)
        ga = GeometryArray.from_owned(owned)
        result = ga.intersection_all()

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.intersection_all(arr)

        assert _geom_equiv(result, expected)

    def test_union_all_strict_native_decline_is_not_swallowed(self, monkeypatch):
        """Strict native errors from GPU reductions must not fall through to Shapely."""
        import vibespatial.constructive.union_all as union_all_module

        def _strict_decline(*_args, **_kwargs):
            raise StrictNativeFallbackError("test strict native union decline")

        monkeypatch.setattr(union_all_module, "union_all_gpu_owned", _strict_decline)

        ga = GeometryArray.from_owned(from_shapely_geometries([box(0, 0, 1, 1), box(2, 0, 3, 1)]))

        with pytest.raises(StrictNativeFallbackError, match="strict native union decline"):
            ga.union_all(method="unary")

    def test_public_union_all_gpu_failure_is_observable_before_shapely(self, monkeypatch):
        """Wrapper-level GPU failures must raise in strict mode before CPU reduce."""
        import vibespatial.constructive.union_all as union_all_module

        def _fail_native_reduction(*_args, **_kwargs):
            raise RuntimeError("forced public wrapper reduction failure")

        def _fail_shapely(*_args, **_kwargs):
            raise AssertionError("strict native should stop before shapely")

        monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
        monkeypatch.setattr(union_all_module, "union_all_gpu_owned", _fail_native_reduction)
        monkeypatch.setattr(shapely, "union_all", _fail_shapely)
        vibespatial.clear_fallback_events()

        ga = GeometryArray.from_owned(
            from_shapely_geometries([box(0, 0, 1, 1), box(2, 0, 3, 1)])
        )

        with pytest.raises(StrictNativeFallbackError, match=r"geopandas\.array\.union_all"):
            ga.union_all(method="unary")

        events = vibespatial.get_fallback_events(clear=True)
        assert events
        assert events[-1].surface == "geopandas.array.union_all"
        assert events[-1].reason == "native global reduction failed; falling back to Shapely"
        assert events[-1].requested.value == "gpu"
