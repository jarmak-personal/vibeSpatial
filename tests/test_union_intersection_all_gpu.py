"""Tests for GPU global set operations.

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
from shapely.geometry import MultiPolygon, Polygon, box

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


def test_known_coverage_union_has_no_count_allocation_exports() -> None:
    source = Path("src/vibespatial/constructive/binary_constructive.py").read_text()

    removed_reasons = (
        "binary constructive row-aligned coverage segment allocation fence",
        "binary constructive single-row coverage segment allocation fence",
        "grouped polygon coverage-union validity-count fence",
        "grouped polygon coverage-union valid-group count fence",
    )
    assert all(reason not in source for reason in removed_reasons)


def test_grouped_union_capacity_noops_keep_row_indirected_group_shape() -> None:
    source = Path("src/vibespatial/kernels/constructive/segmented_union.py").read_text()
    start = source.index("def segmented_union_all_device_grouped(")
    function_source = source[start:]

    assert "all_input_rows_present = _capacity_all_valid_noops or" in function_source
    assert "d_current_offsets = d_group_offsets" in function_source
    assert "d_output_has_values" in function_source
    assert function_source.count("_device_row_activity_view(") == 3
    assert ").physicalize_device_rows(allow_capacity_allocation=True)" not in (
        function_source[: function_source.index("d_group_local_by_row")]
    )
    assert "d_valid_positions" not in function_source
    assert "d_live_group_local" not in function_source
    assert "device_valid_nonempty_mask(current)" in function_source
    assert "grouped union degenerate topology admission packet" not in function_source
    assert "d_admission_flags" not in function_source
    assert "safe_parts = device_select_owned_capacity_partitions(" in function_source
    assert "degenerate_parts = device_select_owned_capacity_partitions(" in function_source
    assert "cp.flatnonzero(" not in function_source


def test_grouped_union_partition_recursion_is_monotonic() -> None:
    source = Path("src/vibespatial/kernels/constructive/segmented_union.py").read_text()
    start = source.index("def segmented_union_all_device_grouped(")
    function_source = source[start:]
    tree = ast.parse(function_source)
    function = tree.body[0]
    recursive_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "segmented_union_all_device_grouped"
    ]

    assert len(recursive_calls) == 3
    for call in recursive_calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert {
            "_skip_rectangle_strip",
            "_skip_disjoint_pack",
            "_skip_coverage_area_proof",
        } <= keywords.keys()


def test_grouped_union_coverage_proof_reuses_constructive_residuals() -> None:
    source = Path("src/vibespatial/kernels/constructive/segmented_union.py").read_text()
    start = source.index("def _grouped_union_coverage_failure_device(")
    end = source.index("\n\ndef ", start + 1)
    function_source = source[start:end]

    assert "_grouped_union_constructive_coverage_failure_device(" in function_source
    assert "binary_predicate_expression(" not in function_source
    assert "_grouped_union_topological_coverage_failure_device" not in source


def test_grouped_rectangle_strip_union_uses_device_capacity_partition() -> None:
    source = Path("src/vibespatial/kernels/constructive/segmented_union.py").read_text()
    strip_start = source.index("def _grouped_rectangle_strip_union_device(")
    strip_end = source.index("\ndef ", strip_start + 1)
    strip_source = source[strip_start:strip_end]
    grouped_start = source.index("def segmented_union_all_device_grouped(")
    grouped_source = source[grouped_start:]

    assert "grouped rectangle-strip topology admission packet" not in strip_source
    assert "copy_device_to_host" not in strip_source
    assert "device_mask_owned_capacity(result, d_supported)" in strip_source
    assert "_native_grouped_strip_all_supported" not in source
    assert "d_remainder_rows = ~d_strip_group_mask" in grouped_source
    assert "device_select_owned_capacity_partitions(" in grouped_source


def test_grouped_union_residual_repair_stays_capacity_backed() -> None:
    source = Path("src/vibespatial/kernels/constructive/segmented_union.py").read_text()
    carrier_start = source.index("class _GroupedUnionCoverageFailure:")
    carrier_end = source.index("\n\ndef _grouped_union_failure_from_mask", carrier_start)
    carrier_source = source[carrier_start:carrier_end]
    repair_start = source.index("def _repair_grouped_union_uncovered_rows_device(")
    repair_end = source.index("\n\ndef segmented_union_all_device_grouped(", repair_start)
    repair_source = source[repair_start:repair_end]
    residual_start = source.index("def _grouped_union_residual_capacity_device(")
    residual_source = source[residual_start:repair_start]
    producer_start = source.index("def _grouped_union_constructive_coverage_failure_device(")
    coverage_start = source.index("def _grouped_union_coverage_failure_device(")
    failed_residual_start = source.index("def _failed_input_residuals_against_candidate_parts_gpu(")
    failed_residual_end = source.index("\ndef ", failed_residual_start + 1)
    failed_residual_source = source[failed_residual_start:failed_residual_end]
    producer_source = source[producer_start:coverage_start]

    assert "failed_selection: Any" in carrier_source
    assert "failed_groups: Any" in carrier_source
    assert "failed_positions" not in carrier_source
    assert "failed_group_ids" not in carrier_source
    assert "NativeGroupedSelection(" in residual_source
    assert "build_empty_polygon_rows_device(" in residual_source
    assert "_regroup_native_grouped_parts_with_grouped_union_gpu(" in residual_source
    assert "device_scatter_owned_capacity_selection(" in residual_source
    assert "inputs._device_indexed_take(" in failed_residual_source
    assert "candidate._device_indexed_take(" in failed_residual_source
    assert "physicalize_device_rows" not in failed_residual_source
    assert source.count("_dispatch_grouped_polygon_known_coverage_union_gpu(") == 1
    assert "residual_disjoint_append" not in repair_source
    assert "residual_row_union" not in repair_source
    assert "iterative_residual" not in repair_source
    assert "_repair_depth" not in repair_source
    assert "_grouped_union_failure_count_host(" not in source
    assert "repaired coverage admission scalar fence" not in repair_source
    assert "_partial_repair" not in repair_source
    assert "_grouped_union_failure_area_gap_device" not in source
    assert "binary_constructive_owned(" not in repair_source
    assert "cp.flatnonzero(" not in producer_source
    assert "cp.unique(" not in producer_source
    assert "d_residual_area > cp.float64(0.0)" in producer_source
    assert "d_tolerance" not in producer_source


def test_multipart_explode_never_physicalizes_indexed_rows() -> None:
    source = Path("src/vibespatial/constructive/binary_constructive.py").read_text()

    for function_name in (
        "_explode_polygonal_rows_to_polygon_capacity_gpu",
        "_explode_point_rows_to_point_capacity_gpu",
        "_explode_lineal_rows_to_line_capacity_gpu",
    ):
        start = source.index(f"def {function_name}(")
        end = source.index("\ndef ", start + 1)
        function_source = source[start:end]
        assert "physicalize_device_rows" not in function_source
        assert "return None" in function_source

    helper_start = source.index("def _fixed_or_max_structural_count(")
    helper_end = source.index("\ndef ", helper_start + 1)
    helper_source = source[helper_start:helper_end]
    assert "max_{field}" in helper_source
    assert "structural_upper_bound" in helper_source


def test_grouped_polygon_regroup_does_not_switch_algorithms_on_exception() -> None:
    source = Path("src/vibespatial/constructive/binary_constructive.py").read_text()
    start = source.index("def _regroup_native_grouped_parts_with_grouped_union_gpu(")
    end = source.index("\ndef ", start + 1)
    function_source = source[start:end]

    assert "except Exception" not in function_source
    assert "native grouped multipart union failed" not in function_source
    assert "_materialize_overlay_execution_plan(" in function_source


def test_polygon_global_union_uses_one_native_grouped_carrier() -> None:
    source = Path("src/vibespatial/constructive/union_all.py").read_text()
    dissolve_source = Path("src/vibespatial/overlay/dissolve.py").read_text()

    start = source.index("def _native_grouped_polygon_union_all(")
    end = source.index(
        "\n\n# ---------------------------------------------------------------------------", start
    )
    helper = source[start:end]
    union_start = source.index("def union_all_gpu_owned(")
    coverage_start = source.index("def coverage_union_all_gpu_owned(")
    union_source = source[union_start:coverage_start]

    assert "NativeGrouped.from_sorted_offsets(" in helper
    assert "segmented_union_all_device_grouped(" in helper
    assert "output_row_count=1" in helper
    assert "_native_grouped_polygon_union_all(" in union_source
    assert "_spatially_localize_polygon_union_inputs" not in source
    assert "_try_exact_union_disjoint_bbox_components" not in source
    assert "_try_exact_union_bbox_disjoint_color_subsets" not in source
    assert "_union_all_tree_reduce_gpu" not in dissolve_source
    assert "def union_all_gpu(" not in dissolve_source


def test_polygon_global_union_filters_at_device_row_capacity() -> None:
    source = Path("src/vibespatial/constructive/union_all.py").read_text()
    filter_start = source.index("def _take_valid_nonempty_rows(")
    filter_end = source.index("\ndef ", filter_start + 1)
    filter_source = source[filter_start:filter_end]
    union_start = source.index("def union_all_gpu_owned(")
    coverage_start = source.index("def coverage_union_all_gpu_owned(")
    union_source = source[union_start:coverage_start]
    coverage_source = source[coverage_start:]

    assert "if retain_device_capacity:" in filter_source
    assert "device_mask_owned_capacity(" in filter_source
    assert "retain_device_capacity=polygonal_gpu" in union_source
    assert "retain_device_capacity=polygonal_gpu" in coverage_source
    assert union_source.index("retain_device_capacity=polygonal_gpu") < union_source.index(
        "_native_grouped_polygon_union_all("
    )
    assert coverage_source.index("retain_device_capacity=polygonal_gpu") < coverage_source.index(
        "_native_grouped_polygon_union_all("
    )


def test_gpu_global_reductions_do_not_switch_to_cpu_after_execution_failure() -> None:
    source = Path("src/vibespatial/constructive/union_all.py").read_text()
    tree_start = source.index("def _tree_reduce_global(")
    tree_end = source.index("\ndef _native_grouped_polygon_union_all(", tree_start)
    tree_source = source[tree_start:tree_end]

    assert "binary_constructive_cpu" not in tree_source
    assert "OVERLAY_GPU_FAILURE_THRESHOLD" not in source
    assert "consecutive_gpu_failures" not in tree_source
    assert "GPU tree reduction failed" not in source

    disjoint_start = source.index("def disjoint_subset_union_all_owned(")
    disjoint_end = source.index(
        "\n# ---------------------------------------------------------------------------\n# GPU implementation",
        disjoint_start,
    )
    disjoint_source = source[disjoint_start:disjoint_end]
    assert "except Exception" not in disjoint_source
    assert "gpu_repair_invalid_polygons" not in disjoint_source
    assert "still_invalid_rows" not in disjoint_source
    assert "_tree_reduce_global(" not in disjoint_source
    assert "union_all_gpu_owned(" in disjoint_source


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
        cp = pytest.importorskip("cupy")

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
        assert "binary constructive single-row coverage segment allocation fence" not in reasons

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
        assert "binary constructive row-aligned coverage segment allocation fence" not in reasons

        reset_d2h_transfer_count()
        assert binary_module._all_rows_valid(result)
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
        assert "binary constructive all-validity scalar fence" not in reasons

        grouped_rows = from_shapely_geometries(
            [box(20, 0, 21, 1), box(21, 0, 22, 1)],
            residency=Residency.DEVICE,
        )
        seed_all_validity_cache(grouped_rows)
        grouped_rows._validity = None

        reset_d2h_transfer_count()
        grouped_result = binary_module._dispatch_grouped_polygon_known_coverage_union_gpu(
            grouped_rows,
            cp.asarray([0, 0], dtype=cp.int32),
            output_row_count=1,
            assume_all_valid=True,
            assume_source_rows_valid=True,
        )
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert grouped_result is not None
        assert "grouped polygon coverage-union segment-count fence" not in reasons

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

    def test_aligned_union_capacity_topology_preserves_null_and_exact_rows(self):
        from vibespatial.constructive.binary_constructive import (
            binary_constructive_owned,
        )

        left_geometries = [
            box(0, 0, 2, 2),
            box(10, 0, 12, 2),
            box(20, 0, 24, 4),
            None,
            box(40, 0, 42, 2),
            Polygon(),
            box(60, 0, 62, 2),
            MultiPolygon([box(70, 0, 71, 1), box(73, 0, 74, 1)]),
        ]
        right_geometries = [
            box(1, 0, 3, 2),
            box(14, 0, 16, 2),
            box(21, 1, 22, 2),
            box(30, 0, 31, 1),
            None,
            box(50, 0, 51, 1),
            box(62, 0, 64, 2),
            box(71, 0, 73, 1),
        ]
        left = from_shapely_geometries(left_geometries)
        right = from_shapely_geometries(right_geometries)

        result = binary_constructive_owned(
            "union",
            left,
            right,
            dispatch_mode="gpu",
        )

        got = list(result.to_shapely())
        expected = [
            None if lhs is None or rhs is None else lhs.union(rhs)
            for lhs, rhs in zip(left_geometries, right_geometries, strict=True)
        ]
        for actual, oracle in zip(got, expected, strict=True):
            if oracle is None:
                assert actual is None
            else:
                assert actual is not None
                assert _geom_equiv(actual, oracle, tolerance=1.0e-9)

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

    def test_union_all_lowers_polygon_inputs_to_native_grouped(self, monkeypatch):
        """Polygon union should enter the one-group constructive carrier."""
        from vibespatial.constructive import union_all as union_all_module

        polys = [box(i, 0, i + 2, 2) for i in range(4)]
        owned = from_shapely_geometries(polys)
        called: dict[str, int] = {}

        original = union_all_module._native_grouped_polygon_union_all

        def _wrapped(owned_input, *, precision_plan):
            called["rows"] = owned_input.row_count
            return original(owned_input, precision_plan=precision_plan)

        monkeypatch.setattr(
            union_all_module,
            "_native_grouped_polygon_union_all",
            _wrapped,
        )

        result = union_all_module.union_all_gpu_owned(owned, dispatch_mode="gpu")

        assert called == {"rows": 4}
        assert result.row_count == 1

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

    def test_known_coverage_union_nodes_partial_shared_edges_before_parity(self):
        """Coverage cancellation operates on noded atoms, not source segments."""
        from vibespatial.constructive import binary_constructive as binary_module
        from vibespatial.geometry.owned import seed_all_validity_cache

        left_poly = box(0, 0, 2, 2)
        right_poly = box(2, 1, 3, 2)
        left = from_shapely_geometries([left_poly], residency=Residency.DEVICE)
        right = from_shapely_geometries([right_poly], residency=Residency.DEVICE)
        seed_all_validity_cache(left)
        seed_all_validity_cache(right)

        result = binary_module._dispatch_single_row_polygon_known_coverage_union_gpu(
            left,
            right,
        )

        assert result is not None
        actual = result.to_shapely()[0]
        expected = shapely.union(left_poly, right_poly)
        assert shapely.is_valid(actual)
        assert shapely.equals(actual, expected)

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
        assert reasons == []

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
        assert reasons == []

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

        for fn_name in (
            "union_all_gpu_owned",
            "coverage_union_all_gpu_owned",
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
            assert reasons == []

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

    def test_polygon_union_bypasses_global_pairwise_tree(self, monkeypatch):
        """Polygon global union must not regress to the generic pairwise tree."""
        from vibespatial.constructive import union_all as union_all_module

        monkeypatch.setattr(
            union_all_module,
            "_tree_reduce_global",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("polygon union entered the pairwise tree")
            ),
        )

        polys = [box(i, 0, i + 2, 2) for i in range(8)]
        owned = from_shapely_geometries(polys)
        result = union_all_module.union_all_gpu_owned(owned, dispatch_mode="gpu")

        arr = np.empty(len(polys), dtype=object)
        arr[:] = polys
        expected = shapely.union_all(arr)

        assert _geom_equiv(_to_shapely(result), expected)

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

    def test_native_grouped_failure_propagates_without_cpu_retry(self, monkeypatch):
        """An admitted grouped failure must propagate without changing engines."""
        from vibespatial.constructive import union_all as union_all_module

        def _fail_grouped(*_args, **_kwargs):
            raise RuntimeError("forced native grouped union failure")

        monkeypatch.setattr(
            union_all_module,
            "_native_grouped_polygon_union_all",
            _fail_grouped,
        )
        owned = from_shapely_geometries(
            [box(0, 0, 2, 2), box(1, 0, 3, 2)],
            residency=Residency.DEVICE,
        )
        vibespatial.clear_fallback_events()

        with pytest.raises(RuntimeError, match="forced native grouped union failure"):
            union_all_module.union_all_gpu_owned(owned, dispatch_mode="gpu")

        events = vibespatial.get_fallback_events(clear=True)
        assert events == []

    def test_strict_native_pairwise_decline_is_not_swallowed(self, monkeypatch):
        """Pairwise strict-native declines inside tree reduction must propagate."""
        from vibespatial.constructive import union_all as union_all_module

        monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

        def _strict_grouped_decline(*_args, **_kwargs):
            raise StrictNativeFallbackError("forced grouped strict decline")

        monkeypatch.setattr(
            union_all_module,
            "_native_grouped_polygon_union_all",
            _strict_grouped_decline,
        )
        owned = from_shapely_geometries(
            [box(0, 0, 2, 2), box(1, 0, 3, 2)],
            residency=Residency.DEVICE,
        )

        with pytest.raises(StrictNativeFallbackError, match="forced grouped strict decline"):
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

    def test_native_grouped_failure_propagates_without_cpu_retry(self, monkeypatch):
        """Coverage reduction must not switch engines after admission."""
        from vibespatial.constructive import union_all as union_all_module

        def _fail_grouped(*_args, **_kwargs):
            raise RuntimeError("forced native coverage failure")

        monkeypatch.setattr(
            union_all_module,
            "_native_grouped_polygon_union_all",
            _fail_grouped,
        )
        owned = from_shapely_geometries(
            [box(0, 0, 1, 1), box(1, 0, 2, 1)],
            residency=Residency.DEVICE,
        )
        vibespatial.clear_fallback_events()

        with pytest.raises(RuntimeError, match="forced native coverage failure"):
            union_all_module.coverage_union_all_gpu_owned(owned, dispatch_mode="gpu")

        events = vibespatial.get_fallback_events(clear=True)
        assert events == []


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

    def test_intersection_failure_propagates_without_cpu_retry(self, monkeypatch):
        """Intersection execution failure must not switch engines."""
        from vibespatial.constructive import union_all as union_all_module

        def _fail_tree_reduce(*_args, **_kwargs):
            raise RuntimeError("forced intersection tree failure")

        monkeypatch.setattr(union_all_module, "_tree_reduce_global", _fail_tree_reduce)
        owned = from_shapely_geometries(
            [box(0, 0, 3, 3), box(1, 1, 4, 4)],
            residency=Residency.DEVICE,
        )
        vibespatial.clear_fallback_events()

        with pytest.raises(RuntimeError, match="forced intersection tree failure"):
            union_all_module.intersection_all_gpu_owned(owned, dispatch_mode="gpu")

        events = vibespatial.get_fallback_events(clear=True)
        assert events == []


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

    def test_public_union_all_gpu_failure_propagates_without_shapely(self, monkeypatch):
        """Wrapper-level native failures must not execute a second engine."""
        import vibespatial.constructive.union_all as union_all_module

        def _fail_native_reduction(*_args, **_kwargs):
            raise RuntimeError("forced public wrapper reduction failure")

        def _fail_shapely(*_args, **_kwargs):
            raise AssertionError("strict native should stop before shapely")

        monkeypatch.setattr(union_all_module, "union_all_gpu_owned", _fail_native_reduction)
        monkeypatch.setattr(shapely, "union_all", _fail_shapely)
        vibespatial.clear_fallback_events()

        ga = GeometryArray.from_owned(from_shapely_geometries([box(0, 0, 1, 1), box(2, 0, 3, 1)]))

        with pytest.raises(RuntimeError, match="forced public wrapper reduction failure"):
            ga.union_all(method="unary")

        assert vibespatial.get_fallback_events(clear=True) == []
