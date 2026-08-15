from __future__ import annotations

import ast
import importlib
import math
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

import vibespatial
import vibespatial.api as geopandas
import vibespatial.api._native_results as native_results_module
from vibespatial import write_geoparquet
from vibespatial.api import GeoDataFrame, GeoSeries, read_file
from vibespatial.api._native_result_core import GeometryNativeResult
from vibespatial.api._native_results import (
    LeftConstructiveResult,
    NativeAttributeTable,
    NativeTabularResult,
    NativeTabularSelection,
    PairwiseConstructiveResult,
    RelationIndexResult,
    to_native_tabular_result,
)
from vibespatial.api._native_state import NativeFrameState, attach_native_state
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.api.testing import assert_geodataframe_equal
from vibespatial.api.tools.overlay import overlay
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import (
    TAG_FAMILIES,
    DiagnosticKind,
    OwnedGeometryArray,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.crossover import WorkloadShape
from vibespatial.runtime.fallbacks import StrictNativeFallbackError
from vibespatial.runtime.materialization import (
    MaterializationBoundary,
    clear_materialization_events,
    get_materialization_events,
)
from vibespatial.runtime.residency import Residency, TransferTrigger
from vibespatial.testing import strict_native_environment

overlay_module = importlib.import_module("vibespatial.api.tools.overlay")
overlay_gpu_module = importlib.import_module("vibespatial.overlay.gpu")
overlay_split_module = importlib.import_module("vibespatial.overlay.split")
segment_primitives_module = importlib.import_module("vibespatial.spatial.segment_primitives")
_SHOOTOUT_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "shootout"
if str(_SHOOTOUT_DIR) not in sys.path:
    sys.path.insert(0, str(_SHOOTOUT_DIR))


@pytest.mark.skipif(not vibespatial.has_gpu_runtime(), reason="GPU runtime required")
def test_overlay_inspects_native_composition_without_owned_physicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibespatial.api._native_result_core import NativeGeometryComposition

    parts = [
        GeometryNativeResult.from_owned(
            from_shapely_geometries([geometry], residency=Residency.DEVICE),
            crs=None,
        )
        for geometry in (box(0.0, 0.0, 1.0, 1.0), box(2.0, 0.0, 3.0, 1.0))
    ]
    composition = NativeGeometryComposition.concat(parts, crs=None)
    values = DeviceGeometryArray._from_composition(composition, crs=None)
    series = GeoSeries(values, crs=None, copy=False)

    def _fail_physicalization(_self):
        raise AssertionError("overlay inspection must not physicalize composition")

    monkeypatch.setattr(
        NativeGeometryComposition,
        "_singular_owned_device",
        _fail_physicalization,
    )

    assert overlay_module._series_owned(series) is None
    assert overlay_module._series_family_summary(series) == (True, True, False, False)
    assert overlay_module._series_non_missing_all_polygons(series) == (True, True)
    assert overlay_module._series_first_geom_type(series) == "Polygon"
    np.testing.assert_array_equal(
        overlay_module._series_polygon_mask(series),
        np.asarray([True, True]),
    )
    assert values.native_composition is composition
    assert values.cached_owned() is None
    assert composition._singular_owned_cache is None


@pytest.mark.skipif(not vibespatial.has_gpu_runtime(), reason="GPU runtime required")
def test_overlay_intersection_preserves_strictly_contained_fp64_sliver() -> None:
    sliver = Polygon(
        [
            (673.2050807568874, 400.0),
            (673.2050807568877, 399.9999999999999),
            (673.2050807568876, 400.0),
        ]
    )
    container = box(600.0, 300.0, 800.0, 500.0)
    left = GeoDataFrame(
        {"left_id": [1]},
        geometry=GeoSeries(
            GeometryArray.from_owned(from_shapely_geometries([sliver], residency=Residency.DEVICE))
        ),
    )
    right = GeoDataFrame(
        {"right_id": [2]},
        geometry=GeoSeries(
            GeometryArray.from_owned(
                from_shapely_geometries([container], residency=Residency.DEVICE)
            )
        ),
    )

    with strict_native_environment():
        result = overlay(left, right, how="intersection")

    assert len(result) == 1
    assert shapely.equals_exact(result.geometry.iloc[0], sliver, tolerance=0.0)


@pytest.mark.skipif(not vibespatial.has_gpu_runtime(), reason="GPU runtime required")
def test_sparse_polygon_composition_proves_logical_family_domain() -> None:
    import cupy as cp

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeGeometryComposition,
        NativeGeometryCompositionPart,
    )
    from vibespatial.geometry.device_array import DeviceGeometryArray

    parts = tuple(
        NativeGeometryCompositionPart(
            geometry=GeometryNativeResult.from_owned(
                from_shapely_geometries([box(float(row), 0.0, float(row + 1), 1.0)]),
                crs=None,
            ),
            output_rows=cp.asarray([row], dtype=cp.int64),
        )
        for row in range(2)
    )
    composition = NativeGeometryComposition(
        parts=parts,
        row_count=2,
        crs=None,
        trusted_singular_rows=True,
    )
    series = GeoSeries(DeviceGeometryArray._from_composition(composition))

    assert overlay_module._series_family_summary(series) == (True, True, False, False)


@pytest.mark.skipif(not vibespatial.has_gpu_runtime(), reason="GPU runtime required")
def test_keep_geom_type_device_mask_preserves_positive_fp64_slivers() -> None:
    import cupy as cp

    left_geom = box(0.0, 0.0, 1_000.0, 1_000.0)
    right_geom = box(999.9999999999, 0.0, 2_000.0, 1_000.0)
    overlap = shapely.intersection(left_geom, right_geom)

    left_owned = from_shapely_geometries([left_geom], residency=Residency.DEVICE)
    right_owned = from_shapely_geometries([right_geom], residency=Residency.DEVICE)
    area_owned = from_shapely_geometries([overlap], residency=Residency.DEVICE)
    left = GeoSeries(GeometryArray.from_owned(left_owned))
    right = GeoSeries(GeometryArray.from_owned(right_owned))

    keep = overlay_module._native_polygon_keep_geom_type_positive_area_device_mask(
        left,
        right,
        np.asarray([0], dtype=np.intp),
        np.asarray([0], dtype=np.intp),
        np.asarray([0], dtype=np.intp),
        area_owned=area_owned,
    )
    assert bool(cp.asnumpy(keep)[0])


def test_few_right_exact_path_uses_canonical_row_indirected_topology() -> None:
    source = Path(overlay_module.__file__).read_text()
    function = source.split("def _few_right_intersection_owned(", 1)[1].split(
        "\ndef ",
        1,
    )[0]

    assert ".device_take(" in function
    assert "_expand_right_segments_for_pair_rows" not in source
    assert "_should_cache_few_right_segments" not in source
    assert "cached_right_segments" not in function


def test_broadcast_right_remainder_gate_uses_physical_geometry_shape() -> None:
    class _Buffer:
        def __init__(self, coordinate_count: int, ring_count: int) -> None:
            self.x = np.empty(coordinate_count, dtype=np.float64)
            self.ring_offsets = np.arange(ring_count + 1, dtype=np.int32)

    class _Owned:
        def __init__(self, row_count: int, coordinate_count: int) -> None:
            self.row_count = row_count
            self.families = {"polygon": _Buffer(coordinate_count, row_count)}

    right = _Owned(row_count=1, coordinate_count=5)
    rectangles = _Owned(row_count=32, coordinate_count=160)
    dense_polygons = _Owned(row_count=32, coordinate_count=32_000)

    assert overlay_module._overlay_broadcast_right_work_units(rectangles, right) == 320
    assert overlay_module._overlay_broadcast_right_work_units(dense_polygons, right) == 32_160


def test_overlay_relation_pair_estimate_expands_source_density_by_pair_count() -> None:
    class _Buffer:
        def __init__(self, coordinate_count: int, ring_count: int) -> None:
            self.x = np.empty(coordinate_count, dtype=np.float64)
            self.ring_offsets = np.arange(ring_count + 1, dtype=np.int32)

    class _Owned:
        def __init__(self, row_count: int, coordinate_count: int) -> None:
            self.row_count = row_count
            self.families = {"polygon": _Buffer(coordinate_count, row_count)}

    left = _Owned(row_count=10, coordinate_count=50)
    right = _Owned(row_count=2, coordinate_count=600)

    estimate = overlay_module._overlay_relation_pair_work_estimate(
        left,
        right,
        pair_count=4,
    )

    assert estimate.coordinate_count == 1_220
    assert estimate.segment_count == 1_220
    assert estimate.ring_count == 8
    assert estimate.dispatch_unit_count() == 1_220


def _attach_owned_overlay_state(
    gdf: GeoDataFrame,
    *,
    attribute_storage: str = "pandas",
) -> NativeFrameState:
    non_geometry = gdf.drop(columns=[gdf._geometry_column_name])
    if attribute_storage == "pandas":
        attributes = NativeAttributeTable(dataframe=non_geometry)
    elif attribute_storage == "arrow":
        attributes = NativeAttributeTable(
            arrow_table=pa.Table.from_pandas(non_geometry, preserve_index=False),
            index_override=gdf.index,
            column_override=tuple(non_geometry.columns),
        )
    else:
        raise ValueError(f"unsupported attribute storage {attribute_storage!r}")
    result = NativeTabularResult(
        attributes=attributes,
        geometry=GeometryNativeResult.from_owned(
            from_shapely_geometries(list(gdf.geometry)),
            crs=gdf.crs,
        ),
        geometry_name=gdf._geometry_column_name,
        column_order=tuple(gdf.columns),
    )
    state = NativeFrameState.from_native_tabular_result(result)
    attach_native_state(gdf, state)
    return state


def test_overlay_runtime_d2h_exports_are_operation_named() -> None:
    overlay_dir = Path(__file__).resolve().parents[1] / "src" / "vibespatial" / "overlay"
    offenders: list[str] = []
    for path in sorted(overlay_dir.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "copy_device_to_host"
                and not any(keyword.arg == "reason" for keyword in node.keywords)
            ):
                offenders.append(
                    f"{path.relative_to(overlay_dir.parent.parent.parent)}:{node.lineno}"
                )
    assert offenders == []


@pytest.mark.gpu
def test_broadcast_right_intersection_batch_preserves_nonempty_rows() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.constructive.binary_constructive import (
        _dispatch_polygon_intersection_overlay_broadcast_right_gpu,
    )

    left_values = [box(float(i * 2), 0.0, float(i * 2 + 1), 1.0) for i in range(20)]
    right_value = box(5.25, 0.25, 31.5, 0.75)
    left = from_shapely_geometries(left_values, residency=Residency.DEVICE)
    right = from_shapely_geometries([right_value], residency=Residency.DEVICE)

    result = _dispatch_polygon_intersection_overlay_broadcast_right_gpu(
        left,
        right,
    )

    assert result is not None
    assert result.row_count == len(left_values)
    result_values = np.asarray(GeometryArray.from_owned(result)._data, dtype=object)
    actual_nonempty = ~shapely.is_missing(result_values) & ~shapely.is_empty(result_values)
    expected_values = shapely.intersection(
        np.asarray(left_values, dtype=object),
        right_value,
    )
    expected_nonempty = ~shapely.is_missing(expected_values) & ~shapely.is_empty(expected_values)
    assert actual_nonempty.tolist() == expected_nonempty.tolist()


def test_dissolve_gpu_certification_has_no_raw_cupy_scalar_syncs() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "vibespatial" / "overlay" / "dissolve.py"
    tree = ast.parse(path.read_text(), filename=str(path))

    def _contains_cupy_call(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "cp"
            ):
                return True
        return False

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "item":
            offenders.append(f"{path.name}:{node.lineno}: .item()")
        if (
            isinstance(func, ast.Name)
            and func.id in {"bool", "int", "float"}
            and node.args
            and _contains_cupy_call(node.args[0])
        ):
            offenders.append(f"{path.name}:{node.lineno}: {func.id}(cp.*)")

    assert offenders == []


def test_overlay_bypass_count_fences_are_operation_named() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "vibespatial" / "overlay" / "bypass.py"
    tree = ast.parse(path.read_text(), filename=str(path))

    def _contains_cupy_call(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "cp"
            ):
                return True
        return False

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "item":
            offenders.append(f"{path.name}:{node.lineno}: .item()")
        if (
            isinstance(func, ast.Name)
            and func.id in {"bool", "int", "float"}
            and node.args
            and _contains_cupy_call(node.args[0])
        ):
            offenders.append(f"{path.name}:{node.lineno}: {func.id}(cp.*)")

    assert offenders == []


def test_overlay_public_tool_uses_shared_host_boundary_helper() -> None:
    overlay_path = (
        Path(__file__).resolve().parents[1] / "src" / "vibespatial" / "api" / "tools" / "overlay.py"
    )
    tree = ast.parse(overlay_path.read_text(), filename=str(overlay_path))
    local_helpers = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_overlay_device_to_host", "_overlay_bool_scalar"}
    }

    assert local_helpers == set()


def test_overlay_public_tool_has_no_raw_cupy_scalar_syncs() -> None:
    overlay_path = (
        Path(__file__).resolve().parents[1] / "src" / "vibespatial" / "api" / "tools" / "overlay.py"
    )
    tree = ast.parse(overlay_path.read_text(), filename=str(overlay_path))
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
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "item":
            failures.append(f"raw .item() at line {node.lineno}")
        if (
            isinstance(func, ast.Name)
            and func.id in {"bool", "int", "float"}
            and node.args
            and _contains_cupy_reduction(node.args[0])
        ):
            failures.append(f"raw {func.id}(cp reduction) at line {node.lineno}")

    assert failures == []


def test_grouped_polygon_hole_assembly_has_no_count_sizing_exports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "src/vibespatial/api/tools/overlay.py").read_text()

    assert "overlay grouped polygon-hole ring allocation fence" not in source
    assert "overlay grouped polygon-hole coordinate allocation fence" not in source
    assert "overlay passthrough scatter coverage count scalar fence" not in source


def test_overlay_make_valid_consumes_atomic_repair_without_postcheck_export() -> None:
    source = Path(overlay_module.__file__).read_text()
    function = source.split("def _make_valid_geoseries(", 1)[1].split(
        "\ndef ",
        1,
    )[0]

    assert "is_valid_owned" not in function
    assert "remaining_invalid" not in function
    assert "post_repair_keep_mask" not in function
    assert "allow_keep_geom_type_drop_invalid" not in function
    assert "complete-or-decline" in function


def test_overlay_intersection_native_has_no_exception_driven_algorithm_switches() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "src/vibespatial/api/tools/overlay.py").read_text()
    tree = ast.parse(source)
    broad_handlers = [
        handler.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for handler in node.handlers
        if handler.type is None
        or any(
            isinstance(part, ast.Name) and part.id in {"Exception", "BaseException"}
            for part in ast.walk(handler.type)
        )
    ]
    assert broad_handlers == []
    start = source.index("def _overlay_intersection_native(")
    end = source.index("\ndef _overlay_intersection(", start + 1)
    intersection_source = source[start:end]

    assert "except Exception" not in intersection_source
    assert "trimming pool" not in intersection_source
    assert "falling back to gathered" not in intersection_source
    assert "falling back to boundary" not in intersection_source
    assert "falling back to Shapely" not in intersection_source


def test_few_right_polygon_partition_has_no_legacy_sh_host_gate() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "src/vibespatial/api/tools/overlay.py").read_text()

    assert "_few_right_sh_intersection_owned" not in source
    assert "_host_convex_single_ring_polygon_mask" not in source
    assert "overlay few-right non-null sparse rows host boundary" not in source
    assert "overlay few-right sh result validity scalar fence" not in source
    assert "_OVERLAY_HOST_EXACT_SEMANTIC_OVERRIDE_MAX_WORK_UNITS" not in source
    assert "needs_exact_override" not in source

    tree = ast.parse(source)
    preference = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_should_prefer_exact_polygon_gpu"
    )
    assert "keep_geom_type" not in ast.unparse(preference)


def test_grouped_difference_has_no_sequential_or_exception_fallback_executor() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "src/vibespatial/api/tools/overlay.py").read_text()

    assert "_try_device_sequential_grouped_difference" not in source
    assert "native_grouped_sequential_difference_gpu" not in source
    assert "_try_secondary_grouped_difference" not in source
    assert "grouped_overlay_difference_plan_build_failed_gpu" not in source
    assert "grouped_overlay_difference_materialize_failed_gpu" not in source
    assert "grouped_overlay_difference_postcheck_failed_gpu" not in source
    assert "_grouped_difference_has_containment_pairs" not in source
    assert "_grouped_difference_has_right_inside_left_pairs" not in source
    assert "d_right_inside_left" not in source
    assert "_repair_invalid_rows_with_native_group_union" not in source
    assert "rectangle-hole unsupported-row repair" not in source
    assert "_filter_effective_polygon_difference_pairs" not in source
    assert "polygon_difference_effective_pair_filter_declined_gpu" not in source

    grouped_start = source.index("def _native_grouped_union_difference_owned(")
    grouped_end = source.index("\ndef ", grouped_start + 1)
    grouped_source = source[grouped_start:grouped_end]
    assert "def _exact_unioned_difference(" in grouped_source
    assert "NativeDeviceSelection.from_mask(" in grouped_source
    assert "device_select_owned_capacity_partitions(" in grouped_source
    assert "d_supported_group_rows" not in grouped_source
    assert "supported_group_count" not in grouped_source
    assert "d_unsupported_group_rows" not in grouped_source
    assert "cp.flatnonzero(d_supported_groups)" not in grouped_source
    assert "_explode_polygonal_rows_to_polygon_capacity_gpu(" in grouped_source
    assert "unioned_right.physicalize_device_rows(" not in grouped_source
    assert "native_grouped_union_difference_physicalize_device_rows" not in grouped_source
    assert "topology_left.physicalize_device_rows(" not in grouped_source
    assert "native_grouped_union_difference_left_physicalize_device_rows" not in grouped_source
    assert "NativeGroupedSelection(" in grouped_source
    assert "NativeGrouped.from_dense_codes(" not in grouped_source

    for function_name in (
        "_grouped_rectangle_hole_difference_owned",
        "_grouped_polygon_hole_difference_owned",
        "_grouped_polygon_donut_difference_owned",
    ):
        function_start = source.index(f"def {function_name}(")
        function_end = source.index("\ndef ", function_start + 1)
        function_source = source[function_start:function_end]
        assert "flatnonzero" not in function_source
        assert "admission scalar fence" not in function_source
        assert "cp.all(" not in function_source
        assert "_grouped_direct_difference_capacity_partition(" in function_source

    assert "def _group_offsets_size_metadata(" not in source
    grouped_difference_start = source.index("def _grouped_overlay_difference_owned(")
    grouped_difference_end = source.index("\ndef ", grouped_difference_start + 1)
    grouped_difference_source = source[grouped_difference_start:grouped_difference_end]
    assert "_skip_direct_specializations=True" in grouped_difference_source
    assert "d_collective_groups" not in grouped_difference_source
    assert "direct-partition-collective-remainder" not in grouped_difference_source
    assert "d_exact_groups[right_group_rows]" in grouped_difference_source
    assert "device_select_owned_capacity_partitions(" in grouped_difference_source

    capacity_start = source.index("def _grouped_overlay_difference_capacity_owned(")
    capacity_end = source.index("\ndef ", capacity_start + 1)
    capacity_source = source[capacity_start:capacity_end]
    assert "NativeGrouped.from_dense_sorted_offsets(" in capacity_source
    assert "_device_pair_lexicographic_order(" in capacity_source
    assert "_overlay_device_to_host(" not in capacity_source
    assert "unique(" not in capacity_source
    assert "flatnonzero(" not in capacity_source
    assert "maybe_trim_pool_memory" not in capacity_source
    assert "for batch_" not in capacity_source
    assert "def _batched_overlay_difference_owned(" not in source
    assert "def _difference_scatter_indices(" not in source

    constructive_source = (
        repo_root / "src/vibespatial/constructive/binary_constructive.py"
    ).read_text()
    capacity_start = constructive_source.index(
        "def _explode_polygonal_rows_to_polygon_capacity_gpu("
    )
    capacity_end = constructive_source.index("\ndef ", capacity_start + 1)
    capacity_source = constructive_source[capacity_start:capacity_end]
    assert "_polygon_part_selection_from_capacities" in capacity_source
    assert "cp.flatnonzero" not in capacity_source
    assert "count_scatter_total" not in capacity_source
    assert "_device_resolve" not in capacity_source

    selection_start = constructive_source.index("def _polygon_part_selection_from_capacities(")
    selection_end = constructive_source.index("\ndef ", selection_start + 1)
    selection_source = constructive_source[selection_start:selection_end]
    assert "NativeDeviceSelection.from_mask" in selection_source
    assert "_device_indexed_take" in selection_source
    assert "as_capacity_prefix" in selection_source

    indexed_capacity_start = constructive_source.index("def _indexed_polygonal_part_capacities(")
    indexed_capacity_end = constructive_source.index("\ndef ", indexed_capacity_start + 1)
    indexed_capacity_source = constructive_source[indexed_capacity_start:indexed_capacity_end]
    assert "preserve_indexed_view=False" not in indexed_capacity_source
    assert "physicalize_device_rows" not in indexed_capacity_source
    assert "_device_indexed_take" in indexed_capacity_source
    assert "first_level_count_per_row" in indexed_capacity_source
    assert "ring_capacity" in indexed_capacity_source
    assert "coord_capacity" in indexed_capacity_source

    assemble_source = (repo_root / "src/vibespatial/overlay/assemble.py").read_text()
    classify_start = assemble_source.index("def classify_grouped_polygonal_complement_groups_gpu(")
    classify_end = assemble_source.index("\ndef ", classify_start + 1)
    classify_source = assemble_source[classify_start:classify_end]
    classify_parts_start = assemble_source.index(
        "def classify_grouped_polygonal_complement_parts_gpu("
    )
    classify_parts_end = assemble_source.index("\ndef ", classify_parts_start + 1)
    classify_parts_source = assemble_source[classify_parts_start:classify_parts_end]
    assert "left._device_indexed_take(" in classify_parts_source
    assert "d_left_polygon_rows" in classify_source
    assert "d_right_polygon_rows" in classify_source
    assert "left.device_take(" not in classify_source
    assemble_start = assemble_source.index("def assemble_grouped_polygonal_complement_gpu(")
    assemble_end = assemble_source.index("\ndef ", assemble_start + 1)
    complement_source = assemble_source[assemble_start:assemble_end]
    assert "selected_group_rows" not in complement_source
    assert "cp.flatnonzero(~d_support_mask)" not in complement_source
    assert "result_state.validity = d_result_validity" in complement_source
    assert "result_state.tags = cp.where(" in complement_source
    assert "selection.active_capacity_mask()" in complement_source
    assert "preserve_indexed_view=True" in complement_source
    assert "if left.is_indexed_view:" not in complement_source
    assert "right_ring_capacity" in complement_source
    assert "right_coord_capacity" in complement_source
    assert "d_interior_logical_count" in complement_source
    assert "total_part_capacity" in complement_source
    assert "total_ring_capacity" in complement_source
    assert "total_coord_capacity" in complement_source
    assert "\n    interior_count =" not in complement_source

    output_start = assemble_source.index("def _build_device_resident_polygon_output(")
    output_end = assemble_source.index("\ndef ", output_start + 1)
    output_source = assemble_source[output_start:output_end]
    assert "NativeDeviceSelection.from_mask" in output_source
    assert "cp.zeros(output_row_count + 1" in output_source
    assert "cp.arange(output_row_count" in output_source
    assert "cp.flatnonzero" not in output_source
    assert "_device_int_scalar" not in output_source
    assert "polygon_count" not in output_source
    assert "multipolygon_count" not in output_source


def test_many_vs_one_uses_canonical_capacity_partitioner() -> None:
    source = Path("src/vibespatial/api/tools/overlay.py").read_text()
    start = source.index("def _many_vs_one_intersection_owned(")
    end = source.index("\ndef ", start + 1)
    function_source = source[start:end]

    assert "broadcast_right_polygon_intersection_capacity_gpu(" in function_source
    assert "tile_single_row(" not in function_source
    assert "broadcast_right_capacity_partition_gpu" in function_source
    assert "flatnonzero" not in function_source
    assert "_prepare_many_vs_one_intersection_chunks" not in source
    assert "_assemble_indexed_owned_chunks" not in source
    assert "_try_passthrough_scatter_assembly" not in source

    constructive_source = Path("src/vibespatial/constructive/binary_constructive.py").read_text()
    carrier_start = constructive_source.index(
        "def broadcast_right_polygon_intersection_capacity_gpu("
    )
    carrier_end = constructive_source.index("\ndef ", carrier_start + 1)
    carrier_source = constructive_source[carrier_start:carrier_end]
    assert "tile_single_row(right_one, int(left.row_count))" in carrier_source
    assert "_dispatch_partitioned_polygon_intersection_gpu(" in carrier_source
    assert "materialize_broadcast(" not in carrier_source
    assert "flatnonzero" not in carrier_source


def test_few_right_constructive_consumes_planned_group_cardinality() -> None:
    source = Path("src/vibespatial/api/tools/overlay.py").read_text()
    start = source.index("def _few_right_intersection_owned(")
    end = source.index("\ndef ", start + 1)
    function_source = source[start:end]

    assert "_right_group_count: int | None = None" in function_source
    assert "unique_right_count = int(_right_group_count)" in function_source
    assert "cp.unique" not in function_source


def test_pair_source_filter_preserves_device_pair_capacity() -> None:
    source = Path("src/vibespatial/api/tools/overlay.py").read_text()
    start = source.index("def _overlay_intersection_native(")
    end = source.index("\ndef _overlay_intersection(", start + 1)
    intersection_source = source[start:end]

    assert "cp.flatnonzero(~d_keep).size" not in source
    assert "overlay pair-source identity admission scalar fence" not in intersection_source
    assert "pair_selection = NativeDeviceSelection.from_mask(" in intersection_source
    assert "device_mask_owned_capacity(owned, d_keep)" in intersection_source
    assert "NativeTabularSelection(" in intersection_source

    keep_start = intersection_source.index("def _apply_intersection_pair_keep_mask(keep_mask):")
    keep_end = intersection_source.index("\n    used_owned", keep_start)
    device_keep_source = intersection_source[keep_start:keep_end].split(
        "        keep = np.asarray(keep_mask, dtype=bool)",
        maxsplit=1,
    )[0]
    assert "flatnonzero" not in device_keep_source
    assert "copy_device_to_host" not in device_keep_source
    assert "d_idx1 = d_idx1[" not in device_keep_source


def test_keep_geom_type_device_filter_retains_output_capacity() -> None:
    source = Path("src/vibespatial/api/tools/overlay.py").read_text()
    start = source.index("def _filter_polygon_intersection_rows_for_keep_geom_type(")
    end = source.index("\ndef ", start + 1)
    filter_source = source[start:end]

    assert "filtered_owned = device_mask_owned_capacity(area_owned, d_keep)" in filter_source
    assert "area_owned.device_take(d_filtered_rows)" not in filter_source
    assert "kept_count = int(d_filtered_rows.size)" not in filter_source


def test_overlay_difference_nonempty_filter_uses_capacity_selection() -> None:
    source = Path("src/vibespatial/api/tools/overlay.py").read_text()
    start = source.index("def _overlay_difference_native(")
    end = source.index("\ndef _overlay_difference(", start + 1)
    difference_source = source[start:end]

    device_start = difference_source.index(
        "device_keep_rows = _owned_valid_nonempty_mask_device(differences_owned)"
    )
    host_start = difference_source.index(
        "keep_rows = _owned_valid_nonempty_mask(differences_owned)",
        device_start,
    )
    device_source = difference_source[device_start:host_start]
    assert "device_mask_owned_capacity(" in device_source
    assert "_left_constructive_capacity_to_native_tabular_result(" in device_source
    assert "NativeDeviceSelection.from_mask(" in device_source
    assert "flatnonzero" not in device_source
    assert "copy_device_to_host" not in device_source


def test_left_constructive_capacity_reuses_native_frame_attributes() -> None:
    frame = GeoDataFrame(
        {"value": [3, 7]},
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)],
        index=pd.Index(["a", "b"], name="source"),
    )
    state = _attach_owned_overlay_state(frame, attribute_storage="arrow")
    replacement = GeometryNativeResult.from_owned(
        from_shapely_geometries([box(0, 0, 0.5, 1), box(2, 0, 2.5, 1)]),
        crs=frame.crs,
    )

    result = native_results_module._left_constructive_capacity_to_native_tabular_result(
        geometry=replacement,
        df=frame,
        geometry_name=frame._geometry_column_name,
    )

    assert result.attributes is state.attributes
    assert result.index_plan is state.index_plan
    assert result.geometry is replacement
    assert result.provenance.operation == "left_constructive_capacity"


def test_overlay_selected_face_indices_host_bridge_records_materialization() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for selected-face host bridge")
    cp = pytest.importorskip("cupy")
    from vibespatial.overlay.faces import _selected_face_indices_to_host

    clear_materialization_events()
    with strict_native_environment():
        got = _selected_face_indices_to_host(cp.asarray([2, 0], dtype=cp.int32))

    events = get_materialization_events(clear=True)
    assert got.tolist() == [2, 0]
    assert len(events) == 1
    assert events[0].boundary is MaterializationBoundary.INTERNAL_HOST_CONVERSION
    assert events[0].operation == "selected_face_indices_to_host"
    assert events[0].detail == "faces=2, bytes=8"
    assert events[0].d2h_transfer is True
    assert events[0].strict_disallowed is False


def test_overlay_gpu_face_selection_stays_capacity_backed() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    faces_source = (repo_root / "src/vibespatial/overlay/faces.py").read_text()
    selection_start = faces_source.index("def _select_overlay_face_selection_gpu(")
    selection_end = faces_source.index("\ndef ", selection_start + 1)
    selection_source = faces_source[selection_start:selection_end]
    assert "NativeDeviceSelection.from_mask" in selection_source
    assert "flatnonzero" not in selection_source

    gpu_source = (repo_root / "src/vibespatial/overlay/gpu.py").read_text()
    materialize_start = gpu_source.index("def _materialize_overlay_execution_plan(")
    materialize_end = gpu_source.index("\ndef ", materialize_start + 1)
    materialize_source = gpu_source[materialize_start:materialize_end]
    assert "_select_overlay_face_selection_gpu" in materialize_source
    assert "selected_faces" in materialize_source

    assemble_source = (repo_root / "src/vibespatial/overlay/assemble.py").read_text()
    assemble_start = assemble_source.index("def _build_polygon_output_from_faces_gpu(")
    assemble_end = assemble_source.index("\ndef ", assemble_start + 1)
    face_assembly_source = assemble_source[assemble_start:assemble_end]
    assert "selected_face_indices.source_mask()" in face_assembly_source
    assert face_assembly_source.count("_extract_face_boundary_rings_gpu(") == 1
    assert "1 - d_face_selected" not in face_assembly_source
    assert "classify_hole_faces" not in face_assembly_source
    assert "count_boundary_ring_containment_depth" in face_assembly_source
    assert "(d_containment_depth & np.int32(1)) == 0" in face_assembly_source
    assert "(d_containment_depth & np.int32(1)) != 0" in face_assembly_source

    helper_start = assemble_source.index("def _extract_face_boundary_rings_gpu(")
    helper_end = assemble_source.index("\ndef ", helper_start + 1)
    helper_source = assemble_source[helper_start:helper_end]
    assert "NativeDeviceSelection.from_mask" in helper_source
    assert "partition_capacity_positions" in helper_source
    assert "d_boundary_inverse" in helper_source
    assert "ring_count = boundary_capacity // 3" in helper_source
    assert "d_ring_active" in helper_source
    assert "cp.flatnonzero" not in helper_source

    kernel_source = (repo_root / "src/vibespatial/overlay/gpu_kernels.py").read_text()
    boundary_next_start = kernel_source.index("compute_boundary_next(")
    boundary_next_end = kernel_source.index("\n}", boundary_next_start) + 2
    boundary_next_source = kernel_source[boundary_next_start:boundary_next_end]
    assert "boundary_active" in boundary_next_source
    assert "if (!boundary_active[boundary_pos])" in boundary_next_source
    ring_scatter_start = kernel_source.index("scatter_boundary_ring_coordinates(")
    ring_scatter_end = kernel_source.index("\n}", ring_scatter_start) + 2
    ring_scatter_source = kernel_source[ring_scatter_start:ring_scatter_end]
    assert "ring_active" in ring_scatter_source
    assert "if (!ring_active[ring])" in ring_scatter_source
    nesting_section = face_assembly_source[
        face_assembly_source.index("d_ring_active_i8 =") : face_assembly_source.index(
            'with hotpath_stage("overlay.assemble.output_grouping"'
        )
    ]
    assert nesting_section.count("NativeDeviceSelection.from_mask") == 1
    assert "positive_boundary_selection" not in nesting_section
    assert "d_containment_group_start" in nesting_section
    assert "d_containment_group_end" in nesting_section
    assert "cp.flatnonzero" not in nesting_section

    kernel_start = kernel_source.index("assign_holes_to_exteriors(")
    kernel_end = kernel_source.index("\n}", kernel_start) + 2
    kernel_body = kernel_source[kernel_start:kernel_end]
    assert "group_start" in kernel_body
    assert "group_end" in kernel_body

    metrics_start = kernel_source.index("compute_centered_boundary_ring_areas(")
    metrics_end = kernel_source.index("\n}", metrics_start) + 2
    metrics_body = kernel_source[metrics_start:metrics_end]
    assert "origin_x" in metrics_body
    assert "out_area" in metrics_body


def test_overlay_face_assembly_prefers_device_path_without_selected_face_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device face assembly")
    cp = pytest.importorskip("cupy")
    assemble_module = importlib.import_module("vibespatial.overlay.assemble")
    faces_module = importlib.import_module("vibespatial.overlay.faces")
    host_fallback_module = importlib.import_module("vibespatial.overlay.host_fallback")

    sentinel = object()

    def _gpu_builder(half_edge_graph, faces, selected_face_indices, **_kwargs):
        assert hasattr(selected_face_indices, "__cuda_array_interface__")
        return sentinel

    def _host_bridge(_selected_face_indices):
        raise AssertionError("selected faces should stay device-resident")

    def _host_builder(*_args, **_kwargs):
        raise AssertionError("CPU face assembly should not run")

    monkeypatch.setattr(
        assemble_module,
        "_build_polygon_output_from_faces_gpu",
        _gpu_builder,
    )
    monkeypatch.setattr(faces_module, "_selected_face_indices_to_host", _host_bridge)
    monkeypatch.setattr(host_fallback_module, "_build_polygon_output_from_faces", _host_builder)

    result = faces_module._assemble_faces_from_device_indices(
        SimpleNamespace(),
        SimpleNamespace(runtime_selection=None),
        cp.asarray([2, 0], dtype=cp.int32),
    )

    assert result is sentinel


def test_overlay_face_assembly_decline_is_atomic_without_host_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for selected-face host bridge")
    cp = pytest.importorskip("cupy")
    assemble_module = importlib.import_module("vibespatial.overlay.assemble")
    faces_module = importlib.import_module("vibespatial.overlay.faces")

    def _gpu_builder(*_args, **_kwargs):
        return None

    def _host_bridge(_selected_face_indices):
        raise AssertionError("device face assembly decline must be atomic")

    monkeypatch.setattr(
        assemble_module,
        "_build_polygon_output_from_faces_gpu",
        _gpu_builder,
    )
    monkeypatch.setattr(faces_module, "_selected_face_indices_to_host", _host_bridge)

    with pytest.raises(RuntimeError, match="returned no device result"):
        faces_module._assemble_faces_from_device_indices(
            SimpleNamespace(),
            SimpleNamespace(runtime_selection=None),
            cp.asarray([2, 0], dtype=cp.int32),
        )


def test_overlay_nonempty_filter_uses_device_metadata_without_host_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device non-empty filter")
    pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [box(0, 0, 1, 1), Polygon()],
        residency=Residency.DEVICE,
    )

    def _fail_host_state():
        raise AssertionError("device non-empty filtering should not materialize host state")

    monkeypatch.setattr(owned, "_ensure_host_state", _fail_host_state)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        filtered = overlay_gpu_module._filter_non_empty_owned_device(owned)

    assert filtered is not None
    assert filtered.residency is Residency.DEVICE
    assert filtered.row_count == 1
    assert filtered.device_state is not None
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_overlay_group_pair_positions_use_host_known_total_without_scalar_fence() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device group expansion")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    group_starts = cp.asarray([0, 2, 5], dtype=cp.int64)
    group_ends = cp.asarray([2, 5, 6], dtype=cp.int64)

    reset_d2h_transfer_count()
    with assert_zero_d2h_transfers():
        positions = overlay_gpu_module._expand_group_pair_positions(
            group_starts,
            group_ends,
            total_count=6,
        )

    assert positions.get().tolist() == [0, 1, 2, 3, 4, 5]
    reset_d2h_transfer_count()


def test_overlay_device_group_pair_expansion_requires_relation_cardinality() -> None:
    source = Path(overlay_gpu_module.__file__).read_text()
    helper = source.split("def _expand_group_pair_positions(", 1)[1].split("\ndef ", 1)[0]

    assert "overlay grouped pair-position total allocation fence" not in helper
    assert "requires relation cardinality" in helper


def _assert_owned_row_mapping_valid(series: GeoSeries) -> None:
    owned = getattr(series.values, "_owned", None)
    assert owned is not None
    state = owned._ensure_device_state(preserve_indexed_view=True)

    def _host_array(values):
        return values.get() if hasattr(values, "get") else np.asarray(values)

    def _collect_bad_rows(
        validity: np.ndarray,
        tags: np.ndarray,
        family_row_offsets: np.ndarray,
        family_counts: dict[GeometryFamily, int],
    ) -> list[tuple[int, str, int, int]]:
        bad_rows: list[tuple[int, str, int, int]] = []
        for row_index in np.flatnonzero(validity):
            family = TAG_FAMILIES.get(int(tags[row_index]))
            assert family is not None, f"valid row {row_index} has null tag"
            family_count = family_counts[family]
            family_row = int(family_row_offsets[row_index])
            if family_row < 0 or family_row >= family_count:
                bad_rows.append((int(row_index), family.value, family_row, family_count))
                if len(bad_rows) >= 8:
                    break
        return bad_rows

    host_bad_rows = _collect_bad_rows(
        np.asarray(owned.validity, dtype=bool),
        np.asarray(owned.tags),
        np.asarray(owned.family_row_offsets),
        {family: owned.families[family].row_count for family in owned.families},
    )
    device_bad_rows = _collect_bad_rows(
        _host_array(state.validity),
        _host_array(state.tags),
        _host_array(state.family_row_offsets),
        {
            family: int(buffer.geometry_offsets.size) - 1
            for family, buffer in state.families.items()
        },
    )

    assert not host_bad_rows, f"owned host row mapping invalid: {host_bad_rows}"
    assert not device_bad_rows, f"owned device row mapping invalid: {device_bad_rows}"


def _assert_all_geometry_coordinates_finite(series: GeoSeries) -> None:
    for row_index, geom in enumerate(np.asarray(series.array, dtype=object)):
        if geom is None:
            continue
        coords = shapely.get_coordinates(geom)
        if coords.size == 0:
            continue
        assert np.isfinite(coords).all(), f"row {row_index} contained non-finite coordinates"
        assert float(np.abs(coords).max()) < 1.0e7, (
            f"row {row_index} contained implausible coordinate magnitude"
        )


def test_geometry_array_owned_supports_spatial_input_without_materialization() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    owned = from_shapely_geometries(
        [Polygon([(0, 0), (2, 0), (2, 2), (0, 0)])],
        residency=Residency.DEVICE,
    )
    owned.diagnostics.clear()

    array = GeometryArray.from_owned(owned)

    assert array.supports_owned_spatial_input() is True
    assert [
        event for event in owned.diagnostics if event.kind == DiagnosticKind.MATERIALIZATION
    ] == []


def test_overlay_union_keep_geom_type_false_drops_empty_difference_rows() -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2, 3]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    result = overlay(left, right, how="union", keep_geom_type=False)

    assert len(result) == 6
    assert result.geometry.notna().all()


def test_overlay_union_reuses_intersecting_pair_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    real_query = overlay_module._intersecting_index_pairs
    calls = 0

    def _counting_query(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_query(*args, **kwargs)

    monkeypatch.setattr(overlay_module, "_intersecting_index_pairs", _counting_query)
    result = overlay(left, right, how="union")

    assert len(result) == 7
    assert calls == 1


def test_overlay_accepts_logical_polygonal_indexed_owned_with_unused_point_buffer() -> None:
    base = from_shapely_geometries(
        [
            Point(99, 99),
            box(0, 0, 2, 2),
            MultiPolygon([box(3, 0, 5, 2)]),
        ]
    )
    view = OwnedGeometryArray._indexed_view(base, np.asarray([1, 2], dtype=np.int64))
    left = GeoDataFrame(
        {"left_id": [1, 2]},
        geometry=GeoSeries(GeometryArray.from_owned(view)),
    )
    right = GeoDataFrame(
        {"right_id": [10]},
        geometry=GeoSeries([box(1, 1, 4, 3)]),
    )

    assert overlay_module._series_family_summary(left.geometry) == (
        True,
        True,
        False,
        False,
    )
    assert overlay_module._series_non_missing_all_polygons(left.geometry) == (
        True,
        True,
    )

    result = overlay(left, right, how="intersection")

    assert len(result) == 2
    assert set(result.geometry.geom_type) == {"Polygon"}


def test_overlay_intersection_reuses_cached_sjoin_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=DeviceGeometryArray._from_sequence(
            [
                box(0, 0, 2, 2),
                box(3, 0, 5, 2),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=DeviceGeometryArray._from_sequence(
            [
                box(1, 1, 4, 3),
                box(10, 10, 12, 12),
            ]
        ),
    )

    geopandas.sjoin(left, right, predicate="intersects")
    monkeypatch.setattr(
        overlay_module,
        "_intersecting_index_pairs",
        lambda *args, **kwargs: pytest.fail("overlay should reuse cached sjoin pairs"),
    )

    result = overlay(left, right, how="intersection")

    assert len(result) == 2


def test_overlay_intersection_reuses_cached_sjoin_pairs_for_polygon_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2, 3]},
        geometry=DeviceGeometryArray._from_sequence(
            [
                box(0, 0, 2, 2),
                Point(100, 100),
                box(3, 0, 5, 2),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=DeviceGeometryArray._from_sequence(
            [
                box(1, 1, 4, 3),
                box(10, 10, 12, 12),
            ]
        ),
    )

    geopandas.sjoin(left, right, predicate="intersects")
    poly_mask = left.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    left_poly = left[poly_mask]
    monkeypatch.setattr(
        overlay_module,
        "_intersecting_index_pairs",
        lambda *args, **kwargs: pytest.fail(
            "overlay should reuse cached subset-compatible sjoin pairs"
        ),
    )

    result = overlay(left_poly, right, how="intersection")

    assert len(result) == 2


def test_overlay_cached_subset_relation_stays_capacity_backed_static() -> None:
    pair_cache_source = Path(
        importlib.import_module("vibespatial.api.tools._pair_cache").__file__
    ).read_text()
    remap_source = pair_cache_source.split(
        "def _device_subset_remap_result(",
        1,
    )[1].split("\ndef _cached_entry_result", 1)[0]
    overlay_source = Path(overlay_module.__file__).read_text()
    consumer_source = overlay_source.split(
        "def _overlay_relation_selection_intersection_native(",
        1,
    )[1].split("\ndef _owned_subset_is_known_valid_rectangles", 1)[0]
    remnant_source = (
        Path(importlib.import_module("vibespatial.constructive.boundary_remnants").__file__)
        .read_text()
        .split(
            "def polygon_pair_boundary_remnants_capacity_device(",
            1,
        )[1]
        .split("\ndef polygon_make_valid_linework_composition_device", 1)[0]
    )

    assert "NativeRelationSelection(" in remap_source
    assert "NativeDeviceSelection.from_mask(" in remap_source
    assert "cp.flatnonzero" not in remap_source
    assert "_device_intersection_pairs(" not in remap_source
    assert "copy_device_to_host" not in remap_source
    assert "_relation_selection_constructive_to_native_tabular_result(" in consumer_source
    assert "NativeDeviceSelection.from_mask(" in consumer_source
    assert "compact_rowset(" not in consumer_source
    assert "copy_device_to_host" not in consumer_source
    assert "polygon_pair_boundary_remnants_capacity_device(" in consumer_source
    assert overlay_source.count("polygon_pair_boundary_remnants_capacity_device(") >= 2
    assert "polygon_pair_boundary_remnants_device" not in overlay_source
    assert "_geometry_composition_from_owned_parts_at_capacity(" in remnant_source
    assert "cp.arange(row_count" in remnant_source
    assert "cp.flatnonzero" not in remnant_source
    assert "device_take(" not in remnant_source


def test_overlay_intersection_reuses_cached_pairs_when_only_nonparticipating_rows_are_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2, 3]},
        geometry=DeviceGeometryArray._from_sequence(
            [
                box(0, 0, 2, 2),
                Polygon([(20, 20), (22, 22), (20, 22), (22, 20), (20, 20)]),
                box(3, 0, 5, 2),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1]},
        geometry=DeviceGeometryArray._from_sequence([box(1, 1, 4, 3)]),
    )

    geopandas.sjoin(left, right, predicate="intersects")
    monkeypatch.setattr(
        overlay_module,
        "_intersecting_index_pairs",
        lambda *args, **kwargs: pytest.fail(
            "overlay should reuse cached pairs for valid participating rows"
        ),
    )

    result = overlay(left, right, how="intersection")

    assert len(result) == 2


def test_overlay_intersection_single_mask_uses_broadcast_right_carrier_on_gpu() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"name": ["west", "east"]},
        geometry=GeoSeries(
            [
                box(0, 0, 2, 2),
                box(3, 0, 5, 2),
            ]
        ),
    )
    right = GeoDataFrame(
        geometry=GeoSeries([box(1, -1, 4, 3)]),
    )

    vibespatial.clear_dispatch_events()
    result = overlay(left, right, how="intersection")
    events = vibespatial.get_dispatch_events(clear=True)

    expected = GeoDataFrame(
        {"name": ["west", "east"]},
        geometry=GeoSeries(
            [
                box(1, 0, 2, 2),
                box(3, 0, 4, 2),
            ]
        ),
    )
    assert_geodataframe_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_like=True,
    )
    assert any(
        event.surface == "geopandas.overlay"
        and event.implementation == "broadcast_right_capacity_partition_gpu"
        and event.selected is ExecutionMode.GPU
        for event in events
    )


def test_overlay_intersection_keep_geom_type_preserves_geometry_collection_boundary() -> None:
    left = GeoDataFrame(
        {
            "left": [0, 1],
            "geometry": [
                box(0, 0, 1, 1),
                box(1, 1, 3, 3).union(box(1, 3, 5, 5)),
            ],
        }
    )
    right = GeoDataFrame(
        {
            "right": [0, 1],
            "geometry": [
                box(0, 0, 1, 1),
                box(3, 1, 4, 2).union(box(4, 1, 5, 4)),
            ],
        }
    )

    kept = overlay(left, right, keep_geom_type=True)
    assert kept.geometry.geom_type.tolist() == ["Polygon", "Polygon"]

    all_geoms = overlay(left, right, keep_geom_type=False)
    assert all_geoms.geometry.geom_type.tolist() == [
        "Polygon",
        "Point",
        "GeometryCollection",
    ]
    assert all_geoms.geometry.iloc[2].equals(
        GeometryCollection([box(4, 3, 5, 4), LineString([(3, 1), (3, 2)])])
    )


def test_overlay_intersecting_index_pairs_can_bypass_public_sindex_query_for_small_owned_polygons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()

    class _Pairs:
        left_indices = np.asarray([0], dtype=np.int32)
        right_indices = np.asarray([0], dtype=np.int32)

    monkeypatch.setattr(
        overlay_module,
        "generate_bounds_pairs",
        lambda *args, **kwargs: _Pairs(),
    )
    monkeypatch.setattr(
        right.sindex,
        "query",
        lambda *args, **kwargs: pytest.fail(
            "public sindex.query should not run on the bbox fast path"
        ),
    )

    idx1, idx2 = overlay_module._intersecting_index_pairs(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
    )

    assert idx1.tolist() == [0]
    assert idx2.tolist() == [0]


def test_overlay_device_pair_order_uses_soa_integer_radix_keys() -> None:
    source = Path(overlay_module.__file__).read_text()
    function_start = source.index("def _device_pair_lexicographic_order(")
    function_end = source.index("\ndef ", function_start + 1)
    function_source = source[function_start:function_end]

    assert function_source.count("_stable_radix_order_pass") == 3
    assert "cp.lexsort" not in function_source
    assert "cp.stack" not in function_source
    assert "astype(cp.float" not in function_source
    assert source.count("_device_pair_lexicographic_order(d_left, d_right)") == 3


def test_overlay_intersecting_index_pairs_uses_native_relation_for_owned_polygons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibespatial.api._native_relation import NativeRelation
    from vibespatial.runtime import ExecutionMode
    from vibespatial.spatial.query_types import SpatialQueryExecution

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    sindex = right.sindex

    monkeypatch.setattr(overlay_module, "_OVERLAY_BBOX_PAIR_FAST_PATH_MAX_PAIRS", 0)
    monkeypatch.setattr(
        sindex,
        "query",
        lambda *args, **kwargs: pytest.fail("overlay must not use public sindex.query"),
    )

    seen = {}

    def _fake_query_relation(geometry, **kwargs):
        seen["geometry"] = geometry
        seen["kwargs"] = kwargs
        return (
            NativeRelation(
                left_indices=np.asarray([0, 1], dtype=np.int32),
                right_indices=np.asarray([0, 1], dtype=np.int32),
                predicate="intersects",
                left_row_count=2,
                right_row_count=2,
                sorted_by_left=True,
            ),
            SpatialQueryExecution(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU,
                implementation="owned_gpu_spatial_query",
                reason="test native relation",
            ),
        )

    monkeypatch.setattr(sindex, "query_relation", _fake_query_relation)

    idx1, idx2 = overlay_module._intersecting_index_pairs(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
    )

    assert seen["geometry"] is left_owned
    assert seen["kwargs"]["return_device"] is True
    assert seen["kwargs"]["query_row_count"] == left_owned.row_count
    assert idx1.tolist() == [0, 1]
    assert idx2.tolist() == [0, 1]


def test_overlay_symmetric_difference_reuses_intersecting_pair_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    real_query = overlay_module._intersecting_index_pairs
    calls = 0

    def _counting_query(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_query(*args, **kwargs)

    monkeypatch.setattr(overlay_module, "_intersecting_index_pairs", _counting_query)
    result = overlay(left, right, how="symmetric_difference")

    assert len(result) == 4
    assert calls == 1


def test_overlay_symmetric_difference_native_concat_preserves_device_geometry_state() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    points = GeoDataFrame(
        geometry=GeoSeries(
            [
                Point(2, 2),
                Point(3, 4),
                Point(9, 8),
                Point(-12, -15),
            ],
            crs="EPSG:3857",
        )
    )
    buffered = points.copy()
    buffered["geometry"] = buffered.buffer(4)
    buffered["type"] = "plot"
    mask = GeoDataFrame(
        {"attr2": ["site-boundary"]},
        geometry=GeoSeries(
            [Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])],
            crs="EPSG:3857",
        ),
    )

    with strict_native_environment():
        result = overlay(buffered, mask, how="symmetric_difference")

    values = result.geometry.values
    assert values.cached_owned() is None
    assert values.native_composition is not None
    owned = values.physicalize_owned()
    assert owned.residency is Residency.DEVICE
    assert owned.device_state is not None
    assert owned.row_count == len(result)
    assert all(
        not (buffer.host_materialized and buffer.x.size == 0 and buffer.geometry_offsets.size > 1)
        for buffer in owned.families.values()
    )
    assert len(owned.to_shapely()) == len(result)


def test_overlay_identity_reuses_intersecting_pair_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    real_query = overlay_module._intersecting_index_pairs
    calls = 0

    def _counting_query(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_query(*args, **kwargs)

    monkeypatch.setattr(overlay_module, "_intersecting_index_pairs", _counting_query)
    result = overlay(left, right, how="identity")

    assert len(result) == 5
    assert calls == 1


def test_overlay_few_right_keep_geom_type_uses_capacity_partitioner() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"left_id": list(range(16))},
        geometry=GeoSeries(
            [box(float(i), 0.0, float(i + 1), 1.0) for i in range(16)],
            crs="EPSG:3857",
        ),
    )
    right = GeoDataFrame(
        {"right_id": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(-1.0, -1.0), (8.0, -0.5), (8.0, 1.5), (-1.0, 2.0)]),
                Polygon([(8.0001, -0.5), (17.0, -1.0), (17.0, 2.0), (8.0001, 1.5)]),
            ],
            crs="EPSG:3857",
        ),
    )

    vibespatial.clear_dispatch_events()
    with strict_native_environment():
        result = overlay(left, right, how="intersection", keep_geom_type=True)
    dispatch_events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 16
    assert result.geometry.geom_type.eq("Polygon").all()
    assert any(
        event.surface == "geopandas.overlay"
        and event.implementation == "few_right_polygon_partition_gpu"
        and event.selected is ExecutionMode.GPU
        for event in dispatch_events
    )


def test_overlay_partitioned_intersection_classifies_rectangles_without_host_helpers() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    binary_constructive_module = importlib.import_module(
        "vibespatial.constructive.binary_constructive"
    )
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    left_owned = from_shapely_geometries(
        [box(0.0, 0.0, 2.0, 2.0), box(3.0, 0.0, 5.0, 2.0)],
        residency=Residency.DEVICE,
    )
    right_owned = from_shapely_geometries(
        [box(1.0, 1.0, 3.0, 3.0), box(4.0, 1.0, 6.0, 3.0)],
        residency=Residency.DEVICE,
    )

    assert not hasattr(binary_constructive_module, "_host_rectangle_polygon_mask")
    reset_d2h_transfer_count()
    result = overlay_module._few_right_partitioned_polygon_intersection_owned(
        left_owned,
        right_owned,
        dispatch_mode=ExecutionMode.GPU,
    )
    transfers = get_d2h_transfer_events(clear=True)
    reasons = {event.reason for event in transfers}

    assert result is not None
    assert result.residency is Residency.DEVICE
    assert result.row_count == left_owned.row_count
    assert "polygon-rectangle dense single-ring scalar fence" not in reasons
    assert "polygon-rectangle empty-mask scalar fence" not in reasons
    assert "polygon-rectangle ring-offset scalar fence" not in reasons
    assert "polygon-rectangle max-input-vertices scalar fence" not in reasons
    reset_d2h_transfer_count()


def test_mixed_polygon_rect_intersection_partitions_sh_rows_before_exact_tail() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive.binary_constructive import (
        _dispatch_partitioned_polygon_intersection_gpu,
    )

    left_geoms = [
        box(0.0, 0.0, 4.0, 4.0),
        Polygon([(0.0, 2.0), (2.0, 0.0), (4.0, 2.0), (2.0, 4.0), (0.0, 2.0)]),
        Polygon(
            [
                (0.0, 0.0),
                (4.0, 0.0),
                (4.0, 1.0),
                (1.0, 1.0),
                (1.0, 4.0),
                (0.0, 4.0),
                (0.0, 0.0),
            ]
        ),
    ]
    right_geoms = [
        Polygon([(1.0, -1.0), (5.0, 1.5), (1.0, 5.0), (1.0, -1.0)]),
        Polygon(
            [(1.0, 0.5), (3.0, 0.5), (3.8, 2.0), (3.0, 3.5), (1.0, 3.5), (0.2, 2.0), (1.0, 0.5)]
        ),
        Polygon(
            [
                (0.5, -0.5),
                (4.5, -0.5),
                (4.5, 2.0),
                (2.0, 2.0),
                (2.0, 4.5),
                (0.5, 4.5),
                (0.5, -0.5),
            ]
        ),
    ]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

    vibespatial.clear_dispatch_events()
    result = _dispatch_partitioned_polygon_intersection_gpu(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    events = vibespatial.get_dispatch_events(clear=True)

    assert result is not None
    assert result.residency is Residency.DEVICE
    assert result.row_count == len(left_geoms)
    assert any(
        event.implementation == "polygon_intersection_partitioned_capacity_gpu"
        and "partition_counts=device-resident" in event.detail
        for event in events
    )
    expected = [
        left_item.intersection(right_item)
        for left_item, right_item in zip(left_geoms, right_geoms, strict=True)
    ]
    for actual_geom, expected_geom in zip(result.to_shapely(), expected, strict=True):
        assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) < 1e-9


def test_overlay_device_non_rectangles_decline_rectangle_mask_without_host_probe() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    binary_constructive_module = importlib.import_module(
        "vibespatial.constructive.binary_constructive"
    )
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [
            Polygon(
                [
                    (0.0, 0.0),
                    (2.0, 0.0),
                    (1.3, 1.7),
                    (0.0, 0.0),
                ]
            ),
        ],
        residency=Residency.DEVICE,
    )
    assert not hasattr(binary_constructive_module, "_host_rectangle_polygon_mask")

    reset_d2h_transfer_count()
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        device_polygon_shape_mask_bounds,
    )

    result = device_polygon_shape_mask_bounds(owned)
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result is not None
    d_simple, d_rectangle, _d_bounds = result
    assert cp.asnumpy(d_simple).tolist() == [True]
    assert cp.asnumpy(d_rectangle).tolist() == [False]
    assert "owned geometry device-take nested slice-size allocation fence" not in reasons
    assert "owned geometry host metadata validity boundary" not in reasons


def test_overlay_sh_clip_gate_admits_device_rectangle_without_host_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    bypass_module = importlib.import_module("vibespatial.overlay.bypass")
    right_owned = from_shapely_geometries(
        [box(0.0, 0.0, 10.0, 10.0)],
        residency=Residency.DEVICE,
    )

    monkeypatch.setattr(
        right_owned,
        "_ensure_host_family_structure",
        lambda *_args, **_kwargs: pytest.fail(
            "device rectangle clip admission should not inspect host structure"
        ),
    )

    assert bypass_module._is_clip_polygon_sh_eligible(right_owned) == (True, 5)


def test_overlay_intersection_uses_public_sindex_query_in_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    with strict_native_environment():
        vibespatial.clear_dispatch_events()
        result = overlay(left, right, how="intersection")
        events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 3
    assert any(
        event.surface in {"geopandas.sindex.query", "geopandas.overlay.sindex"} for event in events
    )


def test_overlay_intersection_owned_backing_does_not_force_strtree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1]},
        geometry=GeoSeries([box(0, 0, 1, 1)]),
    )
    right = GeoDataFrame(
        {"col2": [1]},
        geometry=GeoSeries([box(2, 2, 3, 3)]),
    )
    left.geometry.values.to_owned()
    right.geometry.values.to_owned()

    assert overlay_module._series_owned(left.geometry) is not None
    assert overlay_module._series_owned(right.geometry) is not None

    def _fail_strtree(*_args, **_kwargs):
        raise AssertionError("owned-backed overlay should not force STRtree materialization")

    sindex_module = importlib.import_module("vibespatial.api.sindex")
    monkeypatch.setattr(overlay_module, "has_gpu_runtime", lambda: True)
    monkeypatch.setattr(
        sindex_module.SpatialIndex,
        "_ensure_strtree",
        _fail_strtree,
        raising=False,
    )
    monkeypatch.setattr(
        overlay_module,
        "_intersecting_index_pairs",
        lambda *_args, **_kwargs: (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        ),
    )

    result, _used_owned = overlay_module._overlay_intersection_native(left, right)

    assert isinstance(result, NativeTabularResult)


def test_overlay_intersection_drops_empty_rows_after_bbox_false_positive_in_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (0, 2), (0, 0)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1]},
        geometry=GeoSeries(
            [
                Polygon([(2, 2), (4, 2), (2, 4), (2, 2)]),
            ]
        ),
    )

    real_query = overlay_module._intersecting_index_pairs
    try:
        overlay_module._intersecting_index_pairs = lambda *args, **kwargs: (
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
        )
        with strict_native_environment():
            result = overlay(left, right, how="intersection")
    finally:
        overlay_module._intersecting_index_pairs = real_query

    assert len(result) == 0


def test_overlay_intersection_single_geometry_only_mask_uses_overlay_carrier() -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                box(0, 0, 2, 2),
                box(2, 0, 4, 2),
            ]
        ),
    )
    right = GeoDataFrame(
        geometry=GeoSeries([box(1, -1, 3, 1)]),
    )

    expected_geometries = shapely.intersection(
        np.asarray(left.geometry.values, dtype=object),
        right.geometry.iloc[0],
    )
    vibespatial.clear_dispatch_events()
    result = overlay(left, right, how="intersection")
    events = vibespatial.get_dispatch_events(clear=True)

    assert result["col1"].tolist() == [1, 2]
    for actual, expected in zip(result.geometry, expected_geometries, strict=True):
        assert shapely.normalize(actual).equals(shapely.normalize(expected))
    assert any(
        event.surface == "geopandas.overlay"
        and event.implementation == "owned_dispatch"
        and "execution_family=broadcast_right_intersection" in event.detail
        and "topology_class=broadcast_mask" in event.detail
        for event in events
    )


def test_overlay_union_promotes_small_pairwise_intersection_in_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    with strict_native_environment():
        vibespatial.clear_dispatch_events()
        result = overlay(left, right, how="union")
        events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 7
    assert any(
        event.surface in {"geopandas.sindex.query", "geopandas.overlay.sindex"} for event in events
    )
    assert any(
        event.surface == "geopandas.array.intersection" and event.selected is ExecutionMode.GPU
        for event in events
    )
    assert any(
        event.surface == "geopandas.array.difference" and event.selected is ExecutionMode.GPU
        for event in events
    )


def test_overlay_difference_keeps_split_polygon_result_in_single_row_under_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (10, 0), (10, 4), (0, 4), (0, 0)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(2, 0), (4, 0), (4, 4), (2, 4), (2, 0)]),
                Polygon([(6, 0), (8, 0), (8, 4), (6, 4), (6, 0)]),
            ]
        ),
    )

    with strict_native_environment():
        vibespatial.clear_dispatch_events()
        result = overlay(left, right, how="difference")
        events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 1
    assert getattr(result.geometry.values, "_owned", None) is not None
    assert result.geometry.iloc[0].geom_type in {"Polygon", "MultiPolygon"}
    assert result.geometry.iloc[0].area > 0
    assert any(
        event.surface == "geopandas.array.difference" and event.selected is ExecutionMode.GPU
        for event in events
    )


def test_grouped_overlay_difference_owned_builds_one_grouped_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibespatial.api._native_grouped import NativeGrouped

    left = from_shapely_geometries(
        [
            box(0, 0, 10, 10),
            box(20, 0, 30, 10),
        ]
    )
    right = from_shapely_geometries(
        [
            box(9, 1, 12, 2),
            box(29, 1, 32, 2),
            box(21, 9, 22, 12),
        ]
    )

    build_calls: list[dict[str, object]] = []
    materialize_calls: list[dict[str, object]] = []
    sentinel = from_shapely_geometries(
        [
            box(0, 0, 9, 9),
            box(20, 0, 29, 9),
        ]
    )

    def _fake_build(left_batch, right_batch, **kwargs):
        build_calls.append(
            {
                "left_rows": left_batch.row_count,
                "right_rows": right_batch.row_count,
                **kwargs,
            }
        )
        return object()

    def _fake_materialize(plan, **kwargs):
        materialize_calls.append(kwargs)
        return sentinel, ExecutionMode.GPU

    monkeypatch.setattr(overlay_gpu_module, "_build_overlay_execution_plan", _fake_build)
    monkeypatch.setattr(
        overlay_gpu_module, "_materialize_overlay_execution_plan", _fake_materialize
    )
    monkeypatch.setattr(overlay_module, "has_gpu_runtime", lambda: True)
    grouped = NativeGrouped.from_sorted_offsets(
        np.asarray([0, 1, 3], dtype=np.int64),
        row_count=3,
        all_groups_observed=True,
        group_size_min=1,
        group_size_max=2,
    )

    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        np.asarray([0, 1, 3], dtype=np.int64),
        dispatch_mode=ExecutionMode.AUTO,
        _native_grouped=grouped,
        _skip_containment_union=True,
    )

    assert result is sentinel
    assert len(build_calls) == 1
    assert build_calls[0]["left_rows"] == 2
    assert build_calls[0]["right_rows"] == 3
    assert build_calls[0]["_row_isolated"] is True
    source_rows = build_calls[0]["_right_geometry_source_rows"]
    segment_source_rows = build_calls[0]["_right_segment_source_rows"]
    if hasattr(source_rows, "__cuda_array_interface__"):
        cp = pytest.importorskip("cupy")
        source_rows = cp.asnumpy(source_rows)
    if hasattr(segment_source_rows, "__cuda_array_interface__"):
        cp = pytest.importorskip("cupy")
        segment_source_rows = cp.asnumpy(segment_source_rows)
    assert np.array_equal(np.asarray(source_rows), np.asarray([0, 1, 1], dtype=np.int32))
    assert np.array_equal(
        np.asarray(segment_source_rows),
        np.asarray([0, 1, 1], dtype=np.int32),
    )
    assert len(materialize_calls) == 1
    assert materialize_calls[0]["operation"] == "difference"
    assert materialize_calls[0]["preserve_row_count"] == 2


def test_grouped_overlay_difference_owned_accepts_device_offsets_without_host_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    left = from_shapely_geometries(
        [
            box(0, 0, 10, 10),
            box(20, 0, 30, 10),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            box(9, 1, 12, 2),
            box(29, 1, 32, 2),
            box(21, 9, 22, 12),
        ],
        residency=Residency.DEVICE,
    )
    sentinel = from_shapely_geometries(
        [
            box(0, 0, 9, 9),
            box(20, 0, 29, 9),
        ],
        residency=Residency.DEVICE,
    )
    observed: dict[str, object] = {}
    original_build = overlay_gpu_module._build_overlay_execution_plan
    original_materialize = overlay_gpu_module._materialize_overlay_execution_plan

    def _fail_group_offsets_host(*_args, **_kwargs):
        raise AssertionError("successful grouped difference should not export full offsets")

    def _fake_build(left_batch, right_batch, **kwargs):
        plan = original_build(left_batch, right_batch, **kwargs)
        if (
            right_batch.row_count == right.row_count
            and kwargs.get("_right_geometry_source_rows") is not None
            and "source_rows" not in observed
        ):
            observed["source_rows"] = kwargs.get("_right_geometry_source_rows")
            observed["segment_source_rows"] = kwargs.get("_right_segment_source_rows")
            observed["exact_plan"] = plan
        return plan

    def _fake_materialize(plan, **kwargs):
        if plan is observed.get("exact_plan") and kwargs.get("operation") == "difference":
            return sentinel, ExecutionMode.GPU
        return original_materialize(plan, **kwargs)

    monkeypatch.setattr(overlay_module, "_group_offsets_to_host", _fail_group_offsets_host)
    monkeypatch.setattr(overlay_gpu_module, "_build_overlay_execution_plan", _fake_build)
    monkeypatch.setattr(
        overlay_gpu_module, "_materialize_overlay_execution_plan", _fake_materialize
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 1, 3], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    source_rows = observed["source_rows"]
    segment_source_rows = observed["segment_source_rows"]
    assert result.residency is Residency.DEVICE
    assert result.row_count == left.row_count
    assert hasattr(source_rows, "__cuda_array_interface__")
    assert hasattr(segment_source_rows, "__cuda_array_interface__")
    assert cp.asnumpy(source_rows).tolist() == [0, 1, 1]
    assert cp.asnumpy(segment_source_rows).tolist() == [0, 1, 1]
    assert not any("grouped difference" in reason for reason in runtime_reasons)


def test_grouped_overlay_difference_device_metadata_path_has_no_runtime_d2h() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    left = from_shapely_geometries(
        [
            box(0, 0, 10, 10),
            box(20, 0, 30, 10),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            box(1, 1, 2, 2),
            box(3, 3, 4, 4),
            box(21, 1, 22, 2),
            box(23, 1, 24, 2),
        ],
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 2, 4], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=2,
    )
    runtime_events = get_d2h_transfer_events(clear=True)

    expected = [
        shapely.difference(
            box(0, 0, 10, 10),
            shapely.union_all(np.asarray([box(1, 1, 2, 2), box(3, 3, 4, 4)], dtype=object)),
        ),
        shapely.difference(
            box(20, 0, 30, 10),
            shapely.union_all(np.asarray([box(21, 1, 22, 2), box(23, 1, 24, 2)], dtype=object)),
        ),
    ]
    actual = np.asarray(result.to_shapely(), dtype=object)

    assert result.row_count == 2
    assert runtime_events == []
    assert sum(event.bytes_transferred for event in runtime_events) <= 32
    for got, want in zip(actual, expected, strict=True):
        assert shapely.symmetric_difference(got, want).area < 1e-8


def test_grouped_overlay_difference_invalid_repair_uses_group_shape_metadata() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    left_geom = Polygon([(0, 0), (12, 0), (12, 8), (0, 8), (0, 0)])
    right_geoms = [
        Polygon([(1, 1), (6, 1), (6, 7), (1, 7), (1, 1)]),
        Polygon([(4, 1), (9, 1), (9, 7), (4, 7), (4, 1)]),
        Polygon([(7, 1), (11, 1), (11, 7), (7, 7), (7, 1)]),
    ]
    left = from_shapely_geometries([left_geom], residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 3], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=1,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    actual = np.asarray(result.to_shapely(), dtype=object)[0]
    expected = shapely.difference(
        left_geom,
        shapely.union_all(np.asarray(right_geoms, dtype=object)),
    )

    assert result.row_count == 1
    assert "overlay grouped difference invalid-row repair count scalar fence" not in runtime_reasons
    assert "overlay grouped difference invalid-row repair allocation fence" not in runtime_reasons
    assert "overlay dissolve native grouped-union valid-row count fence" not in runtime_reasons
    assert "segment extraction total-segments allocation fence" not in runtime_reasons
    assert "segment same-row span summary scalar fence" not in runtime_reasons
    assert "segment filtered candidate total allocation fence" not in runtime_reasons
    assert "geometry analysis homogeneous row-bounds host export" not in runtime_reasons
    assert "owned geometry host metadata validity boundary" not in runtime_reasons
    assert "owned geometry host metadata family-tag boundary" not in runtime_reasons
    assert "owned geometry host metadata family-row-offset boundary" not in runtime_reasons
    assert "owned geometry device-take nested slice-size allocation fence" not in runtime_reasons
    assert shapely.symmetric_difference(actual, expected).area < 1e-8


def test_overlay_difference_device_scatter_avoids_nested_take_fence() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    left_geoms = [
        Polygon([(0, 0), (5, 0), (4, 4), (0, 5), (0, 0)]),
        Polygon([(10, 0), (15, 0), (15, 4), (10, 4), (10, 0)]),
        Polygon([(20, 0), (26, 0), (25, 5), (20, 4), (20, 0)]),
    ]
    right_geoms = [
        box(4, 1, 7, 2),
        box(25, 1, 28, 2),
    ]
    left_owned = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right_owned = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
    left = GeoDataFrame(
        {"parcel_id": [0, 1, 2]},
        geometry=GeoSeries(GeometryArray.from_owned(left_owned)),
    )
    right = GeoDataFrame(
        {"zone_id": [10, 11]},
        geometry=GeoSeries(GeometryArray.from_owned(right_owned)),
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    with strict_native_environment():
        result = overlay(left, right, how="difference")
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert "owned geometry device-take nested slice-size allocation fence" not in runtime_reasons
    assert "owned geometry device-take slice-size allocation fence" not in runtime_reasons
    actual_by_id = {
        int(parcel_id): geom
        for parcel_id, geom in zip(result["parcel_id"], result.geometry, strict=True)
    }
    expected_by_id = {
        0: shapely.difference(left_geoms[0], right_geoms[0]),
        1: left_geoms[1],
        2: shapely.difference(left_geoms[2], right_geoms[1]),
    }
    assert set(actual_by_id) == set(expected_by_id)
    for key, expected in expected_by_id.items():
        assert shapely.symmetric_difference(actual_by_id[key], expected).area < 1e-8


def test_grouped_overlay_difference_device_postcheck_uses_native_validity_expression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    validity_module = importlib.import_module("vibespatial.constructive.validity")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    left = from_shapely_geometries(
        [
            box(0, 0, 10, 10),
            box(20, 0, 30, 10),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            box(9, 1, 12, 2),
            box(29, 1, 32, 2),
            box(21, 9, 22, 12),
        ],
        residency=Residency.DEVICE,
    )
    sentinel = from_shapely_geometries(
        [
            box(0, 0, 9, 9),
            box(20, 0, 29, 9),
        ],
        residency=Residency.DEVICE,
    )

    monkeypatch.setattr(
        validity_module,
        "is_valid_owned",
        lambda *_args, **_kwargs: pytest.fail(
            "device grouped-difference postcheck should not request a host validity mask"
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_group_offsets_to_host",
        lambda *_args, **_kwargs: pytest.fail(
            "device grouped-difference postcheck should not export group offsets"
        ),
    )
    monkeypatch.setattr(
        overlay_gpu_module,
        "_build_overlay_execution_plan",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        overlay_gpu_module,
        "_materialize_overlay_execution_plan",
        lambda *_args, **_kwargs: (sentinel, ExecutionMode.GPU),
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 1, 3], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _skip_direct_specializations=True,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result.residency is Residency.DEVICE
    assert result.row_count == sentinel.row_count
    assert not any("constructive validity" in reason for reason in runtime_reasons)
    assert bool(
        np.all(
            shapely.equals_exact(
                np.asarray(result.to_shapely(), dtype=object),
                np.asarray(sentinel.to_shapely(), dtype=object),
                tolerance=0.0,
            )
        )
    )


def test_grouped_overlay_difference_single_pair_uses_aligned_pairwise_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.constructive import binary_constructive as constructive_module

    left = from_shapely_geometries(
        [
            box(0, 0, 10, 10),
            box(20, 0, 30, 10),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            box(1, 1, 2, 2),
            box(21, 1, 22, 2),
        ],
        residency=Residency.DEVICE,
    )
    sentinel = from_shapely_geometries(
        [
            box(0, 0, 9, 9),
            box(20, 0, 29, 9),
        ],
        residency=Residency.DEVICE,
    )
    calls: list[dict[str, object]] = []

    def _fail_group_offsets_host(*_args, **_kwargs):
        raise AssertionError("aligned single-pair groups should not export offsets")

    def _fake_binary(op, left_arg, right_arg, **kwargs):
        calls.append(
            {
                "op": op,
                "left_is_original": left_arg is left,
                "right_is_original": right_arg is right,
                "dispatch_mode": kwargs.get("dispatch_mode"),
            }
        )
        return sentinel

    monkeypatch.setattr(overlay_module, "_group_offsets_to_host", _fail_group_offsets_host)
    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        _fake_binary,
    )

    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 1, 2], dtype=cp.int64),
        dispatch_mode=ExecutionMode.AUTO,
        _all_groups_observed=True,
        _group_size_min=1,
        _group_size_max=1,
    )

    assert result is sentinel
    assert calls == [
        {
            "op": "difference",
            "left_is_original": True,
            "right_is_original": True,
            "dispatch_mode": ExecutionMode.GPU,
        }
    ]


def test_grouped_overlay_difference_single_pair_respects_cpu_string_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibespatial.constructive import binary_constructive as constructive_module

    left = from_shapely_geometries([box(0, 0, 10, 10)])
    right = from_shapely_geometries([box(1, 1, 2, 2)])
    sentinel = from_shapely_geometries([box(0, 0, 9, 9)])
    calls: list[ExecutionMode | None] = []

    def _fake_binary(op, left_arg, right_arg, **kwargs):
        assert op == "difference"
        assert left_arg is left
        assert right_arg is right
        calls.append(kwargs.get("dispatch_mode"))
        return sentinel

    monkeypatch.setattr(constructive_module, "binary_constructive_owned", _fake_binary)

    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        np.asarray([0, 1], dtype=np.int64),
        dispatch_mode="cpu",
    )

    assert result is sentinel
    assert calls == [ExecutionMode.CPU]


def test_overlay_difference_full_row_grouping_preserves_zero_neighbor_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = from_shapely_geometries(
        [box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10)],
    )
    right = from_shapely_geometries(
        [box(1, 1, 2, 2), box(3, 3, 4, 4), box(41, 1, 42, 2)],
    )
    observed: dict[str, object] = {}

    def _fake_grouped_difference(
        left_batch,
        right_batch,
        group_offsets,
        *,
        _native_grouped,
        **_kwargs,
    ):
        observed["left"] = left_batch
        observed["right_rows"] = right_batch.row_count
        observed["offsets"] = np.asarray(group_offsets)
        observed["grouped"] = _native_grouped
        return left_batch

    monkeypatch.setattr(
        overlay_module,
        "_grouped_overlay_difference_owned",
        _fake_grouped_difference,
    )
    result = overlay_module._grouped_overlay_difference_capacity_owned(
        left,
        right,
        np.asarray([0, 0, 2], dtype=np.int32),
        np.asarray([0, 1, 2], dtype=np.int32),
        None,
        None,
        False,
        ExecutionMode.CPU,
    )

    grouped = observed["grouped"]
    assert result is left
    assert observed["left"] is left
    assert observed["right_rows"] == 3
    np.testing.assert_array_equal(observed["offsets"], [0, 2, 2, 3])
    np.testing.assert_array_equal(grouped.group_offsets, [0, 2, 2, 3])
    np.testing.assert_array_equal(grouped.group_ids, [0, 1, 2])


def test_overlay_difference_keeps_full_row_grouping_on_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    left = from_shapely_geometries(
        [
            box(0, 0, 10, 10),
            box(20, 0, 30, 10),
            box(40, 0, 50, 10),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            box(1, 1, 2, 2),
            box(3, 3, 4, 4),
            box(21, 1, 22, 2),
        ],
        residency=Residency.DEVICE,
    )
    reasons: list[str] = []
    observed: dict[str, object] = {}
    original_bridge = overlay_module._overlay_device_to_host

    def _recording_bridge(value, *, reason: str, dtype=None):
        reasons.append(reason)
        return original_bridge(value, reason=reason, dtype=dtype)

    def _fake_grouped_difference(
        left_batch,
        right_batch,
        group_offsets,
        *,
        dispatch_mode,
        **_kwargs,
    ):
        observed["group_offsets"] = group_offsets
        observed["left_rows"] = left_batch.row_count
        observed["right_rows"] = right_batch.row_count
        return left_batch

    monkeypatch.setattr(overlay_module, "_overlay_device_to_host", _recording_bridge)
    monkeypatch.setattr(
        overlay_module,
        "_grouped_overlay_difference_owned",
        _fake_grouped_difference,
    )

    result = overlay_module._grouped_overlay_difference_capacity_owned(
        left,
        right,
        None,
        None,
        cp.asarray([0, 0, 2], dtype=cp.int32),
        cp.asarray([0, 1, 2], dtype=cp.int32),
        True,
        ExecutionMode.GPU,
    )

    assert result.row_count == 3
    assert observed["left_rows"] == 3
    assert observed["right_rows"] == 3
    assert hasattr(observed["group_offsets"], "__cuda_array_interface__")
    assert cp.asnumpy(observed["group_offsets"]).tolist() == [0, 2, 2, 3]
    assert not reasons


def test_grouped_overlay_difference_native_plan_failure_does_not_switch_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = from_shapely_geometries(
        [box(0, 0, 10, 10)],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [box(1, 1, 2, 2), box(3, 3, 4, 4)],
        residency=Residency.DEVICE,
    )
    dissolve_module = importlib.import_module("vibespatial.overlay.dissolve")

    def _raising_build(*args, **kwargs):
        raise RuntimeError("boom")

    def _declining_native_grouped_union(*_args, **_kwargs):
        return None

    def _fail_group_offsets_host(*_args, **_kwargs):
        raise AssertionError("native grouped decline should not export group offsets")

    monkeypatch.setattr(overlay_gpu_module, "_build_overlay_execution_plan", _raising_build)
    monkeypatch.setattr(
        dissolve_module,
        "execute_native_grouped_union",
        _declining_native_grouped_union,
    )
    monkeypatch.setattr(overlay_module, "_group_offsets_to_host", _fail_group_offsets_host)

    with pytest.raises(RuntimeError, match="boom"):
        overlay_module._grouped_overlay_difference_owned(
            left,
            right,
            np.asarray([0, 2], dtype=np.int64),
            dispatch_mode=ExecutionMode.GPU,
        )


def test_grouped_overlay_difference_plan_failure_does_not_retry_grouped_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.constructive import binary_constructive as constructive_module

    dissolve_module = importlib.import_module("vibespatial.overlay.dissolve")
    left = from_shapely_geometries(
        [box(0, 0, 10, 10), box(20, 0, 30, 10)],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            box(9, 1, 12, 2),
            box(29, 1, 32, 2),
            box(21, 9, 22, 12),
        ],
        residency=Residency.DEVICE,
    )
    unioned = from_shapely_geometries(
        [box(1, 1, 2, 2), box(21, 1, 24, 2)],
        residency=Residency.DEVICE,
    )
    sentinel = from_shapely_geometries(
        [box(0, 0, 9, 9), box(20, 0, 29, 9)],
        residency=Residency.DEVICE,
    )
    union_calls: list[dict[str, object]] = []
    difference_calls: list[dict[str, object]] = []

    def _raising_build(*_args, **_kwargs):
        raise RuntimeError("boom")

    def _fake_grouped_union(grouped, *, _geometries, method, owned):
        union_calls.append(
            {
                "grouped_is_device": grouped.is_device,
                "group_offsets": cp.asnumpy(grouped.group_offsets).tolist(),
                "group_ids": cp.asnumpy(grouped.group_ids).tolist(),
                "right_is_original": owned is right,
                "method": str(method),
            }
        )
        return SimpleNamespace(owned=unioned)

    def _fake_binary(op, left_arg, right_arg, **kwargs):
        difference_calls.append(
            {
                "op": op,
                "left_is_original": left_arg is left,
                "right_is_unioned": right_arg is unioned,
                "dispatch_mode": kwargs.get("dispatch_mode"),
            }
        )
        return sentinel

    monkeypatch.setattr(overlay_gpu_module, "_build_overlay_execution_plan", _raising_build)
    monkeypatch.setattr(dissolve_module, "execute_native_grouped_union", _fake_grouped_union)
    monkeypatch.setattr(constructive_module, "binary_constructive_owned", _fake_binary)
    monkeypatch.setattr(
        overlay_module,
        "_grouped_polygon_hole_difference_owned",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        overlay_module,
        "_group_offsets_to_host",
        lambda *_args, **_kwargs: pytest.fail(
            "native grouped-union difference should not export group offsets"
        ),
    )

    with pytest.raises(RuntimeError, match="boom"):
        overlay_module._grouped_overlay_difference_owned(
            left,
            right,
            np.asarray([0, 1, 3], dtype=np.int64),
            dispatch_mode=ExecutionMode.GPU,
        )

    assert union_calls == []
    assert difference_calls == []


def test_grouped_overlay_difference_preserves_indexed_grouped_union_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.api._native_grouped import NativeGrouped
    from vibespatial.constructive import binary_constructive as constructive_module

    dissolve_module = importlib.import_module("vibespatial.overlay.dissolve")
    left = from_shapely_geometries(
        [
            Polygon([(0, 0), (10, 0), (10, 8), (5, 10), (0, 8)]),
            Polygon([(20, 0), (30, 0), (30, 8), (25, 10), (20, 8)]),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [box(9, 1, 12, 2), box(29, 1, 32, 2)],
        residency=Residency.DEVICE,
    )
    unioned_base = from_shapely_geometries(
        [box(9, 1, 12, 2), box(29, 1, 32, 2)],
        residency=Residency.DEVICE,
    )
    unioned_view = OwnedGeometryArray._indexed_view(
        unioned_base,
        cp.asarray([0, 1], dtype=cp.int64),
    )
    assert unioned_view.is_indexed_view
    sentinel = from_shapely_geometries(
        [box(0, 0, 9, 9), box(20, 0, 29, 9)],
        residency=Residency.DEVICE,
    )
    difference_calls: list[dict[str, object]] = []

    def _fake_grouped_union(*_args, **_kwargs):
        return SimpleNamespace(owned=unioned_view)

    def _fake_binary(op, left_arg, right_arg, **kwargs):
        difference_calls.append(
            {
                "op": op,
                "left_is_original": left_arg is left,
                "right_is_indexed": right_arg.is_indexed_view,
                "dispatch_mode": kwargs.get("dispatch_mode"),
            }
        )
        return sentinel

    monkeypatch.setattr(dissolve_module, "execute_native_grouped_union", _fake_grouped_union)
    monkeypatch.setattr(constructive_module, "binary_constructive_owned", _fake_binary)

    grouped = NativeGrouped.from_sorted_offsets(
        cp.asarray([0, 1, 2], dtype=cp.int64),
        row_count=right.row_count,
        all_groups_observed=True,
        group_size_min=1,
        group_size_max=1,
    )
    result = overlay_module._native_grouped_union_difference_owned(
        left,
        right,
        grouped,
        dispatch_mode=ExecutionMode.GPU,
        stage="indexed-carrier-test",
    )

    assert result is not None
    assert result.residency is Residency.DEVICE
    assert result.row_count == sentinel.row_count
    assert difference_calls == [
        {
            "op": "difference",
            "left_is_original": True,
            "right_is_indexed": True,
            "dispatch_mode": ExecutionMode.GPU,
        }
    ]
    assert bool(
        np.all(
            shapely.equals_exact(
                np.asarray(result.to_shapely(), dtype=object),
                np.asarray(sentinel.to_shapely(), dtype=object),
                tolerance=0.0,
            )
        )
    )


def test_grouped_overlay_difference_prunes_left_covered_groups_before_union() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geoms = [
        box(0.0, 0.0, 2.0, 2.0),
        box(10.0, 0.0, 14.0, 4.0),
    ]
    right_geoms = [
        box(-1.0, -1.0, 3.0, 3.0),
        box(20.0, 20.0, 21.0, 21.0),
        box(12.0, 0.0, 16.0, 4.0),
        box(10.5, 3.0, 12.5, 5.0),
    ]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
    clear_dispatch_events()

    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 2, 4], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=2,
    )
    events = get_dispatch_events(clear=True)

    assert result.row_count == 2
    actual = result.to_shapely()
    expected = shapely.difference(
        left_geoms[1],
        shapely.union_all(np.asarray(right_geoms[2:], dtype=object)),
    )
    assert actual[0].is_empty
    assert shapely.symmetric_difference(actual[1], expected).area < 1.0e-8
    assert any(
        event.implementation == "grouped_overlay_difference_left_covered_prune_gpu"
        for event in events
    )
    assert not any(
        event.implementation == "grouped_overlay_difference_containment_union_gpu"
        for event in events
    )


def test_grouped_overlay_difference_cover_owner_keeps_group_capacity() -> None:
    source = Path("src/vibespatial/api/tools/overlay.py").read_text()
    function_start = source.index("def _grouped_overlay_difference_owned(")
    function_end = source.index("\ndef ", function_start + 1)
    function_source = source[function_start:function_end]
    owner_start = function_source.index("def _left_covered_group_mask()")
    owner_end = function_source.index("\n    if max_group_size", owner_start)
    owner_source = function_source[owner_start:owner_end]

    assert "binary_predicate_expressions(" in owner_source
    assert "return d_covered_mask, right_group_rows" in owner_source
    assert "covered-group plan admission" not in owner_source
    assert "copy_device_to_host" not in owner_source
    assert "covered_groups=device-resident" in owner_source
    assert "cp.flatnonzero(" not in owner_source
    assert "cp.unique(" not in owner_source
    assert "_grouped_overlay_difference_owned(" not in owner_source
    assert function_source.index("covered_group_ownership =") < function_source.index(
        "_grouped_polygon_hole_difference_owned("
    )
    assert "build_empty_polygon_rows_device(left_batch.row_count)" in function_source
    assert "d_claimed |= d_covered_groups" in function_source
    assert "device_select_owned_capacity_partitions(" in function_source


def test_grouped_overlay_difference_physicalizes_indexed_views_for_mutable_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_base_geoms = [
        box(-100.0, -100.0, -90.0, -90.0),
        box(0.0, 0.0, 10.0, 10.0),
    ]
    right_base_geoms = [
        box(-50.0, -50.0, -49.0, -49.0),
        box(8.0, 1.0, 12.0, 3.0),
        box(1.0, 8.0, 3.0, 12.0),
    ]
    left_base = from_shapely_geometries(left_base_geoms, residency=Residency.DEVICE)
    right_base = from_shapely_geometries(right_base_geoms, residency=Residency.DEVICE)
    left = OwnedGeometryArray._indexed_view(
        left_base,
        cp.asarray([1], dtype=cp.int64),
    )
    right = OwnedGeometryArray._indexed_view(
        right_base,
        cp.asarray([1, 2], dtype=cp.int64),
    )
    assert left.is_indexed_view
    assert right.is_indexed_view

    monkeypatch.setattr(
        overlay_module,
        "_grouped_rectangle_hole_difference_owned",
        lambda *_args, **_kwargs: None,
    )
    clear_dispatch_events()
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 2], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=2,
        _skip_direct_specializations=True,
    )
    events = get_dispatch_events(clear=True)

    expected = shapely.difference(
        left_base_geoms[1],
        shapely.union_all(np.asarray(right_base_geoms[1:], dtype=object)),
    )
    actual = result.to_shapely()[0]
    assert result.row_count == 1
    assert shapely.symmetric_difference(actual, expected).area < 1.0e-8
    assert any(
        event.implementation == "grouped_overlay_difference_topology_physicalization_gpu"
        for event in events
    )
    assert not any(
        event.implementation == "grouped_overlay_difference_indexed_physicalization_gpu"
        for event in events
    )
    assert any(event.implementation == "grouped_overlay_difference_gpu" for event in events)
    assert not any(
        event.implementation == "grouped_overlay_difference_containment_union_gpu"
        for event in events
    )


def test_grouped_overlay_difference_physicalized_duplicate_indexed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_base_geoms = [box(0.0, 0.0, 10.0, 10.0)]
    right_geoms = [
        box(8.0, 1.0, 12.0, 3.0),
        box(1.0, 8.0, 3.0, 12.0),
        box(-2.0, 1.0, 2.0, 3.0),
        box(4.0, -2.0, 6.0, 2.0),
    ]
    left_base = from_shapely_geometries(left_base_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
    left = OwnedGeometryArray._indexed_view(
        left_base,
        cp.asarray([0, 0], dtype=cp.int64),
    )

    assert left.is_indexed_view

    monkeypatch.setattr(
        overlay_module,
        "_grouped_rectangle_hole_difference_owned",
        lambda *_args, **_kwargs: None,
    )
    clear_dispatch_events()
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 2, 4], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=2,
    )
    events = get_dispatch_events(clear=True)

    expected = [
        shapely.difference(
            left_base_geoms[0],
            shapely.union_all(np.asarray(right_geoms[:2], dtype=object)),
        ),
        shapely.difference(
            left_base_geoms[0],
            shapely.union_all(np.asarray(right_geoms[2:], dtype=object)),
        ),
    ]
    actual = result.to_shapely()

    assert result.row_count == 2
    for actual_geom, expected_geom in zip(actual, expected, strict=True):
        assert shapely.symmetric_difference(actual_geom, expected_geom).area < 1.0e-8
    assert any(
        event.implementation == "grouped_overlay_difference_topology_physicalization_gpu"
        for event in events
    )
    assert not any(
        event.implementation == "grouped_overlay_difference_indexed_physicalization_gpu"
        for event in events
    )
    assert any(event.implementation == "grouped_overlay_difference_gpu" for event in events)


def test_grouped_overlay_difference_physicalized_duplicate_multipolygon_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geom = MultiPolygon([box(0.0, 0.0, 5.0, 5.0), box(6.0, 0.0, 10.0, 5.0)])
    right_geoms = [
        box(4.0, 1.0, 7.0, 3.0),
        box(8.0, 2.0, 12.0, 4.0),
        box(-1.0, 1.0, 2.0, 3.0),
        box(6.0, 4.0, 8.0, 7.0),
    ]
    left_base = from_shapely_geometries([left_geom], residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
    left = OwnedGeometryArray._indexed_view(
        left_base,
        cp.asarray([0, 0], dtype=cp.int64),
    )

    monkeypatch.setattr(
        overlay_module,
        "_grouped_rectangle_hole_difference_owned",
        lambda *_args, **_kwargs: None,
    )
    clear_dispatch_events()
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 2, 4], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=2,
    )
    events = get_dispatch_events(clear=True)

    expected = [
        shapely.difference(
            left_geom,
            shapely.union_all(np.asarray(right_geoms[:2], dtype=object)),
        ),
        shapely.difference(
            left_geom,
            shapely.union_all(np.asarray(right_geoms[2:], dtype=object)),
        ),
    ]
    actual = result.to_shapely()

    assert left.is_indexed_view
    assert result.row_count == 2
    for actual_geom, expected_geom in zip(actual, expected, strict=True):
        assert shapely.symmetric_difference(actual_geom, expected_geom).area < 1.0e-8
    assert any(
        event.implementation == "grouped_overlay_difference_topology_physicalization_gpu"
        for event in events
    )
    assert not any(
        event.implementation == "grouped_overlay_difference_indexed_physicalization_gpu"
        for event in events
    )
    assert any(event.implementation == "grouped_overlay_difference_gpu" for event in events)


def test_grouped_overlay_difference_uses_topology_for_nonrect_hole_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geom = box(0.0, 0.0, 10.0, 10.0)
    right_geoms = [
        Polygon([(1.0, 1.0), (5.0, 1.0), (3.0, 5.0), (1.0, 1.0)]),
        Polygon([(4.0, 4.0), (8.0, 4.0), (8.0, 8.0), (6.0, 7.0), (4.0, 4.0)]),
    ]
    left = from_shapely_geometries([left_geom], residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
    monkeypatch.setattr(
        overlay_module,
        "_grouped_rectangle_hole_difference_owned",
        lambda *_args, **_kwargs: None,
    )

    clear_dispatch_events()
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 2], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=2,
    )
    events = get_dispatch_events(clear=True)

    expected = shapely.difference(
        left_geom,
        shapely.union_all(np.asarray(right_geoms, dtype=object)),
    )
    assert result.row_count == 1
    assert shapely.symmetric_difference(result.to_shapely()[0], expected).area < 1.0e-8
    assert any(event.implementation == "grouped_overlay_difference_gpu" for event in events)


def test_grouped_overlay_difference_emits_overlapping_holes_in_one_topology_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geom = box(0.0, 0.0, 12.0, 12.0)
    right_geoms = [
        box(1.0, 1.0, 5.0, 5.0),
        box(4.0, 3.0, 9.0, 8.0),
    ]
    left = from_shapely_geometries([left_geom], residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
    monkeypatch.setattr(
        overlay_module,
        "_grouped_rectangle_hole_difference_owned",
        lambda *_args, **_kwargs: None,
    )

    clear_dispatch_events()
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 2], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=2,
    )
    events = get_dispatch_events(clear=True)

    expected = shapely.difference(
        left_geom,
        shapely.union_all(np.asarray(right_geoms, dtype=object)),
    )
    assert result.row_count == 1
    assert shapely.symmetric_difference(result.to_shapely()[0], expected).area < 1.0e-8
    assert any(event.implementation == "grouped_overlay_difference_gpu" for event in events)
    assert not any(
        event.implementation == "native_grouped_union_difference_physicalize_device_rows"
        for event in events
    )


def test_grouped_overlay_difference_emits_multi_hole_union_topology() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geom = box(0.0, 0.0, 30.0, 30.0)
    multi_hole = Polygon(
        [(2.0, 2.0), (22.0, 2.0), (22.0, 22.0), (2.0, 22.0), (2.0, 2.0)],
        holes=[
            [(5.0, 5.0), (9.0, 5.0), (9.0, 9.0), (5.0, 9.0), (5.0, 5.0)],
            [
                (14.0, 14.0),
                (19.0, 14.0),
                (19.0, 19.0),
                (14.0, 19.0),
                (14.0, 14.0),
            ],
        ],
    )
    right_geoms = [multi_hole, box(3.0, 3.0, 4.0, 4.0)]
    left = from_shapely_geometries([left_geom], residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 2], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=2,
    )
    events = get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    expected = shapely.difference(
        left_geom,
        shapely.union_all(np.asarray(right_geoms, dtype=object)),
    )
    actual = result.to_shapely()[0]
    assert actual.geom_type == "MultiPolygon"
    assert len(actual.geoms) == 3
    assert actual.is_valid
    assert shapely.symmetric_difference(actual, expected).area < 1.0e-8
    assert any(event.implementation == "grouped_overlay_difference_gpu" for event in events)
    assert not any(
        event.implementation == "native_grouped_union_difference_gpu" for event in events
    )
    assert "binary predicate tag-pairs host export" not in runtime_reasons


def test_grouped_overlay_difference_preserves_nested_union_components() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geom = box(0.0, 0.0, 30.0, 30.0)
    outer_donut = Polygon(
        [(2.0, 2.0), (22.0, 2.0), (22.0, 22.0), (2.0, 22.0), (2.0, 2.0)],
        holes=[
            [
                (5.0, 5.0),
                (19.0, 5.0),
                (19.0, 19.0),
                (5.0, 19.0),
                (5.0, 5.0),
            ],
        ],
    )
    nested_component = box(9.0, 9.0, 13.0, 13.0)
    right_geoms = [outer_donut, nested_component]
    left = from_shapely_geometries([left_geom], residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 2], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=2,
    )
    events = get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    expected = shapely.difference(
        left_geom,
        shapely.union_all(np.asarray(right_geoms, dtype=object)),
    )
    actual = result.to_shapely()[0]
    assert actual.geom_type == "MultiPolygon"
    assert len(actual.geoms) == 2
    assert sum(len(part.interiors) for part in actual.geoms) == 2
    assert actual.is_valid
    assert shapely.symmetric_difference(actual, expected).area < 1.0e-8
    assert any(event.implementation == "grouped_overlay_difference_gpu" for event in events)
    assert not any(
        event.implementation == "native_grouped_union_difference_gpu" for event in events
    )
    assert "binary predicate tag-pairs host export" not in runtime_reasons


def test_grouped_overlay_difference_emits_mixed_polygonal_union_rows() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geoms = [box(0.0, 0.0, 30.0, 30.0), box(40.0, 0.0, 70.0, 30.0)]
    multi_hole = Polygon(
        [
            (42.0, 2.0),
            (62.0, 2.0),
            (62.0, 22.0),
            (42.0, 22.0),
            (42.0, 2.0),
        ],
        holes=[
            [
                (45.0, 5.0),
                (49.0, 5.0),
                (49.0, 9.0),
                (45.0, 9.0),
                (45.0, 5.0),
            ],
            [
                (54.0, 14.0),
                (59.0, 14.0),
                (59.0, 19.0),
                (54.0, 19.0),
                (54.0, 14.0),
            ],
        ],
    )
    right_geoms = [
        box(2.0, 2.0, 12.0, 12.0),
        box(3.0, 3.0, 4.0, 4.0),
        multi_hole,
        box(43.0, 3.0, 44.0, 4.0),
    ]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

    clear_dispatch_events()
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 2, 4], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=2,
    )
    events = get_dispatch_events(clear=True)

    actual = result.to_shapely()
    expected = [
        shapely.difference(
            left_geoms[group],
            shapely.union_all(np.asarray(right_geoms[group * 2 : group * 2 + 2], dtype=object)),
        )
        for group in range(2)
    ]
    assert [geom.geom_type for geom in actual] == ["Polygon", "MultiPolygon"]
    assert all(geom.is_valid for geom in actual)
    assert all(
        shapely.symmetric_difference(observed, wanted).area < 1.0e-8
        for observed, wanted in zip(actual, expected, strict=True)
    )
    assert any(event.implementation == "grouped_overlay_difference_gpu" for event in events)


def test_grouped_overlay_difference_partitions_contained_and_crossing_groups() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geoms = [box(0.0, 0.0, 30.0, 30.0), box(40.0, 0.0, 60.0, 20.0)]
    contained_multi_hole = Polygon(
        [(2.0, 2.0), (22.0, 2.0), (22.0, 22.0), (2.0, 22.0), (2.0, 2.0)],
        holes=[
            [(5.0, 5.0), (9.0, 5.0), (9.0, 9.0), (5.0, 9.0), (5.0, 5.0)],
            [
                (14.0, 14.0),
                (19.0, 14.0),
                (19.0, 19.0),
                (14.0, 19.0),
                (14.0, 14.0),
            ],
        ],
    )
    right_geoms = [
        contained_multi_hole,
        box(3.0, 3.0, 4.0, 4.0),
        box(50.0, 5.0, 65.0, 15.0),
        box(55.0, 2.0, 68.0, 18.0),
    ]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

    clear_dispatch_events()
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 2, 4], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=2,
    )
    events = get_dispatch_events(clear=True)

    actual = result.to_shapely()
    expected = [
        shapely.difference(
            left_geoms[group],
            shapely.union_all(
                np.asarray(right_geoms[group * 2 : group * 2 + 2], dtype=object),
            ),
        )
        for group in range(2)
    ]
    assert all(geom.is_valid for geom in actual)
    assert all(
        shapely.symmetric_difference(observed, wanted).area < 1.0e-8
        for observed, wanted in zip(actual, expected, strict=True)
    )
    assert (
        getattr(result, "_device_scatter_implementation", None)
        == "device_capacity_partition_selection"
    )
    assert any(event.implementation == "grouped_overlay_difference_gpu" for event in events)
    assert not any(
        event.implementation == "native_grouped_union_difference_gpu" for event in events
    )


def test_grouped_overlay_difference_direct_holes_partition_exact_groups() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geoms = [box(0.0, 0.0, 20.0, 20.0), box(30.0, 0.0, 50.0, 20.0)]
    right_geoms = [
        box(2.0, 2.0, 6.0, 6.0),
        box(12.0, 12.0, 17.0, 17.0),
        box(32.0, 2.0, 38.0, 8.0),
        box(46.0, 5.0, 54.0, 15.0),
    ]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 2, 4], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
    )
    events = get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    actual = result.to_shapely()
    expected = [
        shapely.difference(
            left_geoms[group],
            shapely.union_all(
                np.asarray(right_geoms[group * 2 : group * 2 + 2], dtype=object),
            ),
        )
        for group in range(2)
    ]
    assert all(
        shapely.symmetric_difference(observed, wanted).area < 1.0e-8
        for observed, wanted in zip(actual, expected, strict=True)
    )
    assert (
        getattr(result, "_device_scatter_implementation", None)
        == "device_capacity_partition_selection"
    )
    assert any(
        event.implementation == "grouped_overlay_difference_rectangle_holes_gpu" for event in events
    )
    assert any(
        event.implementation == "grouped_overlay_difference_direct_capacity_partition_gpu"
        for event in events
    )
    assert any(event.implementation == "grouped_overlay_difference_gpu" for event in events)
    assert not any("admission scalar fence" in reason for reason in runtime_reasons)
    assert not any("group-size metadata" in reason for reason in runtime_reasons)


def test_grouped_overlay_difference_rectangle_strips_use_one_exact_topology_plan() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.geometry.owned import seed_all_validity_cache
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geoms = [box(0.0, 0.0, 10.0, 10.0), box(20.0, 0.0, 30.0, 10.0)]
    right_geoms = [
        box(-2.0, 2.0, 4.0, 6.0),
        box(3.0, 2.0, 7.0, 6.0),
        box(6.0, 2.0, 12.0, 6.0),
        box(18.0, 2.0, 24.0, 6.0),
        box(23.0, 2.0, 27.0, 6.0),
    ]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(left)
    seed_all_validity_cache(right)

    clear_dispatch_events()
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 3, 5], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=3,
    )
    events = get_dispatch_events(clear=True)

    actual = np.asarray(result.to_shapely(), dtype=object)
    expected = [
        shapely.difference(left_geoms[0], shapely.union_all(right_geoms[:3])),
        shapely.difference(left_geoms[1], shapely.union_all(right_geoms[3:])),
    ]
    assert all(geom.is_valid for geom in actual)
    assert all(
        shapely.symmetric_difference(observed, wanted).area < 1.0e-8
        for observed, wanted in zip(actual, expected, strict=True)
    )
    assert any(event.implementation == "grouped_overlay_difference_gpu" for event in events)
    assert not any(
        event.implementation == "native_grouped_union_difference_gpu" for event in events
    )


def test_rectangle_strip_difference_keeps_row_capacity_exact_topology() -> None:
    source = Path("src/vibespatial/constructive/binary_constructive.py").read_text()
    start = source.index("def _row_aligned_rectangle_partition_difference_gpu(")
    end = source.index("\ndef ", start + 1)
    function_source = source[start:end]

    assert "topology_capacity={row_count}" in function_source
    assert "_dispatch_polygon_difference_overlay_exact_batch_gpu(" in function_source
    assert "device_mask_owned_capacity(clipped_right, d_overlap)" in function_source
    assert "device_select_owned_capacity_partitions(" in function_source
    assert "cp.flatnonzero(" not in function_source
    assert "device_concat_owned_scatter(" not in function_source


def test_grouped_overlay_difference_emits_disjoint_nonrect_holes_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geom = Polygon(
        [
            (0.0, 0.0),
            (12.0, 0.0),
            (12.0, 12.0),
            (0.0, 12.0),
            (0.0, 0.0),
        ],
    )
    right_geoms = [
        Polygon([(1.0, 1.0), (4.0, 1.0), (2.0, 4.0), (1.0, 1.0)]),
        Polygon(
            [
                (7.0, 7.0),
                (10.0, 7.0),
                (11.0, 9.0),
                (8.5, 10.0),
                (7.0, 7.0),
            ],
        ),
    ]
    left = from_shapely_geometries([left_geom], residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
    monkeypatch.setattr(
        overlay_module,
        "_grouped_rectangle_hole_difference_owned",
        lambda *_args, **_kwargs: None,
    )

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 2], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=2,
    )
    events = get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    expected = shapely.difference(
        left_geom,
        shapely.union_all(np.asarray(right_geoms, dtype=object)),
    )
    actual = result.to_shapely()[0]
    assert result.row_count == 1
    assert shapely.symmetric_difference(actual, expected).area < 1.0e-8
    assert any(
        event.implementation == "grouped_overlay_difference_polygon_holes_gpu" for event in events
    )
    assert not any(
        event.implementation == "grouped_overlay_difference_containment_union_gpu"
        for event in events
    )
    assert "binary predicate tag-pairs host export" not in runtime_reasons
    assert "overlay grouped polygon-hole ring allocation fence" not in runtime_reasons
    assert "overlay grouped polygon-hole coordinate allocation fence" not in runtime_reasons


def test_grouped_overlay_difference_emits_holed_right_as_multipolygon_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geom = box(0.0, 0.0, 20.0, 20.0)
    right_geoms = [
        Polygon(
            [
                (2.0, 2.0),
                (8.0, 2.0),
                (8.0, 8.0),
                (2.0, 8.0),
                (2.0, 2.0),
            ],
            holes=[
                [
                    (4.0, 4.0),
                    (6.0, 4.0),
                    (6.0, 6.0),
                    (4.0, 6.0),
                    (4.0, 4.0),
                ],
            ],
        ),
        Polygon(
            [
                (11.0, 11.0),
                (17.0, 11.0),
                (17.0, 17.0),
                (11.0, 17.0),
                (11.0, 11.0),
            ],
            holes=[
                [
                    (13.0, 13.0),
                    (15.0, 13.0),
                    (15.0, 15.0),
                    (13.0, 15.0),
                    (13.0, 13.0),
                ],
            ],
        ),
    ]
    left = from_shapely_geometries([left_geom], residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
    monkeypatch.setattr(
        overlay_module,
        "_grouped_rectangle_hole_difference_owned",
        lambda *_args, **_kwargs: None,
    )

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 2], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=2,
    )
    events = get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    expected = shapely.difference(
        left_geom,
        shapely.union_all(np.asarray(right_geoms, dtype=object)),
    )
    actual = result.to_shapely()[0]
    assert result.row_count == 1
    assert actual.geom_type == "MultiPolygon"
    assert shapely.symmetric_difference(actual, expected).area < 1.0e-8
    assert any(
        event.implementation == "grouped_overlay_difference_polygon_donuts_gpu" for event in events
    )
    assert not any(
        event.implementation == "grouped_overlay_difference_containment_union_gpu"
        for event in events
    )
    assert "binary predicate tag-pairs host export" not in runtime_reasons
    assert "overlay grouped polygon-hole ring allocation fence" not in runtime_reasons
    assert "overlay grouped polygon-hole coordinate allocation fence" not in runtime_reasons


def test_grouped_overlay_difference_emits_indexed_direct_holes_without_sizing_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geom = Polygon(
        [
            (0.0, 0.0),
            (12.0, 0.0),
            (12.0, 12.0),
            (0.0, 12.0),
            (0.0, 0.0),
        ],
    )
    right_geoms = [
        Polygon([(1.0, 1.0), (3.0, 1.0), (2.0, 3.0), (1.0, 1.0)]),
        Polygon([(5.0, 1.0), (7.0, 1.0), (6.0, 3.0), (5.0, 1.0)]),
        Polygon([(1.0, 7.0), (3.0, 7.0), (2.0, 9.0), (1.0, 7.0)]),
        Polygon([(5.0, 7.0), (7.0, 7.0), (6.0, 9.0), (5.0, 7.0)]),
    ]
    left_base = from_shapely_geometries([left_geom], residency=Residency.DEVICE)
    left = OwnedGeometryArray._indexed_view(
        left_base,
        cp.asarray([0, 0], dtype=cp.int64),
    )
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
    monkeypatch.setattr(
        overlay_module,
        "_grouped_rectangle_hole_difference_owned",
        lambda *_args, **_kwargs: None,
    )

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 2, 4], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=2,
    )
    events = get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    expected = [
        shapely.difference(
            left_geom,
            shapely.union_all(np.asarray(right_geoms[:2], dtype=object)),
        ),
        shapely.difference(
            left_geom,
            shapely.union_all(np.asarray(right_geoms[2:], dtype=object)),
        ),
    ]
    actual = result.to_shapely()
    assert left.is_indexed_view
    assert result.row_count == 2
    for actual_geom, expected_geom in zip(actual, expected, strict=True):
        assert shapely.symmetric_difference(actual_geom, expected_geom).area < 1.0e-8
    assert any(
        event.implementation == "grouped_overlay_difference_polygon_holes_gpu"
        and "sizing=structural" in event.detail
        for event in events
    )
    assert not any(
        event.implementation == "grouped_overlay_difference_containment_union_gpu"
        for event in events
    )
    assert "binary predicate tag-pairs host export" not in runtime_reasons
    assert "overlay grouped polygon-hole ring allocation fence" not in runtime_reasons
    assert "overlay grouped polygon-hole coordinate allocation fence" not in runtime_reasons


def test_grouped_overlay_difference_emits_variable_indexed_direct_holes_without_sizing_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    ignored_left = Polygon(
        [
            (-20.0, -20.0),
            (-12.0, -20.0),
            (-12.0, -12.0),
            (-20.0, -12.0),
            (-20.0, -20.0),
        ],
    )
    left_geom = Polygon(
        [
            (0.0, 0.0),
            (14.0, 0.0),
            (14.0, 14.0),
            (0.0, 14.0),
            (0.0, 0.0),
        ],
        holes=[
            [
                (5.0, 5.0),
                (7.0, 5.0),
                (7.0, 7.0),
                (5.0, 7.0),
                (5.0, 5.0),
            ],
        ],
    )
    right_geoms = [
        Polygon([(1.0, 1.0), (4.0, 1.0), (2.0, 4.0), (1.0, 1.0)]),
        Polygon([(9.0, 1.0), (12.0, 1.0), (11.0, 4.0), (9.0, 1.0)]),
        Polygon([(1.0, 9.0), (4.0, 9.0), (2.0, 12.0), (1.0, 9.0)]),
        Polygon([(9.0, 9.0), (12.0, 9.0), (11.0, 12.0), (9.0, 9.0)]),
    ]
    left_base = from_shapely_geometries(
        [ignored_left, left_geom],
        residency=Residency.DEVICE,
    )
    left = OwnedGeometryArray._indexed_view(
        left_base,
        cp.asarray([1, 1], dtype=cp.int64),
    )
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
    monkeypatch.setattr(
        overlay_module,
        "_grouped_rectangle_hole_difference_owned",
        lambda *_args, **_kwargs: None,
    )

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 2, 4], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=2,
    )
    events = get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    expected = [
        shapely.difference(
            left_geom,
            shapely.union_all(np.asarray(right_geoms[:2], dtype=object)),
        ),
        shapely.difference(
            left_geom,
            shapely.union_all(np.asarray(right_geoms[2:], dtype=object)),
        ),
    ]
    actual = result.to_shapely()
    assert left.is_indexed_view
    assert result.row_count == 2
    for actual_geom, expected_geom in zip(actual, expected, strict=True):
        assert shapely.symmetric_difference(actual_geom, expected_geom).area < 1.0e-8
    assert any(
        event.implementation == "grouped_overlay_difference_polygon_holes_gpu"
        and "sizing=capacity" in event.detail
        for event in events
    )
    assert not any(
        event.implementation == "grouped_overlay_difference_indexed_physicalization_gpu"
        for event in events
    )
    assert "binary predicate tag-pairs host export" not in runtime_reasons
    assert "overlay grouped polygon-hole ring allocation fence" not in runtime_reasons
    assert "overlay grouped polygon-hole coordinate allocation fence" not in runtime_reasons


def test_grouped_overlay_difference_emits_holes_for_holed_left_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geom = Polygon(
        [
            (0.0, 0.0),
            (14.0, 0.0),
            (14.0, 14.0),
            (0.0, 14.0),
            (0.0, 0.0),
        ],
        holes=[
            [
                (5.0, 5.0),
                (7.0, 5.0),
                (7.0, 7.0),
                (5.0, 7.0),
                (5.0, 5.0),
            ],
        ],
    )
    right_geoms = [
        Polygon([(1.0, 1.0), (4.0, 1.0), (2.0, 4.0), (1.0, 1.0)]),
        Polygon(
            [
                (9.0, 9.0),
                (12.0, 9.0),
                (12.0, 11.0),
                (10.0, 12.0),
                (9.0, 9.0),
            ],
        ),
    ]
    left = from_shapely_geometries([left_geom], residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
    monkeypatch.setattr(
        overlay_module,
        "_grouped_rectangle_hole_difference_owned",
        lambda *_args, **_kwargs: None,
    )

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 2], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
        _all_groups_observed=True,
        _group_size_min=2,
        _group_size_max=2,
    )
    events = get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    expected = shapely.difference(
        left_geom,
        shapely.union_all(np.asarray(right_geoms, dtype=object)),
    )
    actual = result.to_shapely()[0]
    assert result.row_count == 1
    assert shapely.symmetric_difference(actual, expected).area < 1.0e-8
    assert any(
        event.implementation == "grouped_overlay_difference_polygon_holes_gpu" for event in events
    )
    assert not any(
        event.implementation == "grouped_overlay_difference_containment_union_gpu"
        for event in events
    )
    assert "binary predicate tag-pairs host export" not in runtime_reasons
    assert "overlay grouped polygon-hole ring allocation fence" not in runtime_reasons
    assert "overlay grouped polygon-hole coordinate allocation fence" not in runtime_reasons


def test_overlay_difference_uses_grouped_overlay_plan_for_grouped_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (10, 0), (10, 6), (0, 6), (0, 0)]),
                Polygon([(12, 0), (22, 0), (22, 6), (12, 6), (12, 0)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2, 3, 4]},
        geometry=GeoSeries(
            [
                Polygon([(2, 0), (4, 0), (4, 6), (2, 6), (2, 0)]),
                Polygon([(6, 0), (8, 0), (8, 6), (6, 6), (6, 0)]),
                Polygon([(14, 0), (16, 0), (16, 6), (14, 6), (14, 0)]),
                Polygon([(18, 0), (20, 0), (20, 6), (18, 6), (18, 0)]),
            ]
        ),
    )

    from vibespatial.constructive import binary_constructive as constructive_module

    build_calls = 0
    materialize_calls = 0
    difference_calls = 0
    original_build = overlay_gpu_module._build_overlay_execution_plan
    original_materialize = overlay_gpu_module._materialize_overlay_execution_plan
    original = constructive_module.binary_constructive_owned

    def _wrapped_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return original_build(*args, **kwargs)

    def _wrapped_materialize(*args, **kwargs):
        nonlocal materialize_calls
        if kwargs.get("operation") == "difference":
            materialize_calls += 1
        return original_materialize(*args, **kwargs)

    def _wrapped_binary_constructive_owned(*args, **kwargs):
        nonlocal difference_calls
        if args and args[0] == "difference":
            difference_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        overlay_gpu_module,
        "_build_overlay_execution_plan",
        _wrapped_build,
    )
    monkeypatch.setattr(
        overlay_gpu_module,
        "_materialize_overlay_execution_plan",
        _wrapped_materialize,
    )
    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        _wrapped_binary_constructive_owned,
    )

    with strict_native_environment():
        result = overlay(left, right, how="difference")

    assert len(result) == 2
    assert build_calls == 1
    assert materialize_calls == 1
    assert difference_calls == 0


def test_overlay_difference_matches_union_for_overlapping_gt2_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1]},
        geometry=GeoSeries([Polygon([(0, 0), (12, 0), (12, 8), (0, 8), (0, 0)])]),
    )
    right = GeoDataFrame(
        {"col2": [1, 2, 3]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (6, 1), (6, 7), (1, 7), (1, 1)]),
                Polygon([(4, 1), (9, 1), (9, 7), (4, 7), (4, 1)]),
                Polygon([(7, 1), (11, 1), (11, 7), (7, 7), (7, 1)]),
            ]
        ),
    )

    from vibespatial.constructive import binary_constructive as constructive_module
    from vibespatial.kernels.constructive import segmented_union as segmented_union_module

    original = segmented_union_module.segmented_union_all
    original_build = overlay_gpu_module._build_overlay_execution_plan
    original_materialize = overlay_gpu_module._materialize_overlay_execution_plan
    original_binary_constructive_owned = constructive_module.binary_constructive_owned
    grouped_union_calls = 0
    grouped_plan_calls = 0
    grouped_materialize_calls = 0
    difference_calls = 0

    def _counted_segmented_union_all(*args, **kwargs):
        nonlocal grouped_union_calls
        grouped_union_calls += 1
        return original(*args, **kwargs)

    def _counted_build(*args, **kwargs):
        nonlocal grouped_plan_calls
        grouped_plan_calls += 1
        return original_build(*args, **kwargs)

    def _counted_materialize(*args, **kwargs):
        nonlocal grouped_materialize_calls
        if kwargs.get("operation") == "difference":
            grouped_materialize_calls += 1
        return original_materialize(*args, **kwargs)

    def _guarded_binary_constructive_owned(*args, **kwargs):
        nonlocal difference_calls
        if args and args[0] == "difference":
            difference_calls += 1
        return original_binary_constructive_owned(*args, **kwargs)

    monkeypatch.setattr(
        segmented_union_module,
        "segmented_union_all",
        _counted_segmented_union_all,
    )
    monkeypatch.setattr(
        overlay_gpu_module,
        "_build_overlay_execution_plan",
        _counted_build,
    )
    monkeypatch.setattr(
        overlay_gpu_module,
        "_materialize_overlay_execution_plan",
        _counted_materialize,
    )
    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        _guarded_binary_constructive_owned,
    )

    with strict_native_environment():
        vibespatial.clear_dispatch_events()
        result = overlay(left, right, how="difference")
        vibespatial.get_dispatch_events(clear=True)

    expected = shapely.difference(
        left.geometry.iloc[0],
        shapely.union_all(np.asarray(right.geometry, dtype=object)),
    )
    assert len(result) == 1
    assert grouped_union_calls == 0
    assert grouped_plan_calls == 1
    assert grouped_materialize_calls == 1
    assert difference_calls == 0
    assert shapely.symmetric_difference(result.geometry.iloc[0], expected).area < 1e-8


def test_overlay_difference_grouped_plan_avoids_brittle_same_row_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (10, 0), (10, 6), (0, 6), (0, 0)]),
                Polygon([(12, 0), (22, 0), (22, 6), (12, 6), (12, 0)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2, 3, 4]},
        geometry=GeoSeries(
            [
                Polygon([(2, 0), (4, 0), (4, 6), (2, 6), (2, 0)]),
                Polygon([(6, 0), (8, 0), (8, 6), (6, 6), (6, 0)]),
                Polygon([(14, 0), (16, 0), (16, 6), (14, 6), (14, 0)]),
                Polygon([(18, 0), (20, 0), (20, 6), (18, 6), (18, 0)]),
            ]
        ),
    )

    def _should_not_run(*args, **kwargs):
        raise AssertionError("grouped overlay plan should not depend on same-row warp candidates")

    monkeypatch.setattr(
        segment_primitives_module,
        "_generate_candidates_gpu_same_row_warp",
        _should_not_run,
    )

    expected = [
        shapely.difference(
            left.geometry.iloc[0],
            shapely.union_all(np.asarray(right.geometry.iloc[:2], dtype=object)),
        ),
        shapely.difference(
            left.geometry.iloc[1],
            shapely.union_all(np.asarray(right.geometry.iloc[2:], dtype=object)),
        ),
    ]

    with strict_native_environment():
        vibespatial.clear_dispatch_events()
        result = overlay(left, right, how="difference")
        events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 2
    assert any(
        event.implementation in {"grouped_overlay_difference_gpu", "grouped_union_difference_gpu"}
        and event.selected is ExecutionMode.GPU
        for event in events
    )
    assert not any(
        event.implementation == "grouped_overlay_difference_plan_build_failed_gpu"
        for event in events
    )
    for got, want in zip(np.asarray(result.geometry, dtype=object), expected, strict=True):
        assert shapely.symmetric_difference(got, want).area < 1e-8


def test_row_isolated_intersection_uses_same_row_candidate_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.runtime.hotpath_trace import (
        reset_hotpath_trace,
        summarize_hotpath_trace,
    )

    left = from_shapely_geometries(
        [
            box(0, 0, 3, 3),
            box(10, 0, 13, 3),
            box(20, 0, 23, 3),
        ]
    )
    right = from_shapely_geometries(
        [
            box(1, 1, 4, 4),
            box(11, 1, 14, 4),
            box(21, 1, 24, 4),
        ]
    )

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()
    result = overlay_gpu_module._overlay_owned(
        left,
        right,
        operation="intersection",
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
    )

    expected = shapely.intersection(
        np.asarray(left.to_shapely(), dtype=object),
        np.asarray(right.to_shapely(), dtype=object),
    )
    actual = np.asarray(result.to_shapely(), dtype=object)

    assert result.row_count == left.row_count
    assert all(
        got.normalize().equals_exact(want.normalize(), tolerance=1e-9)
        for got, want in zip(actual, expected, strict=True)
    )

    summary = {entry["name"]: entry["calls"] for entry in summarize_hotpath_trace()}
    assert summary.get("segment.candidates.same_row_fast_path") == 1
    assert "segment.candidates.binary_search" not in summary


def test_grouped_overlay_difference_forces_gpu_segment_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (10, 0), (10, 6), (0, 6), (0, 0)]),
                Polygon([(12, 0), (22, 0), (22, 6), (12, 6), (12, 0)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2, 3, 4]},
        geometry=GeoSeries(
            [
                Polygon([(2, 0), (4, 0), (4, 6), (2, 6), (2, 0)]),
                Polygon([(6, 0), (8, 0), (8, 6), (6, 6), (6, 0)]),
                Polygon([(14, 0), (16, 0), (16, 6), (14, 6), (14, 0)]),
                Polygon([(18, 0), (20, 0), (20, 6), (18, 6), (18, 0)]),
            ]
        ),
    )

    original = overlay_split_module.classify_segment_intersections
    seen_dispatch_modes: list[ExecutionMode] = []

    def _wrapped_classify(*args, **kwargs):
        if kwargs.get("_require_same_row"):
            seen_dispatch_modes.append(kwargs["dispatch_mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        overlay_split_module,
        "classify_segment_intersections",
        _wrapped_classify,
    )

    with strict_native_environment():
        result = overlay(left, right, how="difference")

    assert len(result) == 2
    assert seen_dispatch_modes
    assert all(mode is ExecutionMode.GPU for mode in seen_dispatch_modes)


def test_overlay_difference_redevelopment_like_followup_overlay_stays_strict_native(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from _data import setup_fixtures

    monkeypatch.setenv("VSBENCH_SCALE", "10k")
    fixtures = setup_fixtures(tmp_path)

    parcels = vibespatial.read_parquet(fixtures["parcels"])
    zones = vibespatial.read_parquet(fixtures["zones"])
    exclusions = vibespatial.read_parquet(fixtures["exclusion_zones"])
    transit = read_file(fixtures["transit"])

    bounds = parcels.total_bounds
    dx = (bounds[2] - bounds[0]) * 0.15
    dy = (bounds[3] - bounds[1]) * 0.15
    clip_box = box(bounds[0] + dx, bounds[1] + dy, bounds[2] - dx, bounds[3] - dy)

    with strict_native_environment():
        study_parcels = vibespatial.clip(parcels, clip_box)
        study_parcels = study_parcels[
            study_parcels.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ].copy()
        developable = overlay(study_parcels, exclusions, how="difference")
        _assert_owned_row_mapping_valid(developable.geometry)

        transit_buffers = transit.copy()
        transit_buffers["geometry"] = transit_buffers.geometry.buffer(150.0)
        near_transit = vibespatial.sjoin(
            developable,
            transit_buffers[["station_id", "geometry"]],
            predicate="intersects",
        )
        candidate_rows = near_transit.index.unique()
        candidates = (
            developable.loc[candidate_rows].copy()
            if len(candidate_rows) > 0
            else developable.iloc[:0].copy()
        )
        _assert_owned_row_mapping_valid(candidates.geometry)
        zoned = overlay(candidates, zones[["zone_type", "geometry"]], how="intersection")

    assert zoned is not None
    assert zoned.geometry.is_valid.all()

    zoned["zone_group"] = zoned["zone_type"].astype(str)
    dissolved = zoned.dissolve(by="zone_group").reset_index()
    assert dissolved.geometry.is_valid.all()


@pytest.mark.parametrize(
    ("scale", "expected_rows"),
    [
        ("1000", 4),
        ("10000", 44),
    ],
)
def test_overlay_intersection_accessibility_redevelopment_fixture_matches_pairwise_oracle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scale: str,
    expected_rows: int,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from _data import setup_fixtures

    polygonal_types = ["Polygon", "MultiPolygon"]
    max_nearest_distance_m = 1_800.0
    transit_buffer_m = 900.0

    monkeypatch.setenv("VSBENCH_SCALE", scale)
    fixtures = setup_fixtures(tmp_path)

    buildings = read_file(fixtures["access_buildings"])
    parcels = vibespatial.read_parquet(fixtures["access_parcels"])
    transit = read_file(fixtures["access_transit"])
    exclusions = vibespatial.read_parquet(fixtures["access_exclusions"])
    admin = read_file(fixtures["access_admin_boundary"])

    utm_crs = buildings.geometry.estimate_utm_crs()
    buildings = buildings.to_crs(utm_crs)
    parcels = parcels.to_crs(utm_crs)
    transit = transit.to_crs(utm_crs)
    exclusions = exclusions.to_crs(utm_crs)
    admin = admin.to_crs(utm_crs)

    buildings = geopandas.clip(buildings, admin)
    parcels = geopandas.clip(parcels, admin)
    buildings = buildings[buildings.geometry.geom_type.isin(polygonal_types)].copy()
    parcels = parcels[parcels.geometry.geom_type.isin(polygonal_types)].copy()

    building_points = buildings[["building_id", "geometry"]].copy()
    building_points["geometry"] = buildings.geometry.centroid
    centroid_oracle = np.asarray(
        shapely.centroid(np.asarray(buildings.geometry.array, dtype=object)),
        dtype=object,
    )
    centroid_distance = np.asarray(
        shapely.distance(
            np.asarray(building_points.geometry.array, dtype=object),
            centroid_oracle,
        ),
        dtype=np.float64,
    )
    assert np.all(centroid_distance <= 1.0e-9)

    nearest = building_points.sjoin_nearest(
        transit[["station_id", "geometry"]],
        how="inner",
        max_distance=max_nearest_distance_m,
        distance_col="station_distance_m",
    )
    nearest = nearest.sort_values(
        ["building_id", "station_distance_m", "station_id"]
    ).drop_duplicates("building_id")

    nearby_building_ids = nearest.loc[
        nearest["station_distance_m"] <= max_nearest_distance_m,
        "building_id",
    ].drop_duplicates()
    nearby_buildings = buildings[buildings["building_id"].isin(nearby_building_ids)].copy()

    transit_buffers = transit.copy()
    transit_buffers["geometry"] = transit_buffers.geometry.buffer(transit_buffer_m)

    developable = overlay(parcels, exclusions, how="difference")
    developable = developable[developable.geometry.geom_type.isin(polygonal_types)].copy()

    served = geopandas.sjoin(
        developable,
        transit_buffers[["station_id", "geometry"]],
        predicate="intersects",
    )
    served_rows = served.index.unique()
    served_parcels = (
        developable.loc[served_rows].copy() if len(served_rows) > 0 else developable.iloc[:0].copy()
    )

    left = served_parcels[["parcel_id", "geometry"]]
    right = nearby_buildings[["building_id", "geometry"]]

    vibespatial.clear_fallback_events()
    actual = overlay(left, right, how="intersection")
    fallback_events = vibespatial.get_fallback_events(clear=True)
    _assert_all_geometry_coordinates_finite(actual.geometry)
    assert not any(
        event.surface == "geopandas.array.make_valid" and event.selected is ExecutionMode.CPU
        for event in fallback_events
    )

    oracle_tree = shapely.STRtree(np.asarray(right.geometry.array, dtype=object))
    idx1, idx2 = oracle_tree.query(
        np.asarray(left.geometry.array, dtype=object),
        predicate="intersects",
    )
    idx1 = np.asarray(idx1, dtype=np.intp)
    idx2 = np.asarray(idx2, dtype=np.intp)

    pair_left = left.iloc[idx1].reset_index(drop=True)
    pair_right = right.iloc[idx2].reset_index(drop=True)
    exact_values = np.asarray(
        shapely.intersection(
            np.asarray(pair_left.geometry.array, dtype=object),
            np.asarray(pair_right.geometry.array, dtype=object),
        ),
        dtype=object,
    )
    exact_area = np.asarray(shapely.area(exact_values), dtype=np.float64)
    polygon_keep_mask = np.array(
        [
            geom is not None
            and not shapely.is_empty(geom)
            and geom.geom_type in polygonal_types
            and np.isfinite(area)
            and float(area) > 0.0
            for geom, area in zip(exact_values, exact_area, strict=True)
        ],
        dtype=bool,
    )
    keep_mask = polygon_keep_mask.copy()
    if polygon_keep_mask.any():
        exact_points = np.asarray(
            shapely.point_on_surface(exact_values[polygon_keep_mask]),
            dtype=object,
        )
        exact_left = np.asarray(pair_left.geometry.array, dtype=object)[polygon_keep_mask]
        exact_right = np.asarray(pair_right.geometry.array, dtype=object)[polygon_keep_mask]
        keep_mask[np.flatnonzero(polygon_keep_mask)] &= np.asarray(
            shapely.contains(exact_left, exact_points)
            & shapely.contains(exact_right, exact_points),
            dtype=bool,
        )

    expected = (
        GeoDataFrame(
            {
                "parcel_id": pair_left.loc[keep_mask, "parcel_id"].to_numpy(),
                "building_id": pair_right.loc[keep_mask, "building_id"].to_numpy(),
                "geometry": exact_values[keep_mask],
            },
            geometry="geometry",
            crs=left.crs,
        )
        .sort_values(["parcel_id", "building_id"])
        .reset_index(drop=True)
    )
    actual = actual.sort_values(["parcel_id", "building_id"]).reset_index(drop=True)

    assert len(actual) == expected_rows
    assert len(actual) == len(expected)
    assert actual[["parcel_id", "building_id"]].equals(expected[["parcel_id", "building_id"]])
    actual_area = np.asarray(shapely.area(actual.geometry.array), dtype=np.float64)
    assert np.all(np.isfinite(actual_area) & (actual_area > 0.0))


def test_binary_constructive_intersection_stays_strict_native_for_multipolygon_polygon_batch() -> (
    None
):
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive.binary_constructive import binary_constructive_owned

    left = from_shapely_geometries(
        [
            shapely.MultiPolygon(
                [
                    box(0, 0, 3, 3),
                    box(5, 0, 8, 3),
                ]
            ),
            shapely.MultiPolygon(
                [
                    box(10, 0, 14, 4),
                    box(12, 6, 16, 10),
                ]
            ),
        ]
    )
    right = from_shapely_geometries(
        [
            box(1, 1, 6, 2.5),
            box(11, 1, 15, 8),
        ]
    )

    with strict_native_environment():
        result = binary_constructive_owned(
            "intersection",
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
        )

    got = result.to_shapely()
    expected = shapely.intersection(
        np.asarray(left.to_shapely(), dtype=object),
        np.asarray(right.to_shapely(), dtype=object),
    ).tolist()
    assert len(got) == len(expected) == 2
    for actual, oracle in zip(got, expected, strict=True):
        if shapely.is_empty(oracle):
            assert actual is None or shapely.is_empty(actual)
            continue
        assert actual is not None
        assert shapely.normalize(actual).equals_exact(
            shapely.normalize(oracle),
            tolerance=1e-9,
        )


def test_binary_constructive_difference_skips_mixed_dispatch_for_polygonal_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from shapely.geometry import MultiPolygon

    from vibespatial.constructive import binary_constructive as constructive_module
    from vibespatial.geometry.owned import from_shapely_geometries

    left = from_shapely_geometries(
        [
            Polygon([(0, 0), (6, 0), (6, 4), (0, 4), (0, 0)]),
            MultiPolygon(
                [
                    Polygon([(10, 0), (13, 0), (13, 4), (10, 4), (10, 0)]),
                    Polygon([(14, 0), (17, 0), (17, 4), (14, 4), (14, 0)]),
                ]
            ),
        ]
    )
    right = from_shapely_geometries(
        [
            Polygon([(2, 0), (4, 0), (4, 4), (2, 4), (2, 0)]),
            Polygon([(11, 1), (16, 1), (16, 3), (11, 3), (11, 1)]),
        ]
    )

    mixed_dispatch_called = False
    original_mixed_dispatch = constructive_module._dispatch_mixed_binary_constructive_gpu

    def _wrapped_mixed_dispatch(*args, **kwargs):
        nonlocal mixed_dispatch_called
        mixed_dispatch_called = True
        return original_mixed_dispatch(*args, **kwargs)

    monkeypatch.setattr(
        constructive_module,
        "_dispatch_mixed_binary_constructive_gpu",
        _wrapped_mixed_dispatch,
    )

    with strict_native_environment():
        result = constructive_module.binary_constructive_owned(
            "difference",
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
        )

    assert result.row_count == 2
    assert not mixed_dispatch_called


def test_binary_constructive_intersection_tiny_mixed_polygonal_batch_prefers_rowwise_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from shapely.geometry import MultiPolygon

    from vibespatial.constructive import binary_constructive as constructive_module
    from vibespatial.geometry.owned import from_shapely_geometries

    left = from_shapely_geometries(
        [
            Polygon([(0, 0), (6, 0), (6, 4), (0, 4), (0, 0)]),
            MultiPolygon(
                [
                    Polygon([(10, 0), (13, 0), (13, 4), (10, 4), (10, 0)]),
                    Polygon([(14, 0), (17, 0), (17, 4), (14, 4), (14, 0)]),
                ]
            ),
        ]
    )
    right = from_shapely_geometries(
        [
            Polygon([(2, 0), (4, 0), (4, 4), (2, 4), (2, 0)]),
            Polygon([(11, 1), (16, 1), (16, 3), (11, 3), (11, 1)]),
        ]
    )

    rowwise_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_exact_batch_gpu",
        lambda left_arg, right_arg, **kwargs: (
            rowwise_calls.append((left_arg.row_count, right_arg.row_count)),
            left_arg,
        )[1],
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_mixed_binary_constructive_gpu",
        lambda *args, **kwargs: pytest.fail(
            "tiny mixed polygonal intersections should use the unified rowwise "
            "overlay shortcut before the mixed tag-pair dispatcher"
        ),
    )

    result = constructive_module._binary_constructive_gpu(
        "intersection",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert rowwise_calls == [(2, 2)]
    assert result is not None
    assert result.row_count == left.row_count


def test_overlay_difference_polygon_line_keeps_noded_polygon_boundary_under_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col3": [1, 2]},
        geometry=GeoSeries(
            [
                LineString([(2, 0), (2, 4), (6, 4)]),
                LineString([(0, 3), (6, 3)]),
            ]
        ),
    )

    with strict_native_environment():
        result = overlay(left, right, how="difference", keep_geom_type=True)

    expected = GeoSeries(
        [
            Polygon([(1, 1), (2, 1), (3, 1), (3, 3), (2, 3), (1, 3), (1, 1)]),
            Polygon([(3, 3), (5, 3), (5, 4), (5, 5), (3, 5), (3, 4), (3, 3)]),
        ],
        crs=left.crs,
    )

    assert len(result) == 2
    for got, exp in zip(result.geometry, expected, strict=True):
        assert got.normalize().equals_exact(exp.normalize(), tolerance=1e-9)


def test_overlay_identity_polygon_point_keeps_difference_rows_separate_under_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col4": [1, 2]},
        geometry=GeoSeries([Point((2, 2)), Point((3, 3))]),
    )

    with strict_native_environment():
        result = overlay(left, right, how="identity", keep_geom_type=True)

    expected = GeoSeries(
        [
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
        ],
        crs=left.crs,
    )

    assert len(result) == 2
    for got, exp in zip(result.geometry, expected, strict=True):
        assert got.normalize().equals_exact(exp.normalize(), tolerance=1e-9)


def test_overlay_difference_line_polygon_splits_outside_segments_under_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col3": [1, 2]},
        geometry=GeoSeries(
            [
                LineString([(2, 0), (2, 4), (6, 4)]),
                LineString([(0, 3), (6, 3)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    with strict_native_environment():
        result = overlay(left, right, how="difference", keep_geom_type=True)

    expected = GeoSeries(
        [
            MultiLineString(
                [
                    [(2, 0), (2, 1)],
                    [(2, 3), (2, 4), (3, 4)],
                    [(5, 4), (6, 4)],
                ]
            ),
            MultiLineString(
                [
                    [(0, 3), (1, 3)],
                    [(5, 3), (6, 3)],
                ]
            ),
        ],
        crs=left.crs,
    )

    assert len(result) == 2
    for got, exp in zip(result.geometry, expected, strict=True):
        assert got.normalize().equals_exact(exp.normalize(), tolerance=1e-9)


def test_overlay_union_collapses_split_polygon_fragments_under_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (10, 0), (10, 4), (0, 4), (0, 0)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(2, 0), (4, 0), (4, 4), (2, 4), (2, 0)]),
                Polygon([(6, 0), (8, 0), (8, 4), (6, 4), (6, 0)]),
            ]
        ),
    )

    with strict_native_environment():
        result = overlay(left, right, how="union")

    assert len(result) == 3


def test_overlay_union_survives_strict_native_mode_for_small_pairwise_polygons() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    with strict_native_environment():
        result = overlay(left, right, how="union")

    assert len(result) == 7
    assert result.geometry.notna().all()


def test_overlay_union_keeps_boundary_touching_difference_rows_separate_in_strict_native_mode() -> (
    None
):
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2, 3]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    with strict_native_environment():
        result = overlay(left, right, how="union", keep_geom_type=True)

    result = result.sort_values(["col1", "col2"], na_position="first").reset_index(drop=True)
    assert len(result) == 3
    assert result.geometry.iloc[0].equals(Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]))


def test_overlay_symmetric_difference_boundary_touches_preserve_polygon_in_strict_native_mode() -> (
    None
):
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2, 3]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    with strict_native_environment():
        result = overlay(left, right, how="symmetric_difference", keep_geom_type=False)

    assert len(result) == 1
    assert result.geometry.iloc[0].equals(Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]))


def test_overlay_union_warns_but_succeeds_on_crs_mismatch_in_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            ]
        ),
        crs=4326,
    )
    right = GeoDataFrame(
        {"col2": [1]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            ]
        ),
        crs=3857,
    )

    with strict_native_environment():
        with pytest.warns(UserWarning, match="CRS mismatch between the CRS"):
            result = overlay(left, right, how="union")

    assert result.crs == left.crs
    assert len(result) == 3
    assert result.geometry.notna().all()


def test_overlay_intersection_warns_on_dropped_lower_dim_results_in_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    right = GeoDataFrame(
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    with strict_native_environment():
        with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay"):
            result = overlay(left, right, keep_geom_type=None)

    assert list(result.geom_type) == ["Polygon", "Polygon"]
    assert len(result) == 2


def test_overlay_intersection_keeps_touch_line_when_keep_geom_type_false_in_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    right = GeoDataFrame(
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    with strict_native_environment():
        result = overlay(left, right, keep_geom_type=False)

    geom_types = result.geom_type.tolist()
    assert geom_types.count("Polygon") == 2
    assert any(geom_type in {"LineString", "MultiLineString"} for geom_type in geom_types)


def test_overlay_intersection_keep_geom_type_true_skips_geometry_collection_cpu_fallback() -> None:
    left = GeoDataFrame(
        {"left": [0, 1]},
        geometry=GeoSeries(
            [
                box(0, 0, 1, 1),
                box(1, 1, 3, 3).union(box(1, 3, 5, 5)),
            ]
        ),
    )
    right = GeoDataFrame(
        {"right": [0, 1]},
        geometry=GeoSeries(
            [
                box(0, 0, 1, 1),
                box(3, 1, 4, 2).union(box(4, 1, 5, 4)),
            ]
        ),
    )

    result = overlay(left, right, how="intersection", keep_geom_type=True)

    assert set(result.geometry.geom_type.unique()) <= {"Polygon", "MultiPolygon"}


def test_overlay_intersection_many_vs_one_remainder_declines_before_host_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_rem = from_shapely_geometries([box(0, 0, 2, 2)], residency=Residency.HOST)
    right_one = from_shapely_geometries([box(1, 1, 3, 3)], residency=Residency.HOST)

    monkeypatch.setattr(overlay_module, "has_gpu_runtime", lambda: False)

    def _wrapped_to_shapely(self, *args, **kwargs):
        pytest.fail("many-vs-one native decline should not materialize Shapely")

    monkeypatch.setattr(OwnedGeometryArray, "to_shapely", _wrapped_to_shapely)

    vibespatial.clear_fallback_events()
    with pytest.raises(overlay_module._OverlayNativeConstructiveDeclined):
        overlay_module._many_vs_one_intersection_owned(left_rem, right_one, 0)
    assert vibespatial.get_fallback_events(clear=True) == []


def test_many_vs_one_exact_remainder_uses_canonical_capacity_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    constructive_source = Path("src/vibespatial/constructive/binary_constructive.py").read_text()
    assert "_dispatch_polygon_intersection_containment_rowwise_gpu" not in constructive_source
    assert "_dispatch_mixed_polygonal_exact_remainder_gpu" not in constructive_source
    assert "row_isolated_polygon_capacity_partition_gpu" in constructive_source
    left = from_shapely_geometries(
        [box(9.5, -2.0 + i * 0.1, 10.5, -1.8 + i * 0.1) for i in range(40)],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [Point(0.0, 0.0).buffer(10.0, quad_segs=64)],
        residency=Residency.DEVICE,
    )

    with strict_native_environment():
        result = overlay_module._many_vs_one_intersection_owned(left, right, 0)

    assert result.row_count == left.row_count
    assert any(geom is not None and not geom.is_empty for geom in result.to_shapely())


def test_aligned_polygon_intersection_partitions_at_row_capacity() -> None:
    source = Path("src/vibespatial/constructive/binary_constructive.py").read_text()
    start = source.index("def _dispatch_polygon_simple_intersection_with_exact_remainder_gpu(")
    end = source.index("\ndef ", start + 1)
    function_source = source[start:end]
    router_start = source.index("def _dispatch_partitioned_polygon_intersection_gpu(")
    router_end = source.index("\ndef ", router_start + 1)
    router_source = source[router_start:router_end]

    assert "NativeDeviceSelection.from_mask(" in function_source
    assert ".partition_capacity_positions()" in function_source
    assert "._apply_row_activity(" in function_source
    assert "device_scatter_owned_capacity_selection(" in function_source
    assert "partition_counts=device-resident" in function_source
    assert "cp.flatnonzero" not in function_source
    assert "except Exception" not in function_source
    assert router_source.count("NativeDeviceSelection.from_mask(") == 5
    assert "device_scatter_owned_capacity_selections_many(" in router_source
    assert "partition_counts=device-resident" in router_source
    assert "cp.flatnonzero" not in router_source
    assert "except Exception" not in router_source
    assert "_dispatch_polygon_intersection_containment_rowwise_gpu" not in source
    assert "_dispatch_mixed_polygonal_exact_remainder_gpu" not in source


def test_keep_geom_type_filter_device_sources_stay_off_host_semantic_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(1, 1, 3, 3).union(box(1, 3, 5, 5)),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(3, 1, 4, 2).union(box(4, 1, 5, 4)),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(4, 3, 5, 4),
                ]
            )
        )
    )

    def _fail_take(series: GeoSeries, rows: np.ndarray) -> np.ndarray:
        raise AssertionError("device-backed keep_geom_type classification should not host-probe")

    monkeypatch.setattr(overlay_module, "_take_geoseries_object_values", _fail_take)

    vibespatial.clear_fallback_events()
    with strict_native_environment():
        filtered, dropped, keep_mask = (
            overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
                left_pairs=None,
                right_pairs=None,
                area_pairs=area_pairs,
                keep_geom_type_warning=True,
                left_source=left_source,
                right_source=right_source,
                left_rows=np.asarray([0], dtype=np.intp),
                right_rows=np.asarray([0], dtype=np.intp),
            )
        )

    assert keep_mask.tolist() == [True]
    assert dropped == 1
    assert len(filtered) == 1
    assert getattr(filtered.values, "_owned", None) is not None
    assert filtered.values._owned.residency is Residency.DEVICE
    events = vibespatial.get_dispatch_events(clear=True)
    assert any(
        event.operation == "keep_geom_type_warning_count"
        and event.implementation == "device_boundary_warning_count"
        for event in events
    )
    assert vibespatial.get_fallback_events(clear=True) == []


def test_overlay_intersection_device_backed_auto_stays_on_gpu_boundary() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"left": [0, 1]},
        geometry=DeviceGeometryArray._from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 2, 2),
                    box(2, 2, 4, 4),
                ],
                residency=Residency.DEVICE,
            )
        ),
    )
    right = GeoDataFrame(
        {"right": [0, 1]},
        geometry=DeviceGeometryArray._from_owned(
            from_shapely_geometries(
                [
                    box(1, 1, 3, 3),
                    box(3, 3, 5, 5),
                ],
                residency=Residency.DEVICE,
            )
        ),
    )

    vibespatial.clear_dispatch_events()
    result = overlay(left, right, how="intersection")
    events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 3
    assert any(
        event.surface == "geopandas.array.intersection" and event.selected is ExecutionMode.GPU
        for event in events
    )
    assert any(
        event.surface == "geopandas.overlay" and event.selected is ExecutionMode.GPU
        for event in events
    )
    assert not any(
        event.surface == "geopandas.overlay" and event.implementation == "shapely_host"
        for event in events
    )
    assert not any(
        event.surface == "geopandas.array.make_valid" and event.selected is ExecutionMode.CPU
        for event in events
    )
    result_owned = getattr(result.geometry.values, "_owned", None)
    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert result_owned is not None
    assert result_owned.residency is Residency.DEVICE


def test_overlay_intersection_device_backed_routing_avoids_public_geometry_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    def _fail_public_export(self):
        raise AssertionError("overlay routing must use private native metadata")

    monkeypatch.setattr(GeoDataFrame, "geom_type", property(_fail_public_export))
    monkeypatch.setattr(GeoDataFrame, "total_bounds", property(_fail_public_export))

    left = GeoDataFrame(
        {"left": [0, 1]},
        geometry=DeviceGeometryArray._from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 2, 2),
                    box(2, 2, 4, 4),
                ],
                residency=Residency.DEVICE,
            )
        ),
    )
    right = GeoDataFrame(
        {"right": [0, 1]},
        geometry=DeviceGeometryArray._from_owned(
            from_shapely_geometries(
                [
                    box(1, 1, 3, 3),
                    box(3, 3, 5, 5),
                ],
                residency=Residency.DEVICE,
            )
        ),
    )

    def _fail_array_export(self, dtype=None, copy=None):
        raise AssertionError("overlay routing must not export device geometry arrays")

    monkeypatch.setattr(DeviceGeometryArray, "__array__", _fail_array_export)
    monkeypatch.setattr(GeometryArray, "__array__", _fail_array_export)

    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()
    result = overlay(left, right, how="intersection")
    events = get_materialization_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert len(result) == 3
    assert not any(
        event.operation in {"geodataframe_geom_type", "geodataframe_total_bounds"}
        for event in events
    )
    assert "DeviceGeometryArray total-bounds device summary host boundary" not in runtime_reasons


def test_overlay_validity_cache_seed_uses_owned_family_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (1, 0), (0, 1), (0, 0)]),
            Polygon([(2, 0), (3, 0), (2, 1), (2, 0)]),
        ]
    )
    series = GeoSeries(GeometryArray.from_owned(owned))

    monkeypatch.setattr(
        GeoSeries,
        "geom_type",
        property(
            lambda self: pytest.fail("owned validity-cache seeding should not use public geom_type")
        ),
    )

    overlay_module._maybe_seed_polygon_validity_cache(series)

    cached = getattr(owned, "_cached_is_valid_mask", None)
    assert cached is not None
    assert cached.tolist() == [True, True]


def test_overlay_polygon_repair_probe_skips_array_export_for_valid_owned_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (1, 0), (0, 1), (0, 0)]),
            Polygon([(2, 0), (3, 0), (2, 1), (2, 0)]),
        ]
    )
    owned._polygon_rect_boundary_overlap = np.asarray([True, False], dtype=bool)
    series = GeoSeries(GeometryArray.from_owned(owned))

    monkeypatch.setattr(
        GeometryArray,
        "__array__",
        lambda *args, **kwargs: pytest.fail(
            "valid owned overlap repair probe should not materialize GeometryArray"
        ),
    )

    result = overlay_module._repair_invalid_polygon_output_rows(series)

    assert result is series


def test_overlay_polygon_repair_probe_device_validity_stays_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    validity_module = importlib.import_module("vibespatial.constructive.validity")
    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (1, 0), (0, 1), (0, 0)]),
            Polygon([(2, 0), (3, 0), (2, 1), (2, 0)]),
        ],
        residency=Residency.DEVICE,
    )
    owned._polygon_rect_boundary_overlap = np.asarray([True, False], dtype=bool)
    series = GeoSeries(GeometryArray.from_owned(owned))

    monkeypatch.setattr(
        validity_module,
        "is_valid_owned",
        lambda *_args, **_kwargs: pytest.fail(
            "device repair probe should use native validity expressions"
        ),
    )
    monkeypatch.setattr(
        GeometryArray,
        "__array__",
        lambda *args, **kwargs: pytest.fail(
            "valid device-owned repair probe should not materialize GeometryArray"
        ),
    )

    result = overlay_module._repair_invalid_polygon_output_rows(series)

    assert result is series


def test_overlay_polygon_repair_invalid_device_declines_in_strict_native() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    invalid = Polygon(
        [
            (680.0, 390.0),
            (680.0, 400.0),
            (680.0, 400.0),
            (680.0, 391.929775259655),
            (679.7726140184767, 390.0),
            (680.0, 390.0),
        ]
    )
    owned = from_shapely_geometries([invalid], residency=Residency.DEVICE)
    owned._polygon_rect_boundary_overlap = np.asarray([True], dtype=bool)
    series = GeoSeries(GeometryArray.from_owned(owned))

    with strict_native_environment(), pytest.raises(StrictNativeFallbackError):
        overlay_module._repair_invalid_polygon_output_rows(series)


def test_overlay_intersection_host_backed_polygons_with_owned_pair_stay_on_gpu_boundary() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"left": [0, 1]},
        geometry=GeoSeries(
            [
                box(0, 0, 2, 2),
                box(2, 2, 4, 4),
            ]
        ),
    )
    right = GeoDataFrame(
        {"right": [0, 1]},
        geometry=GeoSeries(
            [
                box(1, 1, 3, 3),
                box(3, 3, 5, 5),
            ]
        ),
    )

    left_owned, right_owned = overlay_module._extract_owned_pair(left, right)
    assert left_owned is not None
    assert right_owned is not None

    vibespatial.clear_dispatch_events()
    result = overlay(left, right, how="intersection")
    events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 3
    assert any(
        event.surface == "geopandas.array.intersection" and event.selected is ExecutionMode.GPU
        for event in events
    )


def test_extract_owned_pair_promotes_host_peer_when_other_side_is_device_backed() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"left": [0, 1]},
        geometry=DeviceGeometryArray._from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 2, 2),
                    box(2, 2, 4, 4),
                ],
                residency=Residency.DEVICE,
            )
        ),
    )
    right = GeoDataFrame(
        {"right": [0, 1]},
        geometry=GeoSeries(
            [
                box(1, 1, 3, 3),
                box(3, 3, 5, 5),
            ]
        ),
    )

    vibespatial.clear_dispatch_events()
    left_owned, right_owned = overlay_module._extract_owned_pair(left, right)
    events = vibespatial.get_dispatch_events(clear=True)

    assert left_owned is not None
    assert right_owned is not None
    assert left_owned.residency is Residency.DEVICE
    assert right_owned.residency is Residency.DEVICE
    assert any(
        event.surface == "geopandas.overlay" and event.selected is ExecutionMode.GPU
        for event in events
    )
    assert not any(
        event.surface == "geopandas.overlay" and event.implementation == "shapely_host"
        for event in events
    )


def test_overlay_intersection_keep_geom_type_true_skips_full_lower_dim_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    right = GeoDataFrame(
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    monkeypatch.setattr(
        overlay_module,
        "_assemble_polygon_intersection_rows_with_lower_dim",
        lambda *args, **kwargs: pytest.fail("full lower-dim assembly should be skipped"),
    )

    with strict_native_environment():
        result = overlay(left, right, keep_geom_type=True)

    assert list(result.geom_type) == ["Polygon", "Polygon"]


def test_overlay_intersection_keep_geom_type_none_warns_for_geometry_collection_rows() -> None:
    left = GeoDataFrame(
        {"left": [0, 1]},
        geometry=GeoSeries(
            [
                box(0, 0, 1, 1),
                box(1, 1, 3, 3).union(box(1, 3, 5, 5)),
            ]
        ),
    )
    right = GeoDataFrame(
        {"right": [0, 1]},
        geometry=GeoSeries(
            [
                box(0, 0, 1, 1),
                box(3, 1, 4, 2).union(box(4, 1, 5, 4)),
            ]
        ),
    )

    with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay"):
        result = overlay(left, right, how="intersection", keep_geom_type=None)

    assert set(result.geometry.geom_type.unique()) <= {"Polygon", "MultiPolygon"}


def test_overlay_intersection_keep_geom_type_warning_keeps_positive_polygon_part() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    data_dir = (
        Path(__file__).resolve().parent
        / "upstream"
        / "geopandas"
        / "tests"
        / "data"
        / "overlay"
        / "geom_type"
    )
    left = read_file(data_dir / "df1.geojson")
    right = read_file(data_dir / "df2.geojson")

    vibespatial.clear_dispatch_events()
    with strict_native_environment():
        with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay"):
            result = overlay(left, right, how="intersection", keep_geom_type=None)
    events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 1
    assert result.geometry.geom_type.tolist() == ["Polygon"]
    assert float(result.geometry.iloc[0].area) > 0.0
    assert any(
        event.implementation == "polygon_intersection_topology_remnant_gpu" for event in events
    )


def test_overlay_keep_geom_type_relation_selection_skips_legacy_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    data_dir = (
        Path(__file__).resolve().parent
        / "upstream"
        / "geopandas"
        / "tests"
        / "data"
        / "overlay"
        / "geom_type"
    )
    left = read_file(data_dir / "df1.geojson")
    right = read_file(data_dir / "df2.geojson")

    call_count = 0

    def _count_filter(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise AssertionError("relation selection should not use the legacy GeoSeries filter")

    monkeypatch.setattr(
        overlay_module,
        "_filter_polygon_intersection_rows_for_keep_geom_type",
        _count_filter,
    )
    monkeypatch.setattr(
        overlay_module,
        "_make_valid_geoseries",
        lambda gs, *args, **kwargs: gs,
    )

    vibespatial.get_dispatch_events(clear=True)
    with strict_native_environment():
        with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay"):
            result = overlay(left, right, how="intersection", keep_geom_type=None)
    events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 1
    assert call_count == 0
    assert any(
        event.implementation == "cached_relation_selection_constructive_gpu" for event in events
    )


def test_overlay_default_keep_geom_type_skips_warning_refinement_when_warning_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    overlay_module = importlib.import_module("vibespatial.api.tools.overlay")

    left = GeoDataFrame(
        {"left": [0]},
        geometry=GeoSeries([box(0, 0, 2, 2)]),
    )
    right = GeoDataFrame(
        {"right": [0]},
        geometry=GeoSeries([box(2, 0, 4, 2)]),
    )

    def _unexpected_warning_refinement(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("ignored keep-geom-type warning should not refine dropped parts")

    monkeypatch.setattr(
        overlay_module,
        "_device_count_dropped_polygon_intersection_warning_rows",
        _unexpected_warning_refinement,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        result = overlay(left, right, how="intersection", keep_geom_type=None)

    assert result.empty


def test_overlay_intersection_keep_geom_type_none_strict_warning_matches_host_count() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"left": [0, 1]},
        geometry=GeoSeries(
            [
                box(0, 0, 1, 1),
                box(1, 1, 3, 3).union(box(1, 3, 5, 5)),
            ]
        ),
    )
    right = GeoDataFrame(
        {"right": [0, 1]},
        geometry=GeoSeries(
            [
                box(0, 0, 1, 1),
                box(3, 1, 4, 2).union(box(4, 1, 5, 4)),
            ]
        ),
    )

    with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay") as host_warning:
        overlay(left, right, how="intersection", keep_geom_type=None)

    with strict_native_environment():
        with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay") as strict_warning:
            overlay(left, right, how="intersection", keep_geom_type=None)

    assert str(strict_warning[0].message) == str(host_warning[0].message)


def test_overlay_intersection_keep_geom_type_none_skips_full_lower_dim_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    right = GeoDataFrame(
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    monkeypatch.setattr(
        overlay_module,
        "_assemble_polygon_intersection_rows_with_lower_dim",
        lambda *args, **kwargs: pytest.fail("full lower-dim assembly should be skipped"),
    )

    with strict_native_environment():
        with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay"):
            result = overlay(left, right, keep_geom_type=None)

    assert list(result.geom_type) == ["Polygon", "Polygon"]


def test_keep_geom_type_filter_uses_geometry_array_values_not_geoseries_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_pairs = GeoSeries(
        [
            box(0, 0, 1, 1),
            box(0, 0, 1, 1),
            box(0, 0, 1, 1),
        ]
    )
    right_pairs = GeoSeries(
        [
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
            box(2, 2, 3, 3),
        ]
    )
    area_pairs = GeoSeries(
        [
            GeometryCollection([box(0, 0, 1, 1), LineString([(0, 0), (1, 0)])]),
            LineString([(1, 0), (1, 1)]),
            None,
        ]
    )

    def _fail(*_args, **_kwargs):
        pytest.fail("GeoSeries wrapper path should stay cold")

    monkeypatch.setattr(GeoSeries, "__array__", _fail, raising=False)
    monkeypatch.setattr(GeoSeries, "isna", _fail, raising=False)
    monkeypatch.setattr(GeoSeries, "is_empty", property(lambda self: _fail()))
    monkeypatch.setattr(GeoSeries, "geom_type", property(lambda self: _fail()))

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            left_pairs,
            right_pairs,
            area_pairs,
            keep_geom_type_warning=True,
        )
    )

    assert keep_mask.tolist() == [True, False, False]
    assert dropped == 1
    assert len(filtered) == 1
    filtered_values = np.asarray(filtered.array, dtype=object)
    assert shapely.get_type_id(filtered_values).tolist() == [3]


def test_keep_geom_type_filter_preserves_owned_results_without_array_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(0, 0, 1, 1),
                    box(0, 0, 1, 1),
                ]
            )
        )
    )
    right_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(1, 0, 2, 1),
                    box(2, 2, 3, 3),
                ]
            )
        )
    )
    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    LineString([(1, 0), (1, 1)]),
                    None,
                ]
            )
        )
    )

    def _fail(*_args, **_kwargs):
        pytest.fail("owned keep-geom-type filter should not materialize full geometry arrays")

    monkeypatch.setattr(GeometryArray, "__array__", _fail, raising=False)

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            left_pairs,
            right_pairs,
            area_pairs,
            keep_geom_type_warning=False,
        )
    )

    assert keep_mask.tolist() == [True, False, False]
    assert dropped == 0
    assert len(filtered) == 1
    assert getattr(filtered.values, "_owned", None) is not None


def test_keep_geom_type_filter_drops_zero_area_owned_polygon_rows() -> None:
    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(1, 0, 1, 2),
                    None,
                ]
            )
        )
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=False,
        )
    )

    assert keep_mask.tolist() == [True, False, False]
    assert dropped == 0
    assert len(filtered) == 1
    assert filtered.iloc[0].equals(box(0, 0, 1, 1))


def test_keep_geom_type_filter_device_sparse_positive_rows_skip_area_vector_export() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    rows = 16
    source_geoms = [box(float(i), 0.0, float(i) + 10.0, 10.0) for i in range(rows)]
    left_source = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries(source_geoms, residency=Residency.DEVICE))
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries(source_geoms, residency=Residency.DEVICE))
    )
    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [box(0, 0, 1, 1), *([None] * (rows - 1))],
                residency=Residency.DEVICE,
            )
        )
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=False,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(rows, dtype=np.intp),
            right_rows=np.arange(rows, dtype=np.intp),
        )
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert keep_mask.tolist() == [True, *([False] * (rows - 1))]
    assert dropped == 0
    assert len(filtered) == rows
    assert filtered.values._owned.row_count == rows
    assert "overlay keep-geometry-type area-by-row host export" not in runtime_reasons
    assert (
        "overlay keep-geometry-type polygonal positive-area mask host export" not in runtime_reasons
    )
    assert "overlay keep-geometry-type polygonal positive-area terminal rows export" not in (
        runtime_reasons
    )


def test_keep_geom_type_filter_uses_aligned_owned_pairs_without_pair_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")

    left_pairs_owned = from_shapely_geometries(
        [box(0.0, 0.0, 3.0, 3.0)],
        residency=Residency.DEVICE,
    )
    right_pairs_owned = from_shapely_geometries(
        [box(1.0, 1.0, 2.0, 2.0)],
        residency=Residency.DEVICE,
    )
    area_owned = from_shapely_geometries(
        [box(1.0, 1.0, 2.0, 2.0)],
        residency=Residency.DEVICE,
    )
    area_owned._aligned_left_pairs_owned = left_pairs_owned
    area_owned._aligned_right_pairs_owned = right_pairs_owned
    area_pairs = GeoSeries(GeometryArray.from_owned(area_owned))

    monkeypatch.setattr(
        overlay_module,
        "_owned_positive_polygon_mask_and_areas",
        lambda *_args, **_kwargs: pytest.fail(
            "aligned owned pair metadata should keep keep-geom-type filtering on the device path"
        ),
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
        )
    )

    assert hasattr(keep_mask, "__cuda_array_interface__")
    assert cp.asnumpy(keep_mask).tolist() == [True]
    assert dropped == 0
    assert len(filtered) == 1


def test_keep_geom_type_filter_preserves_positive_fp64_area_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    measurement_module = importlib.import_module("vibespatial.constructive.measurement")
    public_area_calls: list[int] = []

    def _public_area_sentinel(owned, *args, **kwargs):
        public_area_calls.append(int(owned.row_count))
        return np.full(int(owned.row_count), 1.0, dtype=np.float64)

    monkeypatch.setattr(measurement_module, "area_owned", _public_area_sentinel)

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [box(350_000.0, 3_078_000.0, 360_000.0, 3_088_000.0)],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [box(349_000.0, 3_077_000.0, 361_000.0, 3_089_000.0)],
                residency=Residency.DEVICE,
            )
        )
    )
    tiny_overlap = Polygon(
        [
            (350_000.0, 3_078_000.0),
            (350_000.0001, 3_078_000.0),
            (350_000.0001, 3_078_000.0001),
            (350_000.0, 3_078_000.0001),
            (350_000.0, 3_078_000.0),
        ]
    )
    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries([tiny_overlap], residency=Residency.DEVICE)
        )
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=False,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.asarray([0], dtype=np.intp),
            right_rows=np.asarray([0], dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True]
    assert dropped == 0
    assert len(filtered) == 1
    assert filtered.values._owned.row_count == 1
    assert public_area_calls == []


def test_keep_geom_type_filter_reuses_owned_overlap_area_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measurement_module = importlib.import_module("vibespatial.constructive.measurement")
    real_area_owned = measurement_module.area_owned
    overlap_area_calls: list[int] = []

    def _count_overlap_area_measurement(owned, *args, **kwargs):
        result = np.asarray(real_area_owned(owned, *args, **kwargs), dtype=np.float64)
        finite = result[np.isfinite(result)]
        if finite.size == 2 and np.allclose(np.sort(finite), np.asarray([1.0, 4.0])):
            overlap_area_calls.append(int(owned.row_count))
        return result

    monkeypatch.setattr(measurement_module, "area_owned", _count_overlap_area_measurement)

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 10, 10),
                    box(20, 20, 30, 30),
                ]
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 10, 10),
                    box(20, 20, 30, 30),
                ]
            )
        )
    )
    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(20, 20, 22, 22),
                ]
            )
        )
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=False,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.asarray([0, 1], dtype=np.intp),
            right_rows=np.asarray([0, 1], dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 0
    assert len(filtered) == 2
    assert overlap_area_calls == [2]


def test_keep_geom_type_filter_skips_device_warning_refinement_when_exact_values_cover_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(0, 0, 1, 1),
                ]
            )
        )
    )
    right_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(1, 0, 2, 1),
                ]
            )
        )
    )
    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    None,
                ]
            )
        )
    )
    area_pairs.values._exact_intersection_values = np.asarray(
        [
            GeometryCollection(
                [
                    box(0, 0, 1, 1),
                    LineString([(0, 0), (1, 1)]),
                ]
            ),
            LineString([(1, 0), (1, 1)]),
        ],
        dtype=object,
    )
    area_pairs.values._exact_intersection_value_mask = np.ones(2, dtype=bool)

    monkeypatch.setattr(
        overlay_module,
        "_clear_device_exact_keep_geom_type_warnings",
        lambda *args, **kwargs: pytest.fail(
            "cached exact intersection values should bypass device warning refinement"
        ),
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            left_pairs,
            right_pairs,
            area_pairs,
            keep_geom_type_warning=True,
        )
    )

    assert keep_mask.tolist() == [True, False]
    assert dropped == 2
    assert len(filtered) == 1
    assert filtered.iloc[0].equals(box(0, 0, 1, 1))


def test_overlay_intersection_small_exact_boundary_reuses_existing_owned_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame({"geometry": [box(0, 0, 2, 2)]}, geometry="geometry")
    right = GeoDataFrame({"geometry": [box(1, 1, 3, 3)]}, geometry="geometry")
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()

    def _fail(self):
        pytest.fail("small exact boundary path should reuse caller-provided owned pairs")

    monkeypatch.setattr(GeometryArray, "to_owned", _fail)

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned,
        right_owned,
        _warn_on_dropped_lower_dim_polygon_results=True,
    )

    assert used_owned is True
    assert_geodataframe_equal(
        result.reset_index(drop=True),
        GeoDataFrame({"geometry": [box(1, 1, 2, 2)]}, geometry="geometry"),
    )


def test_keep_geom_type_filter_warning_can_use_source_rows_without_pair_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_source = GeoSeries(
        [
            box(0, 0, 1, 1),
            box(0, 0, 1, 1),
            box(0, 0, 1, 1),
        ]
    )
    right_source = GeoSeries(
        [
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
            box(2, 2, 3, 3),
        ]
    )
    area_pairs = GeoSeries(
        [
            GeometryCollection([box(0, 0, 1, 1), LineString([(0, 0), (1, 0)])]),
            LineString([(1, 0), (1, 1)]),
            None,
        ]
    )

    real_take = overlay_module._take_geoseries_object_values

    def _take_only_sources(series: GeoSeries, rows: np.ndarray) -> np.ndarray:
        if series is left_source or series is right_source or series is area_pairs:
            return real_take(series, rows)
        raise AssertionError("pair-series materialization should stay cold")

    monkeypatch.setattr(overlay_module, "_take_geoseries_object_values", _take_only_sources)

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(3, dtype=np.intp),
            right_rows=np.arange(3, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, False, False]
    assert dropped == 1
    assert len(filtered) == 1


def test_keep_geom_type_filter_skips_all_row_take_when_every_polygon_row_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_source = GeoSeries(
        [
            box(0, 0, 1, 1),
            box(2, 0, 3, 1),
        ]
    )
    right_source = GeoSeries(
        [
            box(0, 0, 1, 1),
            box(2, 0, 3, 1),
        ]
    )
    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(2, 0, 3, 1),
                ],
                residency=Residency.DEVICE,
            )
        )
    )

    monkeypatch.setattr(
        GeoSeries,
        "take",
        lambda *args, **kwargs: pytest.fail(
            "all-surviving keep-geom-type filter should not take every row"
        ),
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(2, dtype=np.intp),
            right_rows=np.arange(2, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 0
    assert len(filtered) == 2


def test_keep_geom_type_filter_skips_warning_count_when_kept_rows_have_no_boundary_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_source = GeoSeries(
        [
            box(0, 0, 5, 5),
            box(10, 10, 16, 16),
        ]
    )
    right_source = GeoSeries(
        [
            box(1, 1, 2, 2),
            box(12, 12, 13, 13),
        ]
    )
    area_pairs = GeoSeries(
        [
            box(1, 1, 2, 2),
            box(12, 12, 13, 13),
        ]
    )

    monkeypatch.setattr(
        overlay_module,
        "_count_dropped_polygon_intersection_parts",
        lambda *_args, **_kwargs: pytest.fail(
            "nested polygon rows with no boundary overlap should not trigger dropped-count exact intersection"
        ),
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(2, dtype=np.intp),
            right_rows=np.arange(2, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 0
    assert len(filtered) == 2


def test_keep_geom_type_warning_count_uses_native_boundary_topology() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    left_geoms = [
        Polygon([(0.0, 0.0), (3.0, 0.0), (0.0, 3.0), (0.0, 0.0)]),
        Polygon([(10.0, 0.0), (11.0, 0.0), (10.0, 1.0), (10.0, 0.0)]),
    ]
    right_geoms = [
        Polygon([(1.0, -1.0), (4.0, 2.0), (1.0, 5.0), (1.0, -1.0)]),
        Polygon([(11.0, 0.0), (12.0, 0.0), (11.0, 1.0), (11.0, 0.0)]),
    ]
    area_geoms = [
        left_geoms[0].intersection(right_geoms[0]),
        None,
    ]
    left_pairs = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries(left_geoms, residency=Residency.DEVICE))
    )
    right_pairs = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries(right_geoms, residency=Residency.DEVICE))
    )
    area_pairs = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries(area_geoms, residency=Residency.DEVICE))
    )

    vibespatial.clear_dispatch_events()
    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            left_pairs,
            right_pairs,
            area_pairs,
            keep_geom_type_warning=True,
        )
    )

    assert cp.asnumpy(keep_mask).tolist() == [True, False]
    assert dropped == 1
    assert len(filtered) == 2
    assert filtered.values._owned.row_count == 2
    events = vibespatial.get_dispatch_events(clear=True)
    assert any(
        event.implementation == "polygon_pair_warning_candidate_remnants_gpu" for event in events
    )


def test_keep_geom_type_warning_rowset_classifies_multipolygon_area_and_exposed_point() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.constructive.boundary_remnants import (
        polygon_pair_boundary_remnant_mask_capacity_device,
    )

    left_geom = MultiPolygon(
        [
            box(0.0, 0.0, 2.0, 2.0),
            box(4.0, 0.0, 6.0, 2.0),
            box(8.0, 0.0, 9.0, 1.0),
        ]
    )
    right_geom = MultiPolygon(
        [
            box(1.0, 1.0, 3.0, 3.0),
            box(5.0, 1.0, 7.0, 3.0),
            box(9.0, 1.0, 10.0, 2.0),
        ]
    )
    exact = left_geom.intersection(right_geom)
    retained = MultiPolygon([part for part in exact.geoms if isinstance(part, Polygon)])
    assert exact.geom_type == "GeometryCollection"
    assert retained.geom_type == "MultiPolygon"

    ordinary_left = box(20.0, 0.0, 23.0, 3.0)
    ordinary_right = box(21.0, 1.0, 24.0, 4.0)
    left = from_shapely_geometries([left_geom, ordinary_left], residency=Residency.DEVICE)
    right = from_shapely_geometries([right_geom, ordinary_right], residency=Residency.DEVICE)
    area = from_shapely_geometries(
        [retained, ordinary_left.intersection(ordinary_right)],
        residency=Residency.DEVICE,
    )

    warning_result = polygon_pair_boundary_remnant_mask_capacity_device(
        left,
        right,
        area,
        keep_area_mask=cp.asarray([True, True], dtype=cp.bool_),
    )

    assert warning_result is not None
    dropped, supported = warning_result
    assert cp.asnumpy(supported).tolist() == [True, True]
    assert cp.asnumpy(dropped).tolist() == [True, False]


def test_intersection_topology_metadata_replaces_warning_boundary_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import vibespatial.constructive.boundary_remnants as boundary_remnants_module

    left = GeoDataFrame(
        {"left": [0, 1]},
        geometry=GeoSeries(
            [
                MultiPolygon(
                    [
                        box(0.0, 0.0, 2.0, 2.0),
                        box(4.0, 0.0, 6.0, 2.0),
                    ]
                ),
                box(20.0, 0.0, 23.0, 3.0),
            ]
        ),
    )
    right = GeoDataFrame(
        {"right": [0, 1]},
        geometry=GeoSeries(
            [
                MultiPolygon(
                    [
                        box(1.0, 1.0, 3.0, 3.0),
                        box(6.0, 2.0, 7.0, 3.0),
                    ]
                ),
                box(21.0, 1.0, 24.0, 4.0),
            ]
        ),
    )

    def _unexpected_reconstruction(*args, **kwargs):
        raise AssertionError("intersection topology metadata must classify warning rows")

    monkeypatch.setattr(
        boundary_remnants_module,
        "polygon_pair_boundary_remnant_mask_capacity_device",
        _unexpected_reconstruction,
    )
    vibespatial.clear_dispatch_events()
    with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay"):
        result = overlay(left, right, how="intersection", keep_geom_type=None)

    assert len(result) == 2
    assert all(geometry.geom_type == "Polygon" for geometry in result.geometry)
    events = vibespatial.get_dispatch_events(clear=True)
    assert any(
        event.implementation == "polygon_intersection_topology_remnant_gpu"
        for event in events
    )


def test_mixed_rectangle_remainder_preserves_exact_polygon_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.constructive.binary_constructive import (
        _dispatch_partitioned_polygon_intersection_gpu,
    )

    polygon_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_intersection"
    )

    monkeypatch.setattr(
        polygon_module,
        "polygon_intersection_sh_eligible_mask",
        lambda left, _right: cp.zeros(left.row_count, dtype=cp.bool_),
    )

    left = from_shapely_geometries(
        [
            box(0.0, 0.0, 2.0, 2.0),
            Polygon([(0.0, 0.0), (3.0, 0.0), (1.5, 3.0), (0.0, 0.0)]),
            Polygon([(0.0, 0.0), (4.0, 0.0), (2.0, 3.0), (0.0, 0.0)]),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            box(1.0, 1.0, 3.0, 3.0),
            Polygon([(1.0, -1.0), (4.0, 1.0), (1.0, 4.0), (1.0, -1.0)]),
            Polygon([(1.0, 0.5), (3.0, 0.5), (2.0, 2.5), (1.0, 0.5)]),
        ],
        residency=Residency.DEVICE,
    )

    result = _dispatch_partitioned_polygon_intersection_gpu(left, right)

    assert result is not None
    exact_polygon_only = getattr(result, "_polygon_rect_exact_polygon_only", None)
    assert exact_polygon_only is not None
    assert cp.asnumpy(exact_polygon_only).tolist()[1:] == [True, True]


def test_mixed_rectangle_sh_rows_preserve_exact_polygon_metadata() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.constructive.binary_constructive import (
        _dispatch_partitioned_polygon_intersection_gpu,
    )

    left = from_shapely_geometries(
        [
            box(0.0, 0.0, 2.0, 2.0),
            Polygon([(0.0, 0.0), (3.0, 0.0), (1.5, 3.0), (0.0, 0.0)]),
            Polygon([(0.0, 0.0), (4.0, 0.0), (2.0, 3.0), (0.0, 0.0)]),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            box(1.0, 1.0, 3.0, 3.0),
            Polygon([(1.0, -1.0), (4.0, 1.0), (1.0, 4.0), (1.0, -1.0)]),
            Polygon([(1.0, 0.5), (3.0, 0.5), (2.0, 2.5), (1.0, 0.5)]),
        ],
        residency=Residency.DEVICE,
    )

    result = _dispatch_partitioned_polygon_intersection_gpu(left, right)

    assert result is not None
    exact_polygon_only = getattr(result, "_polygon_rect_exact_polygon_only", None)
    assert exact_polygon_only is not None
    assert cp.asnumpy(exact_polygon_only).tolist() == [True, True, True]


def test_keep_geom_type_warning_count_uses_exact_metadata_rowset_without_de9im(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    left_geoms = [
        Polygon([(0.0, 0.0), (3.0, 0.0), (0.0, 3.0), (0.0, 0.0)]),
        Polygon([(10.0, 0.0), (11.0, 0.0), (10.0, 1.0), (10.0, 0.0)]),
    ]
    right_geoms = [
        Polygon([(1.0, -1.0), (4.0, 2.0), (1.0, 5.0), (1.0, -1.0)]),
        Polygon([(11.0, 0.0), (12.0, 0.0), (11.0, 1.0), (11.0, 0.0)]),
    ]
    left_pairs = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries(left_geoms, residency=Residency.DEVICE))
    )
    right_pairs = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries(right_geoms, residency=Residency.DEVICE))
    )
    area_owned = from_shapely_geometries(
        [left_geoms[0].intersection(right_geoms[0]), None],
        residency=Residency.DEVICE,
    )
    area_owned._polygon_rect_exact_polygon_only = cp.asarray(
        [False, False],
        dtype=cp.bool_,
    )
    area_pairs = GeoSeries(GeometryArray.from_owned(area_owned))

    monkeypatch.setattr(
        overlay_module,
        "_device_polygon_keep_geom_type_warning_mask_from_de9im",
        lambda *_args, **_kwargs: pytest.fail(
            "exact metadata rowset should bypass DE-9IM warning classification"
        ),
    )

    vibespatial.clear_dispatch_events()
    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            left_pairs,
            right_pairs,
            area_pairs,
            keep_geom_type_warning=True,
        )
    )

    assert cp.asnumpy(keep_mask).tolist() == [True, False]
    assert dropped == 1
    assert len(filtered) == 2
    assert filtered.iloc[1] is None
    events = vibespatial.get_dispatch_events(clear=True)
    assert any(
        event.operation == "keep_geom_type_warning_count"
        and event.implementation == "device_boundary_warning_count"
        for event in events
    )


def test_keep_geom_type_filter_uses_rect_kernel_overlap_flag_to_skip_warning_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(2, 2, 3, 3),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    area_pairs.values._owned._polygon_rect_boundary_overlap = np.zeros(2, dtype=bool)

    monkeypatch.setattr(
        overlay_module,
        "_count_dropped_polygon_intersection_parts",
        lambda *_args, **_kwargs: pytest.fail(
            "rectangle overlap flag should let keep-geom-type warning skip host dropped-count rebuild"
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_warning_candidate_mask_for_polygon_keep_geom_type",
        lambda *_args, **_kwargs: pytest.fail(
            "rectangle overlap flag should bypass generic warning candidate classification"
        ),
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=GeoSeries([box(-1, -1, 2, 2), box(1, 1, 4, 4)]),
            right_source=GeoSeries([box(0, 0, 1, 1), box(2, 2, 3, 3)]),
            left_rows=np.arange(2, dtype=np.intp),
            right_rows=np.arange(2, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 0
    assert len(filtered) == 2


def test_keep_geom_type_filter_rect_overlap_mask_only_materializes_warning_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            box(2, 2, 3, 3),
            box(4, 4, 5, 5),
        ],
        residency=Residency.DEVICE,
    )
    owned._polygon_rect_boundary_overlap = np.asarray([False, True, False], dtype=bool)
    area_pairs = GeoSeries(GeometryArray.from_owned(owned))
    left_source = GeoSeries(
        [
            box(0, 0, 4, 4),
            box(2, 2, 6, 6),
            box(4, 4, 8, 8),
        ]
    )
    right_source = GeoSeries(
        [
            box(0, 0, 1, 1),
            box(2, 2, 3, 3),
            box(4, 4, 5, 5),
        ]
    )

    observed_rows: list[tuple[int, ...]] = []

    def _take_only_warning_row(series: GeoSeries, rows: np.ndarray) -> np.ndarray:
        observed_rows.append(tuple(int(v) for v in rows))
        return np.asarray(series, dtype=object)[rows]

    monkeypatch.setattr(overlay_module, "_take_geoseries_object_values", _take_only_warning_row)
    monkeypatch.setattr(
        overlay_module,
        "_count_dropped_polygon_intersection_parts",
        lambda left_values, right_values, row_count, **_kwargs: row_count,
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(3, dtype=np.intp),
            right_rows=np.arange(3, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, True, True]
    assert dropped == 1
    assert len(filtered) == 3
    assert observed_rows == [(1,), (1,)]
    assert np.asarray(
        getattr(filtered.values._owned, "_polygon_rect_boundary_overlap", None),
        dtype=bool,
    ).tolist() == [False, True, False]


def test_keep_geom_type_filter_sparse_device_rect_overlap_exports_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    row_count = 24
    area_owned = from_shapely_geometries(
        [box(float(i), 0.0, float(i) + 0.5, 0.5) for i in range(row_count)],
        residency=Residency.DEVICE,
    )
    d_overlap = cp.zeros(row_count, dtype=cp.bool_)
    d_overlap[7] = True
    area_owned._polygon_rect_boundary_overlap = d_overlap
    area_pairs = GeoSeries(GeometryArray.from_owned(area_owned))
    left_source = GeoSeries(
        [box(float(i) - 0.25, -0.25, float(i) + 0.75, 0.75) for i in range(row_count)]
    )
    right_source = GeoSeries([box(float(i), 0.0, float(i) + 0.5, 0.5) for i in range(row_count)])
    observed_rows: list[tuple[int, ...]] = []

    def _take_only_warning_row(series: GeoSeries, rows: np.ndarray) -> np.ndarray:
        observed_rows.append(tuple(int(v) for v in rows))
        return np.asarray(series, dtype=object)[rows]

    monkeypatch.setattr(overlay_module, "_take_geoseries_object_values", _take_only_warning_row)
    monkeypatch.setattr(
        overlay_module,
        "_count_dropped_polygon_intersection_parts",
        lambda left_values, right_values, row_count, **_kwargs: row_count,
    )
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(row_count, dtype=np.intp),
            right_rows=np.arange(row_count, dtype=np.intp),
        )
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert keep_mask.tolist() == [True] * row_count
    assert dropped == 1
    assert len(filtered) == row_count
    assert observed_rows == [(7,), (7,)]
    assert "overlay keep-geom-type owned area-overlap rows host boundary" in runtime_reasons
    assert "overlay keep-geom-type owned area-overlap mask host boundary" not in runtime_reasons


def test_keep_geom_type_de9im_warning_mask_sparse_exports_rows() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    row_count = 24
    left = [box(float(i), 0.0, float(i) + 10.0, 10.0) for i in range(row_count)]
    right = [box(float(i) + 2.0, 2.0, float(i) + 4.0, 4.0) for i in range(row_count)]
    right[7] = box(9.0, 0.0, 11.0, 2.0)
    left_source = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries(left, residency=Residency.DEVICE))
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries(right, residency=Residency.DEVICE))
    )
    keep_mask = np.ones(row_count, dtype=bool)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    warning_mask = overlay_module._device_polygon_keep_geom_type_warning_mask_from_de9im(
        left_source,
        right_source,
        np.arange(row_count, dtype=np.intp),
        np.arange(row_count, dtype=np.intp),
        keep_mask,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert warning_mask.tolist() == [i == 7 for i in range(row_count)]
    assert "overlay keep-geom-type warning rows host boundary" in runtime_reasons
    assert "overlay keep-geom-type warning mask host boundary" not in runtime_reasons


def test_keep_geom_type_filter_rect_overlap_device_sources_stay_off_host_probe() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 4, 4),
                    box(10, 0, 14, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(2, 0, 6, 4),
                    box(12, 0, 16, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    area_owned = from_shapely_geometries(
        [
            box(2, 0, 4, 4),
            box(12, 0, 14, 4),
        ],
        residency=Residency.DEVICE,
    )
    area_owned._polygon_rect_boundary_overlap = np.asarray([True, True], dtype=bool)
    area_pairs = GeoSeries(GeometryArray.from_owned(area_owned))

    vibespatial.clear_fallback_events()
    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(2, dtype=np.intp),
            right_rows=np.arange(2, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 0
    assert len(filtered) == 2
    assert vibespatial.get_fallback_events(clear=True) == []


def test_keep_geom_type_filter_rect_overlap_preserves_positive_fp64_sliver() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [box(0, 0, 1000, 1000)],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [box(0, 0, 1000, 1000)],
                residency=Residency.DEVICE,
            )
        )
    )
    area_owned = from_shapely_geometries(
        [box(0, 0, 0.001, 0.001)],
        residency=Residency.DEVICE,
    )
    area_owned._polygon_rect_boundary_overlap = np.asarray([True], dtype=bool)
    area_pairs = GeoSeries(GeometryArray.from_owned(area_owned))

    vibespatial.clear_fallback_events()
    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.asarray([0], dtype=np.intp),
            right_rows=np.asarray([0], dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True]
    assert dropped == 0
    assert len(filtered) == 1
    assert vibespatial.get_fallback_events(clear=True) == []


def test_keep_geom_type_filter_rect_overlap_missing_polygon_empty_mask_stays_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 10, 4),
                    box(0, 0, 10, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(2, 0, 4, 4),
                    box(6, 0, 8, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    area_owned = from_shapely_geometries(
        [
            box(2, 0, 4, 4),
            box(6, 0, 8, 4),
        ],
        residency=Residency.DEVICE,
    )
    area_owned._polygon_rect_boundary_overlap = np.asarray([True, True], dtype=bool)
    object.__setattr__(
        area_owned.families[GeometryFamily.POLYGON],
        "empty_mask",
        np.empty(0, dtype=bool),
    )
    area_pairs = GeoSeries(GeometryArray.from_owned(area_owned))

    monkeypatch.setattr(
        overlay_module,
        "_take_geoseries_object_values",
        lambda *_args, **_kwargs: pytest.fail(
            "rect-overlap keep-geom-type should not materialize host values when polygon empty_mask metadata is absent"
        ),
    )

    vibespatial.clear_fallback_events()
    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(2, dtype=np.intp),
            right_rows=np.arange(2, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 0
    assert len(filtered) == 2
    assert vibespatial.get_fallback_events(clear=True) == []


def test_keep_geom_type_filter_rect_overlap_device_sources_fall_back_to_conservative_native_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 4, 4),
                    box(10, 0, 14, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(2, 0, 6, 4),
                    box(12, 0, 16, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    area_owned = from_shapely_geometries(
        [
            box(2, 0, 4, 4),
            box(12, 0, 14, 4),
        ],
        residency=Residency.DEVICE,
    )
    area_owned._polygon_rect_boundary_overlap = np.asarray([True, True], dtype=bool)
    area_pairs = GeoSeries(GeometryArray.from_owned(area_owned))

    monkeypatch.setattr(
        overlay_module,
        "_device_count_dropped_polygon_intersection_warning_rows",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        overlay_module,
        "_take_geoseries_object_values",
        lambda *_args, **_kwargs: pytest.fail(
            "device-backed rect-overlap warning counting should stay native when refinement fails"
        ),
    )

    vibespatial.clear_fallback_events()
    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(2, dtype=np.intp),
            right_rows=np.arange(2, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 2
    assert len(filtered) == 2
    assert vibespatial.get_fallback_events(clear=True) == []


def test_keep_geom_type_filter_rect_overlap_host_sources_use_device_pairs_before_host_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_geoms = [
        box(0, 0, 10, 4),
        box(0, 0, 10, 4),
    ]
    right_geoms = [
        box(2, 0, 4, 4),
        box(6, 0, 8, 4),
    ]
    left_source = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries(left_geoms, residency=Residency.HOST))
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries(right_geoms, residency=Residency.HOST))
    )
    left_pairs = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries(left_geoms, residency=Residency.DEVICE))
    )
    right_pairs = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries(right_geoms, residency=Residency.DEVICE))
    )
    area_owned = from_shapely_geometries(
        [
            box(2, 0, 4, 4),
            box(6, 0, 8, 4),
        ],
        residency=Residency.DEVICE,
    )
    area_owned._polygon_rect_boundary_overlap = np.asarray([True, True], dtype=bool)
    area_pairs = GeoSeries(GeometryArray.from_owned(area_owned))

    monkeypatch.setattr(
        overlay_module,
        "_take_geoseries_object_values",
        lambda *_args, **_kwargs: pytest.fail(
            "host-backed sources should not materialize when device-backed pair rows are available"
        ),
    )

    vibespatial.clear_fallback_events()
    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            left_pairs,
            right_pairs,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(2, dtype=np.intp),
            right_rows=np.arange(2, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 0
    assert len(filtered) == 2
    assert vibespatial.get_fallback_events(clear=True) == []


def test_keep_geom_type_filter_kept_rows_with_shared_boundary_on_area_boundary_stays_native() -> (
    None
):
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 10, 4),
                    box(0, 0, 10, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(2, 0, 4, 4),
                    box(6, 0, 8, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(2, 0, 4, 4),
                    box(6, 0, 8, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )

    vibespatial.clear_fallback_events()
    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(2, dtype=np.intp),
            right_rows=np.arange(2, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 0
    assert len(filtered) == 2
    assert vibespatial.get_fallback_events(clear=True) == []


def test_device_count_dropped_polygon_intersection_warning_rows_handles_large_distinct_pair_batches() -> (
    None
):
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    dropped_pairs = 40
    kept_pairs = 40

    left_geoms: list[object] = []
    right_geoms: list[object] = []
    area_geoms: list[object] = []
    keep_mask = np.zeros(dropped_pairs + kept_pairs, dtype=bool)

    for row in range(dropped_pairs):
        x0 = float(row * 10)
        left_geoms.append(box(x0, 0.0, x0 + 1.0, 1.0))
        right_geoms.append(box(x0 + 1.0, 0.0, x0 + 2.0, 1.0))
        area_geoms.append(None)

    for local_row in range(kept_pairs):
        row = dropped_pairs + local_row
        x0 = float(row * 10)
        left_geoms.append(box(x0, 0.0, x0 + 2.0, 2.0))
        right_geoms.append(box(x0 + 1.0, 1.0, x0 + 3.0, 3.0))
        area_geoms.append(box(x0 + 1.0, 1.0, x0 + 2.0, 2.0))
        keep_mask[row] = True

    left_source = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries(left_geoms, residency=Residency.DEVICE))
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries(right_geoms, residency=Residency.DEVICE))
    )
    area_owned = from_shapely_geometries(area_geoms, residency=Residency.DEVICE)

    dropped = overlay_module._device_count_dropped_polygon_intersection_warning_rows(
        area_owned,
        keep_mask,
        np.arange(dropped_pairs + kept_pairs, dtype=np.intp),
        left_source=left_source,
        right_source=right_source,
        left_rows=np.arange(dropped_pairs + kept_pairs, dtype=np.intp),
        right_rows=np.arange(dropped_pairs + kept_pairs, dtype=np.intp),
    )

    assert dropped == dropped_pairs


def test_device_warning_count_does_not_enter_generic_boundary_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    predicates_module = importlib.import_module("vibespatial.predicates.binary")
    monkeypatch.setattr(
        predicates_module,
        "evaluate_binary_predicate",
        lambda *_args, **_kwargs: pytest.fail(
            "native warning classification must not rebuild boundaries through a generic predicate"
        ),
    )

    left_geom = Polygon([(0, 0), (5, 0), (5, 4), (2, 4), (2, 2), (0, 2), (0, 0)])
    right_geom = Polygon([(1, 1), (6, 1), (6, 5), (3, 5), (3, 3), (1, 3), (1, 1)])
    area_geom = left_geom.intersection(right_geom)
    left_source = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries([left_geom], residency=Residency.DEVICE))
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries([right_geom], residency=Residency.DEVICE))
    )
    area_owned = from_shapely_geometries([area_geom], residency=Residency.DEVICE)

    dropped = overlay_module._device_count_dropped_polygon_intersection_warning_rows(
        area_owned,
        np.asarray([True], dtype=bool),
        np.asarray([0], dtype=np.intp),
        left_source=left_source,
        right_source=right_source,
        left_rows=np.asarray([0], dtype=np.intp),
        right_rows=np.asarray([0], dtype=np.intp),
    )

    assert dropped == 0


def test_device_count_dropped_polygon_intersection_warning_rows_classifies_large_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    rows = 129
    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [box(float(i), 0.0, float(i + 1), 1.0) for i in range(rows)],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [box(float(i + 1), 0.0, float(i + 2), 1.0) for i in range(rows)],
                residency=Residency.DEVICE,
            )
        )
    )
    area_owned = from_shapely_geometries([None] * rows, residency=Residency.DEVICE)

    boundary_module = importlib.import_module("vibespatial.constructive.boundary")
    monkeypatch.setattr(
        boundary_module,
        "boundary_owned",
        lambda *_args, **_kwargs: pytest.fail(
            "bounded warning rows should use the exact rowset classifier"
        ),
    )

    dropped = overlay_module._device_count_dropped_polygon_intersection_warning_rows(
        area_owned,
        np.zeros(rows, dtype=bool),
        np.arange(rows, dtype=np.intp),
        left_source=left_source,
        right_source=right_source,
        left_rows=np.arange(rows, dtype=np.intp),
        right_rows=np.arange(rows, dtype=np.intp),
    )

    assert dropped == rows


def test_device_warning_count_uses_candidate_relations_for_wide_touch_rows() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_geoms = [
        Point(0.0, 0.0).buffer(1.0, quad_segs=80),
        Point(10.0, 0.0).buffer(1.0, quad_segs=80),
    ]
    right_geoms = [
        Point(2.0, 0.0).buffer(1.0, quad_segs=80),
        Point(12.0, 0.0).buffer(1.0, quad_segs=80),
    ]
    left_source = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries(left_geoms, residency=Residency.DEVICE))
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries(right_geoms, residency=Residency.DEVICE))
    )
    area_owned = from_shapely_geometries([None, None], residency=Residency.DEVICE)

    vibespatial.clear_dispatch_events()
    dropped = overlay_module._device_count_dropped_polygon_intersection_warning_rows(
        area_owned,
        np.zeros(2, dtype=bool),
        np.arange(2, dtype=np.intp),
        left_source=left_source,
        right_source=right_source,
        left_rows=np.arange(2, dtype=np.intp),
        right_rows=np.arange(2, dtype=np.intp),
    )
    events = vibespatial.get_dispatch_events(clear=True)

    assert dropped == 2
    assert any(
        event.implementation == "polygon_pair_warning_candidate_remnants_gpu" for event in events
    )


def test_clear_device_exact_keep_geom_type_warnings_only_checks_kept_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_rows: list[np.ndarray] = []

    def _fake_cover_mask(
        left_source,
        right_source,
        left_rows,
        right_rows,
        warning_rows: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        observed_rows.append(np.asarray(warning_rows, dtype=np.intp))
        return np.asarray([True, False], dtype=bool)

    monkeypatch.setattr(
        overlay_module,
        "_device_polygon_keep_geom_type_cover_mask",
        _fake_cover_mask,
    )

    warning_mask = np.asarray([True, True, True], dtype=bool)
    keep_mask = np.asarray([False, True, True], dtype=bool)

    updated_mask, warning_rows = overlay_module._clear_device_exact_keep_geom_type_warnings(
        warning_mask,
        keep_mask,
        left_source=None,
        right_source=None,
        left_rows=np.arange(3, dtype=np.intp),
        right_rows=np.zeros(3, dtype=np.intp),
    )

    assert [rows.tolist() for rows in observed_rows] == [[1, 2]]
    assert updated_mask.tolist() == [True, False, True]
    assert warning_rows.tolist() == [0, 2]


def test_device_polygon_keep_geom_type_cover_mask_uses_broadcast_right() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(2, 0, 3, 1),
                    box(4, 0, 5, 1),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [box(-1, -1, 10, 10)],
                residency=Residency.DEVICE,
            )
        )
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    vibespatial.clear_dispatch_events()
    cover_mask = overlay_module._device_polygon_keep_geom_type_cover_mask(
        left_source,
        right_source,
        np.arange(3, dtype=np.intp),
        np.zeros(3, dtype=np.intp),
        np.arange(3, dtype=np.intp),
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    dispatch_events = vibespatial.get_dispatch_events(clear=True)

    assert np.asarray(cover_mask, dtype=bool).tolist() == [True, True, True]
    assert not any(
        "cover area-candidate mask host boundary" in reason for reason in runtime_reasons
    )
    assert "overlay keep-geometry-type cover classification mask host boundary" in runtime_reasons
    assert any(
        event.implementation == "fused_multi_predicate_expression_gpu"
        and event.operation == "covered_by"
        and "workload_shape=broadcast_right_de9im" in event.detail
        for event in dispatch_events
    )


def test_keep_geom_type_filter_many_vs_one_metadata_skips_covered_by_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    monkeypatch.setattr(
        overlay_module,
        "_device_polygon_keep_geom_type_cover_mask",
        lambda *_args, **_kwargs: pytest.fail(
            "many-vs-one containment metadata should skip the advisory cover probe"
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_device_count_dropped_polygon_intersection_warning_rows",
        lambda *_args, **_kwargs: 0,
    )

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [box(0, 0, 4, 4)],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [box(2, 0, 8, 4)],
                residency=Residency.DEVICE,
            )
        )
    )
    area_owned = from_shapely_geometries(
        [box(2, 0, 4, 4)],
        residency=Residency.DEVICE,
    )
    area_owned._polygon_rect_boundary_overlap = np.asarray([True], dtype=bool)
    area_owned._many_vs_one_left_containment_bypass_applied = True
    area_pairs = GeoSeries(GeometryArray.from_owned(area_owned))

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.asarray([0], dtype=np.intp),
            right_rows=np.asarray([0], dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True]
    assert dropped == 0
    assert len(filtered) == 1


def test_exact_keep_mask_keeps_geometry_collection_rows_with_polygon_parts() -> None:
    left_values = np.asarray([box(0, 0, 2, 2)], dtype=object)
    right_values = np.asarray([box(1, 1, 3, 3)], dtype=object)
    exact_values = np.asarray(
        [GeometryCollection([box(1, 1, 2, 2), LineString([(1, 1), (2, 1)])])],
        dtype=object,
    )

    keep_mask, dropped, returned_exact_values = (
        overlay_module._exact_keep_mask_and_dropped_count_for_polygon_intersection_warning_rows(
            left_values,
            right_values,
            exact_values=exact_values,
        )
    )

    assert keep_mask.tolist() == [True]
    assert dropped == 1
    assert returned_exact_values[0].equals(exact_values[0])


def test_keep_geom_type_filter_reuses_cached_exact_intersection_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            box(2, 0, 3, 1),
        ],
        residency=Residency.HOST,
    )
    owned._exact_intersection_values = np.asarray(
        [
            GeometryCollection([box(0, 0, 1, 1), LineString([(0, 0), (1, 0)])]),
            GeometryCollection([box(2, 0, 3, 1), Point(2, 0)]),
        ],
        dtype=object,
    )
    owned._exact_intersection_value_mask = np.asarray([True, True], dtype=bool)
    area_pairs = GeoSeries(GeometryArray.from_owned(owned))

    monkeypatch.setattr(
        overlay_module,
        "_warning_candidate_mask_for_polygon_keep_geom_type",
        lambda *_args, **_kwargs: np.asarray([True, True], dtype=bool),
    )
    monkeypatch.setattr(
        shapely,
        "intersection",
        lambda *_args, **_kwargs: pytest.fail(
            "cached exact intersection values should avoid a second shapely.intersection rebuild"
        ),
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            GeoSeries([box(-1, -1, 2, 2), box(1, -1, 4, 2)]),
            GeoSeries([box(0, 0, 1, 1), box(2, 0, 3, 1)]),
            area_pairs,
            keep_geom_type_warning=True,
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 2
    assert len(filtered) == 2


def test_keep_geom_type_filter_uses_cached_exact_values_for_warning_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            box(2, 0, 3, 1),
        ],
        residency=Residency.HOST,
    )
    owned._exact_intersection_values = np.asarray(
        [
            GeometryCollection([box(0, 0, 1, 1), LineString([(0, 0), (1, 0)])]),
            box(2, 0, 3, 1),
        ],
        dtype=object,
    )
    owned._exact_intersection_value_mask = np.asarray([True, True], dtype=bool)
    area_pairs = GeoSeries(GeometryArray.from_owned(owned))

    monkeypatch.setattr(
        overlay_module,
        "_take_geoseries_object_values",
        lambda *_args, **_kwargs: pytest.fail(
            "fully cached exact intersections should avoid left/right object materialization"
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_warning_candidate_mask_for_polygon_keep_geom_type",
        lambda *_args, **_kwargs: pytest.fail(
            "fully cached exact intersections should avoid boundary-overlap warning probing"
        ),
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 1
    assert len(filtered) == 2


def test_keep_geom_type_filter_rect_overlap_cached_exact_values_skip_source_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            box(2, 0, 3, 1),
        ],
        residency=Residency.HOST,
    )
    owned._polygon_rect_boundary_overlap = np.asarray([True, True], dtype=bool)
    owned._exact_intersection_values = np.asarray(
        [
            GeometryCollection([box(0, 0, 1, 1), LineString([(0, 0), (1, 0)])]),
            box(2, 0, 3, 1),
        ],
        dtype=object,
    )
    owned._exact_intersection_value_mask = np.asarray([True, True], dtype=bool)
    area_pairs = GeoSeries(GeometryArray.from_owned(owned))
    left_source = GeoSeries([box(-1, -1, 2, 2), box(1, -1, 4, 2)])
    right_source = GeoSeries([box(0, 0, 1, 1), box(2, 0, 3, 1)])

    monkeypatch.setattr(
        overlay_module,
        "_take_geoseries_object_values",
        lambda *_args, **_kwargs: pytest.fail(
            "fully cached exact warning rows should avoid source object materialization"
        ),
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(2, dtype=np.intp),
            right_rows=np.arange(2, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 1
    assert len(filtered) == 2


def test_repair_invalid_polygon_output_rows_repairs_rect_boundary_spikes() -> None:
    invalid = Polygon(
        [
            (680.0, 390.0),
            (680.0, 400.0),
            (680.0, 400.0),
            (680.0, 391.929775259655),
            (679.7726140184767, 390.0),
            (680.0, 390.0),
        ]
    )
    valid = box(0.0, 0.0, 1.0, 1.0)
    owned = from_shapely_geometries(
        [invalid, valid],
        residency=Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST,
    )
    owned._polygon_rect_boundary_overlap = np.asarray([True, False], dtype=bool)
    geometries = GeoSeries(GeometryArray.from_owned(owned))

    repaired = overlay_module._repair_invalid_polygon_output_rows(geometries)

    assert bool(shapely.is_valid(repaired.iloc[0]))
    assert shapely.equals(repaired.iloc[0], shapely.make_valid(invalid))
    assert shapely.equals(repaired.iloc[1], valid)
    assert np.asarray(
        getattr(repaired.values, "_polygon_rect_boundary_overlap", None),
        dtype=bool,
    ).tolist() == [True, False]


def test_repair_invalid_polygon_output_rows_exports_sparse_device_overlap_rows() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    invalid = Polygon(
        [
            (680.0, 390.0),
            (680.0, 400.0),
            (680.0, 400.0),
            (680.0, 391.929775259655),
            (679.7726140184767, 390.0),
            (680.0, 390.0),
        ]
    )
    geometries = [box(float(i), 0.0, float(i) + 0.5, 0.5) for i in range(24)]
    geometries[7] = invalid
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    d_overlap = cp.zeros(len(geometries), dtype=cp.bool_)
    d_overlap[7] = True
    owned._polygon_rect_boundary_overlap = d_overlap
    series = GeoSeries(GeometryArray.from_owned(owned))

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    repaired = overlay_module._repair_invalid_polygon_output_rows(series)
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert "overlay rectangle-overlap repair rows host boundary" in reasons
    assert "overlay rectangle-overlap repair mask host boundary" not in reasons
    assert bool(shapely.is_valid(repaired.iloc[7]))
    assert np.asarray(
        getattr(repaired.values, "_polygon_rect_boundary_overlap", None),
        dtype=bool,
    ).tolist() == [i == 7 for i in range(len(geometries))]


def test_repair_invalid_polygon_output_rows_small_batch_falls_back_without_overlap_mask() -> None:
    invalid = Polygon(
        [
            (680.0, 390.0),
            (680.0, 400.0),
            (680.0, 400.0),
            (680.0, 391.929775259655),
            (679.7726140184767, 390.0),
            (680.0, 390.0),
        ]
    )
    geometries = GeoSeries([invalid])

    repaired = overlay_module._repair_invalid_polygon_output_rows(geometries)

    assert bool(shapely.is_valid(repaired.iloc[0]))
    assert shapely.equals(repaired.iloc[0], shapely.make_valid(invalid))


def test_repair_invalid_polygon_output_rows_small_batch_falls_back_when_overlap_mask_is_empty() -> (
    None
):
    invalid = Polygon(
        [
            (680.0, 390.0),
            (680.0, 400.0),
            (680.0, 400.0),
            (680.0, 391.929775259655),
            (679.7726140184767, 390.0),
            (680.0, 390.0),
        ]
    )
    owned = from_shapely_geometries(
        [invalid],
        residency=Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST,
    )
    owned._polygon_rect_boundary_overlap = np.zeros(1, dtype=bool)
    geometries = GeoSeries(GeometryArray.from_owned(owned))

    repaired = overlay_module._repair_invalid_polygon_output_rows(geometries)

    assert bool(shapely.is_valid(repaired.iloc[0]))
    assert shapely.equals(repaired.iloc[0], shapely.make_valid(invalid))


def test_repair_invalid_polygon_output_rows_owned_valid_batch_skips_host_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), box(2.0, 2.0, 3.0, 3.0)],
        residency=Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST,
    )
    geometries = GeoSeries(GeometryArray.from_owned(owned))

    import vibespatial.geometry.host_bridge as host_bridge_module

    def _fail(*_args, **_kwargs):
        raise AssertionError("valid owned rows should not materialize through host_bridge")

    monkeypatch.setattr(host_bridge_module, "owned_to_shapely", _fail)

    repaired = overlay_module._repair_invalid_polygon_output_rows(geometries)

    assert repaired is geometries


def test_overlay_make_valid_owned_rewrap_failure_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibespatial.constructive import make_valid_pipeline as make_valid_pipeline_module

    left = GeoDataFrame(
        {"left": [0]},
        geometry=GeoSeries(
            GeometryArray.from_owned(
                from_shapely_geometries(
                    [Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])],
                    residency=Residency.HOST,
                )
            )
        ),
    )
    right = GeoDataFrame(
        {"right": [0]},
        geometry=GeoSeries(
            GeometryArray.from_owned(
                from_shapely_geometries(
                    [box(-1, -1, 3, 3)],
                    residency=Residency.HOST,
                )
            )
        ),
    )

    fallback_owned = from_shapely_geometries(
        [box(0, 0, 1, 1)],
        residency=Residency.HOST,
    )
    fallback_result = make_valid_pipeline_module.MakeValidResult(
        row_count=1,
        valid_rows=np.asarray([False], dtype=bool),
        repaired_rows=np.asarray([0], dtype=np.intp),
        null_rows=np.asarray([False], dtype=bool),
        method="test",
        keep_collapsed=True,
        owned=fallback_owned,
        selected=ExecutionMode.CPU,
    )

    original_make_valid_owned = make_valid_pipeline_module.make_valid_owned
    make_valid_calls = 0

    def _wrapped_make_valid_owned(*args, **kwargs):
        nonlocal make_valid_calls
        make_valid_calls += 1
        if make_valid_calls == 1:
            return fallback_result
        return original_make_valid_owned(*args, **kwargs)

    monkeypatch.setattr(
        make_valid_pipeline_module,
        "make_valid_owned",
        _wrapped_make_valid_owned,
    )

    original_from_owned = GeometryArray.from_owned
    from_owned_calls = 0

    def _wrapped_from_owned(owned, *args, **kwargs):
        nonlocal from_owned_calls
        if owned is fallback_owned:
            from_owned_calls += 1
            raise NotImplementedError("test rewrap failure")
        return original_from_owned(owned, *args, **kwargs)

    monkeypatch.setattr(GeometryArray, "from_owned", _wrapped_from_owned)

    original_to_shapely = fallback_owned.to_shapely
    materialized = False

    def _wrapped_to_shapely(*args, **kwargs):
        nonlocal materialized
        materialized = True
        return original_to_shapely(*args, **kwargs)

    monkeypatch.setattr(fallback_owned, "to_shapely", _wrapped_to_shapely)

    vibespatial.clear_fallback_events()
    with pytest.raises(NotImplementedError, match="test rewrap failure"):
        with strict_native_environment():
            overlay(left, right, how="intersection")

    assert from_owned_calls == 1
    assert materialized is False
    assert vibespatial.get_fallback_events(clear=True) == []


def test_make_valid_geoseries_uses_seeded_validity_cache_without_recompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validity_module = importlib.import_module("vibespatial.constructive.validity")

    owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), box(2.0, 2.0, 3.0, 3.0)],
        residency=Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST,
    )
    owned._cached_is_valid_mask = np.ones(owned.row_count, dtype=bool)
    geometries = GeoSeries(GeometryArray.from_owned(owned))

    monkeypatch.setattr(
        validity_module,
        "plan_dispatch_selection",
        lambda *args, **kwargs: pytest.fail(
            "seeded validity cache should bypass a fresh is_valid dispatch"
        ),
    )

    repaired = overlay_module._make_valid_geoseries(
        geometries,
        dispatch_mode=ExecutionMode.GPU if vibespatial.has_gpu_runtime() else ExecutionMode.AUTO,
    )

    assert repaired is geometries


def test_candidate_rows_all_valid_uses_owned_validity_cache_without_recompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validity_module = importlib.import_module("vibespatial.constructive.validity")

    owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), box(2.0, 2.0, 3.0, 3.0)],
        residency=Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST,
    )
    owned._cached_is_valid_mask = np.ones(owned.row_count, dtype=bool)
    geometries = GeoSeries(GeometryArray.from_owned(owned))

    monkeypatch.setattr(
        validity_module,
        "is_valid_owned",
        lambda *args, **kwargs: pytest.fail(
            "candidate validity gate should use the source owned validity cache"
        ),
    )

    assert overlay_module._candidate_rows_all_valid(
        geometries,
        np.asarray([1, 0], dtype=np.int32),
    )


def test_candidate_rows_all_valid_skips_recompute_for_device_rectangles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    validity_module = importlib.import_module("vibespatial.constructive.validity")

    owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), box(2.0, 2.0, 3.0, 3.0)],
        residency=Residency.DEVICE,
    )
    geometries = GeoSeries(GeometryArray.from_owned(owned))

    monkeypatch.setattr(
        validity_module,
        "is_valid_owned",
        lambda *args, **kwargs: pytest.fail(
            "dense device rectangles are valid without a generic OGC validity scan"
        ),
    )

    assert overlay_module._candidate_rows_all_valid(
        geometries,
        np.asarray([1, 0], dtype=np.int32),
    )


def test_candidate_rows_all_valid_uses_device_validity_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    validity_module = importlib.import_module("vibespatial.constructive.validity")
    bowtie = Polygon([(0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0), (0.0, 0.0)])
    owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            bowtie,
            None,
        ],
        residency=Residency.DEVICE,
    )
    geometries = GeoSeries(GeometryArray.from_owned(owned))

    monkeypatch.setattr(
        validity_module,
        "is_valid_owned",
        lambda *args, **kwargs: pytest.fail(
            "candidate validity gate should reduce the native validity expression"
        ),
    )

    def _fail_host_metadata(_self):
        raise AssertionError("candidate validity should not materialize host metadata")

    monkeypatch.setattr(type(owned), "_ensure_host_metadata", _fail_host_metadata)

    reset_d2h_transfer_count()
    assert overlay_module._candidate_rows_all_valid(
        geometries,
        np.asarray([0, 2], dtype=np.int32),
    )
    assert not overlay_module._candidate_rows_all_valid(
        geometries,
        np.asarray([1], dtype=np.int32),
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert not any(reason.startswith("owned geometry host metadata") for reason in reasons)
    assert "overlay candidate validity scalar fence" in reasons


def test_series_all_polygons_uses_device_family_flags_for_indexed_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    base = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(2.0, 2.0, 3.0, 3.0),
        ],
        residency=Residency.DEVICE,
    )
    indexed = OwnedGeometryArray._indexed_view(
        base,
        cp.asarray([1, 0], dtype=cp.int64),
    )
    assert indexed.is_indexed_view
    geometries = GeoSeries(GeometryArray.from_owned(indexed))

    def _fail_host_metadata(_self):
        raise AssertionError("polygon-domain query should not materialize host metadata")

    monkeypatch.setattr(type(indexed), "_ensure_host_metadata", _fail_host_metadata)

    reset_d2h_transfer_count()
    assert overlay_module._series_all_polygons(geometries)
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert not any(reason.startswith("owned geometry host metadata") for reason in reasons)
    assert "overlay source geometry family-domain scalar fence" not in reasons

    reset_d2h_transfer_count()
    assert overlay_module._series_first_geom_type(geometries) == "Polygon"
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    assert "overlay source geometry type scalar fence" not in reasons


def test_series_all_polygons_uses_logical_rows_not_unreferenced_physical_families() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.geometry.owned import FAMILY_TAGS, build_device_resident_owned

    physical = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            MultiPolygon([box(10.0, 10.0, 11.0, 11.0)]),
        ],
        residency=Residency.DEVICE,
    )
    physical_state = physical._ensure_device_state(preserve_indexed_view=True)
    logical = build_device_resident_owned(
        device_families=physical_state.families,
        row_count=1,
        tags=cp.asarray([FAMILY_TAGS[GeometryFamily.POLYGON]], dtype=cp.int8),
        validity=cp.ones(1, dtype=cp.bool_),
        family_row_offsets=cp.zeros(1, dtype=cp.int32),
        execution_mode="gpu",
    )
    logical_state = logical._ensure_device_state(preserve_indexed_view=True)
    logical_state.trusted_polygonal_only = None
    logical_state.trusted_family_domain = None
    logical_state.trusted_all_valid = None
    geometries = GeoSeries(GeometryArray.from_owned(logical))

    assert sum(buffer.row_count for buffer in logical.families.values()) == 2
    assert overlay_module._series_all_polygons(geometries)
    assert logical_state.trusted_polygonal_only is True


def test_few_right_partitioned_intersection_has_no_scalar_validity_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    left = from_shapely_geometries(
        [
            box(0.0, 0.0, 2.0, 2.0),
            box(2.0, 0.0, 4.0, 2.0),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            box(1.0, 0.5, 1.5, 1.5),
            box(2.5, 0.5, 3.5, 1.5),
        ],
        residency=Residency.DEVICE,
    )

    def _fail_host_metadata(_self):
        raise AssertionError("few-right SH gate should not materialize host metadata")

    monkeypatch.setattr(type(left), "_ensure_host_metadata", _fail_host_metadata)

    reset_d2h_transfer_count()
    result = overlay_module._few_right_partitioned_polygon_intersection_owned(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result is not None
    assert result.row_count == left.row_count
    assert not any(reason.startswith("owned geometry host metadata") for reason in reasons)
    assert "overlay few-right non-null all scalar fence" not in reasons
    assert "overlay few-right non-null any scalar fence" not in reasons
    assert "overlay few-right non-null sparse rows host boundary" not in reasons
    assert "overlay few-right sh result validity scalar fence" not in reasons


def test_overlay_intersection_seeds_polygon_validity_cache_on_owned_result() -> None:
    residency = Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST
    left = GeoDataFrame(
        {"left": [1]},
        geometry=GeoSeries(
            GeometryArray.from_owned(
                from_shapely_geometries([box(0.0, 0.0, 4.0, 4.0)], residency=residency)
            )
        ),
    )
    right = GeoDataFrame(
        {"right": [2]},
        geometry=GeoSeries(
            GeometryArray.from_owned(
                from_shapely_geometries([box(1.0, 1.0, 3.0, 3.0)], residency=residency)
            )
        ),
    )

    result = overlay(left, right, how="intersection")

    owned = getattr(result.geometry.values, "_owned", None)
    assert owned is not None
    assert owned._cached_is_valid_mask is not None
    np.testing.assert_array_equal(
        owned._cached_is_valid_mask,
        np.ones(len(result), dtype=bool),
    )
    assert result.geometry.iloc[0].equals(box(1.0, 1.0, 3.0, 3.0))


def test_strip_non_polygon_collection_parts_removes_repair_line_artifacts() -> None:
    invalid = Polygon(
        [
            (680.0, 390.0),
            (680.0, 400.0),
            (680.0, 400.0),
            (680.0, 391.929775259655),
            (679.7726140184767, 390.0),
            (680.0, 390.0),
        ]
    )

    repaired = shapely.make_valid(invalid)
    stripped = overlay_module._strip_non_polygon_collection_parts(
        np.asarray([repaired], dtype=object)
    )[0]

    assert stripped.geom_type in {"Polygon", "MultiPolygon"}
    assert shapely.equals(
        stripped,
        Polygon(
            [
                (680.0, 390.0),
                (679.7726140184767, 390.0),
                (680.0, 391.929775259655),
                (680.0, 390.0),
            ]
        ),
    )


def test_keep_geom_type_filter_falls_back_when_owned_family_empty_mask_is_inconsistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(0, 0, 1, 1),
                    box(0, 0, 1, 1),
                ]
            )
        )
    )
    right_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(1, 0, 2, 1),
                    box(2, 2, 3, 3),
                ]
            )
        )
    )
    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    LineString([(1, 0), (1, 1)]),
                    None,
                ]
            )
        )
    )

    left_values = np.asarray(left_pairs.array, dtype=object)
    right_values = np.asarray(right_pairs.array, dtype=object)
    area_values = np.asarray(area_pairs.array, dtype=object)
    area_owned = getattr(area_pairs.values, "_owned", None)
    assert area_owned is not None
    object.__setattr__(
        area_owned.families[GeometryFamily.POLYGON],
        "empty_mask",
        np.empty(0, dtype=bool),
    )

    def _object_values(series: GeoSeries) -> np.ndarray:
        if series is left_pairs:
            return left_values
        if series is right_pairs:
            return right_values
        if series is area_pairs:
            return area_values
        raise AssertionError("unexpected GeoSeries passed to object-value helper")

    def _take_object_values(series: GeoSeries, rows: np.ndarray) -> np.ndarray:
        return _object_values(series)[rows]

    monkeypatch.setattr(overlay_module, "_geoseries_object_values", _object_values)
    monkeypatch.setattr(overlay_module, "_take_geoseries_object_values", _take_object_values)

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            left_pairs,
            right_pairs,
            area_pairs,
            keep_geom_type_warning=True,
        )
    )

    assert keep_mask.tolist() == [True, False, False]
    assert dropped == 1
    assert len(filtered) == 1


def test_overlay_intersection_keep_geom_type_warning_uses_source_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"left": [0, 1]},
        geometry=GeoSeries(
            [
                box(0, 0, 1, 1),
                box(1, 1, 3, 3).union(box(1, 3, 5, 5)),
            ]
        ),
    )
    right = GeoDataFrame(
        {"right": [0, 1]},
        geometry=GeoSeries(
            [
                box(0, 0, 1, 1),
                box(3, 1, 4, 2).union(box(4, 1, 5, 4)),
            ]
        ),
    )

    real_filter = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type
    observed: dict[str, object] = {}

    def _wrapped_filter(left_pairs, right_pairs, area_pairs, **kwargs):
        observed["left_pairs"] = left_pairs
        observed["right_pairs"] = right_pairs
        observed["left_source"] = kwargs.get("left_source")
        observed["right_source"] = kwargs.get("right_source")
        observed["left_rows"] = kwargs.get("left_rows")
        observed["right_rows"] = kwargs.get("right_rows")
        return real_filter(left_pairs, right_pairs, area_pairs, **kwargs)

    monkeypatch.setattr(
        overlay_module,
        "_filter_polygon_intersection_rows_for_keep_geom_type",
        _wrapped_filter,
    )

    with strict_native_environment():
        with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay"):
            overlay(left, right, how="intersection", keep_geom_type=None)

    assert observed["left_pairs"] is None
    assert observed["right_pairs"] is None
    assert observed["left_source"].equals(left.geometry)
    assert observed["right_source"].equals(right.geometry)
    assert hasattr(observed["left_rows"], "__cuda_array_interface__")
    assert hasattr(observed["right_rows"], "__cuda_array_interface__")
    cp = pytest.importorskip("cupy")
    assert np.issubdtype(cp.asarray(observed["left_rows"]).dtype, np.integer)
    assert np.issubdtype(cp.asarray(observed["right_rows"]).dtype, np.integer)


def test_overlay_intersection_many_vs_one_remainder_prefers_direct_row_isolated_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2, 3]},
        geometry=GeoSeries(
            [
                Polygon([(-1, 1), (2, 1), (2, 4), (-1, 4), (-1, 1)]),
                Polygon([(1, -1), (4, -1), (4, 2), (1, 2), (1, -1)]),
                Polygon([(3, 3), (6, 3), (6, 6), (3, 6), (3, 3)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"zone": [1]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6), (0, 0)]),
            ]
        ),
    )

    vibespatial.clear_dispatch_events()
    with strict_native_environment():
        result = overlay(left, right, how="intersection", keep_geom_type=True)
    events = vibespatial.get_dispatch_events(clear=True)

    assert list(result["col1"]) == [1, 2]
    assert any(
        event.surface == "geopandas.overlay"
        and event.operation == "overlay_intersection"
        and event.implementation == "owned_dispatch"
        and getattr(getattr(event, "selected", None), "value", None) == "gpu"
        for event in events
    )
    assert not any(
        event.surface == "geopandas.overlay" and event.implementation == "shapely_host"
        for event in events
    )


def test_overlay_intersection_many_vs_one_remainder_avoids_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2, 3]},
        geometry=GeoSeries(
            [
                Polygon([(-1, 1), (2, 1), (2, 4), (-1, 4), (-1, 1)]),
                Polygon([(1, -1), (4, -1), (4, 2), (1, 2), (1, -1)]),
                Polygon([(3, 3), (6, 3), (6, 6), (3, 6), (3, 3)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"zone": [1]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6), (0, 0)]),
            ]
        ),
    )

    original = overlay_module.record_fallback_event

    def _wrapped_record_fallback_event(*args, **kwargs):
        reason = kwargs.get("reason", "")
        if "many-vs-one remainder" in str(reason):
            pytest.fail("many-vs-one polygon remainder should stay on GPU")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        overlay_module,
        "record_fallback_event",
        _wrapped_record_fallback_event,
    )

    result = overlay(left, right, how="intersection", keep_geom_type=True)

    assert list(result["col1"]) == [1, 2]


def test_overlay_intersection_many_vs_one_auto_keeps_public_path_on_gpu() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2, 3]},
        geometry=GeoSeries(
            [
                Polygon([(-1, 1), (2, 1), (2, 4), (-1, 4), (-1, 1)]),
                Polygon([(1, -1), (4, -1), (4, 2), (1, 2), (1, -1)]),
                Polygon([(3, 3), (6, 3), (6, 6), (3, 6), (3, 3)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"zone": [1]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6), (0, 0)]),
            ]
        ),
    )

    vibespatial.clear_dispatch_events()
    result = overlay(left, right, how="intersection", keep_geom_type=True)
    events = vibespatial.get_dispatch_events(clear=True)

    assert list(result["col1"]) == [1, 2]
    assert not any(
        event.surface == "vibespatial.predicates.binary" and event.operation == "covered_by"
        for event in events
    )
    assert any(
        event.surface == "geopandas.overlay"
        and event.operation == "overlay_intersection"
        and event.implementation == "owned_dispatch"
        and getattr(getattr(event, "selected", None), "value", None) == "gpu"
        for event in events
    )
    assert not any(
        event.surface == "geopandas.overlay" and event.implementation == "shapely_host"
        for event in events
    )


def test_overlay_intersection_polygonal_geometrycollection_mask_matches_polygon_mask() -> None:
    left_geometry = DeviceGeometryArray._from_owned(
        from_shapely_geometries(
            [
                Polygon([(-1, 1), (2, 1), (2, 4), (-1, 4), (-1, 1)]),
                Polygon([(1, -1), (4, -1), (4, 2), (1, 2), (1, -1)]),
                Polygon([(3, 3), (6, 3), (6, 6), (3, 6), (3, 3)]),
            ],
            residency=Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST,
        )
    )
    left = GeoDataFrame(
        {"col1": [1, 2, 3]},
        geometry=left_geometry,
    )
    polygon_mask = Polygon([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6), (0, 0)])
    right_polygon = GeoDataFrame(
        {"zone": [1]},
        geometry=GeoSeries([polygon_mask]),
    )
    right_collection = GeoDataFrame(
        {"zone": [1]},
        geometry=GeoSeries(
            [
                GeometryCollection(
                    [
                        polygon_mask,
                        MultiLineString(
                            [[(0, 0), (6, 0)], [(2, 2), (2, 6)]],
                        ),
                    ]
                )
            ]
        ),
    )

    vibespatial.clear_fallback_events()
    result = overlay(left, right_collection, how="intersection")
    events = vibespatial.get_fallback_events(clear=True)
    expected = overlay(left, right_polygon, how="intersection")

    result = result.sort_values("col1").reset_index(drop=True)
    expected = expected.sort_values("col1").reset_index(drop=True)

    assert_geodataframe_equal(result, expected)
    assert not any(
        event.surface == "DeviceGeometryArray.intersection"
        and "unsupported other type for owned constructive path" in str(event.reason)
        for event in events
    )


def test_overlay_keep_geom_type_rejects_source_collection_but_normalizes_mask() -> None:
    polygon = box(0, 0, 2, 2)
    line = LineString([(0, 0), (2, 0)])
    collection_mask = GeometryCollection([polygon, line])
    left_collection = GeoDataFrame({"name": ["source"]}, geometry=[collection_mask])
    right_polygon = GeoDataFrame({"mask": [1]}, geometry=[box(1, 1, 3, 3)])

    with pytest.raises(TypeError):
        overlay(left_collection, right_polygon, how="intersection", keep_geom_type=True)

    left_collection_late = GeoDataFrame(
        {"name": ["polygon", "collection"]},
        geometry=[polygon, collection_mask],
    )
    late_result = overlay(
        left_collection_late,
        right_polygon,
        how="intersection",
        keep_geom_type=True,
    )
    assert set(late_result.geometry.geom_type) <= {"Polygon"}

    left_polygon = GeoDataFrame({"name": ["source"]}, geometry=[polygon])
    right_collection = GeoDataFrame({"mask": [1]}, geometry=[collection_mask])
    result = overlay(left_polygon, right_collection, how="intersection", keep_geom_type=True)

    assert result.geometry.geom_type.tolist() == ["Polygon"]
    assert result.geometry.iloc[0].equals(polygon)


@pytest.mark.parametrize("collection_side", ["left", "right"])
def test_overlay_keep_geom_type_false_preserves_collection_lower_dimensional_parts(
    collection_side: str,
) -> None:
    polygon = box(0, 0, 2, 2)
    point = Point(5, 5)
    collection = GeometryCollection([polygon, point])
    other = box(10, 10, 12, 12)
    if collection_side == "left":
        left = GeoDataFrame({"left": [1]}, geometry=[collection])
        right = GeoDataFrame({"right": [1]}, geometry=[other])
    else:
        left = GeoDataFrame({"left": [1]}, geometry=[other])
        right = GeoDataFrame({"right": [1]}, geometry=[collection])

    result = overlay(left, right, how="union", keep_geom_type=False)
    collections = result.loc[result.geometry.geom_type == "GeometryCollection", "geometry"]

    assert len(collections) == 1
    assert shapely.equals(collections.iloc[0], collection)


def test_polygonal_collection_normalization_preserves_device_residency() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    polygon = box(0, 0, 2, 2)
    line = LineString([(0, 0), (2, 0)])
    frame = GeoDataFrame(
        {"mask": [1]},
        geometry=GeoSeries([GeometryCollection([polygon, line])]),
    )
    frame.geometry.values._owned = from_shapely_geometries(
        [polygon],
        residency=Residency.DEVICE,
    )

    normalized, changed = overlay_module._normalize_polygonal_collection_input(frame)

    assert changed
    assert normalized.geometry.dtype.name == "device_geometry"
    owned = getattr(normalized.geometry.values, "_owned", None)
    assert owned is not None
    assert owned.residency is Residency.DEVICE


def test_polygonal_collection_normalization_ignores_lower_dimensional_collections() -> None:
    collection = GeometryCollection(
        [
            LineString([(0, 0), (1, 0)]),
            Point(0, 0),
        ]
    )
    frame = GeoDataFrame({"mask": [1]}, geometry=GeoSeries([collection]))

    normalized, changed = overlay_module._normalize_polygonal_collection_input(frame)

    assert normalized is frame
    assert not changed


def test_overlay_intersection_many_vs_one_uses_one_device_capacity_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

    left = from_shapely_geometries(
        [box(float(i), 0.0, float(i) + 2.0, 2.0) for i in range(24)],
        residency=Residency.HOST,
    )
    right = from_shapely_geometries(
        [box(0.5, 0.5, 23.5, 2.5)],
        residency=Residency.HOST,
    )
    calls: list[tuple[int, int, Residency, Residency]] = []

    def _capacity_partition(left_arg, right_arg, *, right_row, dispatch_mode):
        calls.append(
            (
                left_arg.row_count,
                right_arg.row_count,
                left_arg.residency,
                right_arg.residency,
            )
        )
        assert right_row == 0
        assert dispatch_mode is ExecutionMode.GPU
        return left_arg

    monkeypatch.setattr(
        constructive_module,
        "broadcast_right_polygon_intersection_capacity_gpu",
        _capacity_partition,
    )

    result = overlay_module._many_vs_one_intersection_owned(left, right, 0)

    assert calls == [(24, 1, Residency.DEVICE, Residency.DEVICE)]
    assert result.row_count == 24


def test_overlay_intersection_many_vs_one_no_gpu_declines_before_capacity_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_sub = from_shapely_geometries(
        [box(float(i), 0.0, float(i) + 2.0, 2.0) for i in range(64)],
        residency=Residency.HOST,
    )
    right_one = from_shapely_geometries(
        [box(0.5, 0.5, 63.5, 2.5)],
        residency=Residency.HOST,
    )

    monkeypatch.setattr(overlay_module, "has_gpu_runtime", lambda: False)
    from vibespatial.constructive import binary_constructive as constructive_module

    monkeypatch.setattr(
        constructive_module,
        "broadcast_right_polygon_intersection_capacity_gpu",
        lambda *args, **kwargs: pytest.fail(
            "no-GPU many-vs-one native decline should bypass capacity partitioning"
        ),
    )

    vibespatial.clear_fallback_events()
    with pytest.raises(overlay_module._OverlayNativeConstructiveDeclined):
        overlay_module._many_vs_one_intersection_owned(
            left_sub,
            right_one,
            0,
        )
    assert vibespatial.get_fallback_events(clear=True) == []


def test_many_vs_one_exact_batch_plans_broadcast_right_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_sub = from_shapely_geometries(
        [box(float(i), 0.0, float(i) + 2.0, 2.0) for i in range(64)],
        residency=Residency.DEVICE,
    )
    right_one = from_shapely_geometries(
        [box(0.5, 0.5, 63.5, 2.5)],
        residency=Residency.DEVICE,
    )
    seen: dict[str, object] = {}

    def _fake_plan_dispatch_selection(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            selected=ExecutionMode.CPU,
            requested=ExecutionMode.AUTO,
            reason="synthetic cpu selection",
        )

    monkeypatch.setattr(
        overlay_module,
        "plan_dispatch_selection",
        _fake_plan_dispatch_selection,
    )

    result = overlay_module._many_vs_one_intersection_owned(
        left_sub,
        right_one,
        0,
    )

    assert result.row_count == left_sub.row_count
    assert seen["workload_shape"] is WorkloadShape.BROADCAST_RIGHT
    assert seen["current_residency"] is Residency.DEVICE


def test_host_exact_polygon_intersection_series_batch_preserves_exact_cache() -> None:
    left = GeoSeries(
        [
            box(0.0, 0.0, 2.0, 2.0),
            box(4.0, 0.0, 6.0, 2.0),
        ]
    )
    right = GeoSeries(
        [
            box(1.0, 0.0, 3.0, 2.0),
            box(5.0, 0.0, 7.0, 2.0),
        ]
    )

    result = overlay_module._host_exact_polygon_intersection_series_batch(
        left,
        right,
        np.asarray([0, 1], dtype=np.intp),
        np.asarray([0, 1], dtype=np.intp),
        crs=left.crs,
        requested=ExecutionMode.AUTO,
        reason="test host exact pair batch",
    )

    exact_mask = np.asarray(
        getattr(getattr(result.values, "_owned", None), "_exact_intersection_value_mask", None),
        dtype=bool,
    )
    assert exact_mask.tolist() == [True, True]
    assert result.geom_type.tolist() == ["Polygon", "Polygon"]


def test_overlay_intersection_many_vs_one_unexpected_failure_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2, 3]},
        geometry=GeoSeries(
            [
                Polygon([(-1, 1), (2, 1), (2, 4), (-1, 4), (-1, 1)]),
                Polygon([(1, -1), (4, -1), (4, 2), (1, 2), (1, -1)]),
                Polygon([(3, 3), (6, 3), (6, 6), (3, 6), (3, 3)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"zone": [1]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6), (0, 0)]),
            ]
        ),
    )

    calls = 0

    def _wrapped_many_vs_one(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise MemoryError("synthetic many-vs-one constructive failure")

    monkeypatch.setattr(
        overlay_module,
        "_many_vs_one_intersection_owned",
        _wrapped_many_vs_one,
    )

    with (
        strict_native_environment(),
        pytest.raises(
            MemoryError,
            match="synthetic many-vs-one constructive failure",
        ),
    ):
        overlay(left, right, how="intersection", keep_geom_type=True)

    assert calls == 1


def test_overlay_intersection_few_right_fast_path_batches_exact_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

    left = GeoDataFrame(
        {"col1": np.arange(24, dtype=np.int32)},
        geometry=GeoSeries([box(i, 0, i + 1, 1) for i in range(24)]),
    )
    right = GeoDataFrame(
        {"zone_type": ["A", "B", "C"]},
        geometry=GeoSeries(
            [
                Polygon(
                    [(0, 0), (16, 0), (16, 2), (0, 2)],
                    holes=[[(4, 1.25), (6, 1.25), (6, 1.75), (4, 1.75)]],
                ),
                Polygon(
                    [(8, 0), (24, 0), (24, 2), (8, 2)],
                    holes=[[(12, 1.25), (14, 1.25), (14, 1.75), (12, 1.75)]],
                ),
                Polygon(
                    [(16, 0), (32, 0), (32, 2), (16, 2)],
                    holes=[[(20, 1.25), (22, 1.25), (22, 1.75), (20, 1.75)]],
                ),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    idx1 = np.arange(24, dtype=np.int32)
    idx2 = np.repeat(np.arange(3, dtype=np.int32), 8)

    exact_calls: list[int] = []

    def _fake_rowwise_exact(
        left_arg,
        right_arg,
        *,
        dispatch_mode=ExecutionMode.GPU,
        _cached_right_segments=None,
    ):
        exact_calls.append(left_arg.row_count)
        geoms = [box(float(i) + 0.1, 0.1, float(i) + 0.9, 0.4) for i in range(left_arg.row_count)]
        return from_shapely_geometries(geoms, residency=Residency.DEVICE)

    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_exact_batch_gpu",
        _fake_rowwise_exact,
    )
    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        lambda *args, **kwargs: pytest.fail(
            "few-right public intersection should fuse exact leftovers through the rowwise helper"
        ),
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=(idx1, idx2),
    )

    assert used_owned is True
    assert exact_calls == [24]
    assert result["col1"].tolist() == idx1.tolist()
    assert result["zone_type"].tolist() == ["A"] * 8 + ["B"] * 8 + ["C"] * 8
    expected = [box(float(i) + 0.1, 0.1, float(i) + 0.9, 0.4) for i in range(24)]
    for got_geom, expected_geom in zip(result.geometry, expected, strict=True):
        assert got_geom.normalize().equals(expected_geom.normalize())


def test_rowwise_polygon_intersection_preserves_pairwise_containment_on_device() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive.binary_constructive import (
        _dispatch_polygon_intersection_overlay_exact_batch_gpu,
    )

    left_geoms = [
        box(1.0, 1.0, 2.0, 2.0),
        box(0.0, 0.0, 5.0, 5.0),
        box(8.0, 0.0, 11.0, 3.0),
        MultiPolygon([box(20.0, 0.0, 21.0, 1.0), box(22.0, 0.0, 23.0, 1.0)]),
    ]
    right_geoms = [
        box(0.0, 0.0, 3.0, 3.0),
        box(1.0, 1.0, 2.0, 2.0),
        box(9.0, -1.0, 12.0, 2.0),
        box(19.0, -1.0, 24.0, 2.0),
    ]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

    vibespatial.clear_fallback_events()
    result = _dispatch_polygon_intersection_overlay_exact_batch_gpu(left, right)

    assert result is not None
    assert result.row_count == len(left_geoms)
    got = np.asarray(GeometryArray.from_owned(result)._data, dtype=object)
    expected = shapely.intersection(
        np.asarray(left_geoms, dtype=object),
        np.asarray(right_geoms, dtype=object),
    )
    assert np.count_nonzero(~shapely.is_empty(got)) == 4
    for got_geom, expected_geom in zip(got, expected, strict=True):
        assert got_geom.normalize().equals(expected_geom.normalize())
    assert vibespatial.get_fallback_events(clear=True) == []


def test_rowwise_polygon_intersection_containment_preserves_indexed_view_carrier() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.constructive.binary_constructive import (
        _dispatch_polygon_intersection_overlay_exact_batch_gpu,
    )
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    source_geoms = [
        Polygon([(0, 0), (2, 0), (2, 1), (1, 2), (0, 1), (0, 0)]),
        Polygon([(10, 10), (13, 10), (13, 11), (12, 12), (10, 11), (10, 10)]),
        Polygon([(20, 20), (21, 20), (22, 22), (20, 21), (20, 20)]),
    ]
    ordered_left = [source_geoms[2], source_geoms[0], source_geoms[1]]
    right_geoms = [
        box(19.0, 19.0, 24.0, 24.0),
        box(-1.0, -1.0, 3.0, 3.0),
        box(9.0, 9.0, 15.0, 15.0),
    ]
    left_base = from_shapely_geometries(source_geoms, residency=Residency.DEVICE)
    left = OwnedGeometryArray._indexed_view(
        left_base,
        cp.asarray([2, 0, 1], dtype=cp.int64),
    )
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

    assert left.is_indexed_view

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = _dispatch_polygon_intersection_overlay_exact_batch_gpu(left, right)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result is not None
    assert result.row_count == len(ordered_left)
    assert left.is_indexed_view
    assert getattr(result, "_device_scatter_implementation", None) == (
        "device_capacity_selection_scatter"
    )
    assert not any("owned geometry device-take" in reason for reason in runtime_reasons)

    got = np.asarray(GeometryArray.from_owned(result)._data, dtype=object)
    expected = shapely.intersection(
        np.asarray(ordered_left, dtype=object),
        np.asarray(right_geoms, dtype=object),
    )
    for got_geom, expected_geom in zip(got, expected, strict=True):
        assert got_geom.normalize().equals(expected_geom.normalize())


def test_rowwise_polygon_intersection_keeps_indexed_view_for_exact_overlay() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.constructive.binary_constructive import (
        _dispatch_polygon_intersection_overlay_exact_batch_gpu,
    )
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    source_geoms = [
        Polygon(
            [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)],
            holes=[[(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]],
        ),
        Polygon(
            [(10, 0), (14, 0), (14, 4), (10, 4), (10, 0)],
            holes=[[(11, 1), (12, 1), (12, 2), (11, 2), (11, 1)]],
        ),
        Polygon(
            [(20, 0), (24, 0), (24, 4), (20, 4), (20, 0)],
            holes=[[(21, 1), (22, 1), (22, 2), (21, 2), (21, 1)]],
        ),
    ]
    ordered_left = [source_geoms[2], source_geoms[0], source_geoms[1]]
    right_geoms = [
        box(23.0, 0.5, 25.0, 2.5),
        box(3.0, 0.5, 5.0, 2.5),
        box(13.0, 0.5, 15.0, 2.5),
    ]
    left_base = from_shapely_geometries(source_geoms, residency=Residency.DEVICE)
    left = OwnedGeometryArray._indexed_view(
        left_base,
        cp.asarray([2, 0, 1], dtype=cp.int64),
    )
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

    assert left.is_indexed_view

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = _dispatch_polygon_intersection_overlay_exact_batch_gpu(left, right)
    events = get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result is not None
    assert result.row_count == len(ordered_left)
    assert left.is_indexed_view
    assert not any("owned geometry host metadata" in reason for reason in runtime_reasons)
    assert not any("owned geometry device-take" in reason for reason in runtime_reasons)
    assert "binary constructive polygon-family logical-domain scalar fence" not in runtime_reasons
    assert runtime_reasons.count("segment extraction total-segments allocation fence") <= 2
    assert any(
        event.implementation == "row_isolated_polygon_overlay_row_indirected_exact_gpu"
        for event in events
    )

    got = np.asarray(GeometryArray.from_owned(result)._data, dtype=object)
    expected = shapely.intersection(
        np.asarray(ordered_left, dtype=object),
        np.asarray(right_geoms, dtype=object),
    )
    assert np.count_nonzero(~shapely.is_empty(got)) == len(ordered_left)
    for got_geom, expected_geom in zip(got, expected, strict=True):
        assert got_geom.normalize().equals(expected_geom.normalize())


def test_rowwise_polygon_intersection_uses_validated_simple_carrier() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.constructive.binary_constructive import (
        _dispatch_polygon_intersection_overlay_exact_batch_gpu,
    )
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_source = [
        box(0.0, 0.0, 3.0, 3.0),
        box(10.0, 0.0, 13.0, 3.0),
    ]
    right_source = [
        Polygon([(1, -1), (4, 1), (1, 4), (2, 1), (1, -1)]),
        Polygon([(11, -1), (14, 1), (11, 4), (12, 1), (11, -1)]),
    ]
    left = OwnedGeometryArray._indexed_view(
        from_shapely_geometries(left_source, residency=Residency.DEVICE),
        cp.asarray([1, 0], dtype=cp.int64),
    )
    right = OwnedGeometryArray._indexed_view(
        from_shapely_geometries(right_source, residency=Residency.DEVICE),
        cp.asarray([1, 0], dtype=cp.int64),
    )

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = _dispatch_polygon_intersection_overlay_exact_batch_gpu(left, right)
    events = get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result is not None
    assert result.row_count == 2
    assert left.is_indexed_view
    assert right.is_indexed_view
    assert not any("owned geometry host metadata" in reason for reason in runtime_reasons)
    assert not any("owned geometry device-take" in reason for reason in runtime_reasons)
    assert any(
        event.implementation == "row_isolated_polygon_capacity_partition_gpu" for event in events
    )

    got = np.asarray(GeometryArray.from_owned(result)._data, dtype=object)
    expected = shapely.intersection(
        np.asarray([left_source[1], left_source[0]], dtype=object),
        np.asarray([right_source[1], right_source[0]], dtype=object),
    )
    for got_geom, expected_geom in zip(got, expected, strict=True):
        assert got_geom.normalize().equals_exact(
            expected_geom.normalize(),
            tolerance=1e-12,
        )


def test_rowwise_polygon_intersection_uses_logical_polygonal_proof_for_mixed_indexed_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.constructive import binary_constructive as constructive_module
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_source = [
        Point(100.0, 100.0),
        Polygon([(0, 0), (3, 0), (3, 1), (2, 3), (0, 3), (0, 0)]),
        LineString([(50.0, 0.0), (51.0, 1.0)]),
        box(10.0, 0.0, 13.0, 3.0),
        Polygon([(20, 0), (23, 0), (24, 1.5), (23, 3), (20, 3), (20, 0)]),
    ]
    right_source = [
        LineString([(60.0, 0.0), (61.0, 1.0)]),
        box(21.0, 1.0, 25.0, 4.0),
        Point(200.0, 200.0),
        box(1.0, 1.0, 4.0, 4.0),
        box(11.0, 1.0, 14.0, 4.0),
    ]
    left_rows = cp.asarray([4, 1, 3], dtype=cp.int64)
    right_rows = cp.asarray([1, 3, 4], dtype=cp.int64)
    left = OwnedGeometryArray._indexed_view(
        from_shapely_geometries(left_source, residency=Residency.DEVICE),
        left_rows,
    )
    right = OwnedGeometryArray._indexed_view(
        from_shapely_geometries(right_source, residency=Residency.DEVICE),
        right_rows,
    )

    assert left.is_indexed_view
    assert right.is_indexed_view
    assert GeometryFamily.LINESTRING in left.families
    assert GeometryFamily.POINT in right.families
    assert overlay_module._owned_logical_family_flags(left)[4]
    assert overlay_module._owned_logical_family_flags(right)[4]

    def _fail_resolve(_owned):
        raise AssertionError("logical polygonal indexed views should not be resolved")

    monkeypatch.setattr(
        constructive_module,
        "_resolve_indexed_polygon_fast_path_candidate",
        _fail_resolve,
    )

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = constructive_module._dispatch_polygon_intersection_overlay_exact_batch_gpu(
        left,
        right,
    )
    events = get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result is not None
    assert result.row_count == int(left_rows.size)
    assert left.is_indexed_view
    assert right.is_indexed_view
    assert not any("owned geometry device-take" in reason for reason in runtime_reasons)
    assert "binary constructive polygon-family logical-domain scalar fence" not in runtime_reasons
    assert runtime_reasons.count("segment extraction total-segments allocation fence") <= 2
    assert any(
        event.implementation == "row_isolated_polygon_overlay_row_indirected_exact_gpu"
        for event in events
    )

    got = np.asarray(GeometryArray.from_owned(result)._data, dtype=object)
    expected_left = [left_source[index] for index in cp.asnumpy(left_rows)]
    expected_right = [right_source[index] for index in cp.asnumpy(right_rows)]
    expected = shapely.intersection(
        np.asarray(expected_left, dtype=object),
        np.asarray(expected_right, dtype=object),
    )
    for got_geom, expected_geom in zip(got, expected, strict=True):
        assert got_geom.normalize().equals(expected_geom.normalize())


def test_rowwise_polygon_intersection_duplicate_indexed_rows_stay_row_indirected() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.constructive.binary_constructive import (
        _dispatch_polygon_intersection_overlay_exact_batch_gpu,
    )
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    source_geoms = [
        Polygon([(0, 0), (4, 0), (4, 1), (2, 3), (0, 3), (0, 0)]),
        Polygon([(10, 0), (13, 0), (14, 2), (12, 4), (10, 2), (10, 0)]),
        Polygon([(20, 0), (24, 0), (25, 3), (22, 5), (20, 3), (20, 0)]),
    ]
    pair_rows = cp.asarray([0, 1, 0, 2, 1], dtype=cp.int64)
    left_base = from_shapely_geometries(source_geoms, residency=Residency.DEVICE)
    left = OwnedGeometryArray._indexed_view(left_base, pair_rows)
    right_geoms = [
        box(1, 1, 5, 4),
        box(11, 1, 15, 5),
        box(0.5, 0.5, 2.5, 2.5),
        box(21, 1, 26, 4),
        box(10.5, 0.5, 12.5, 2.5),
    ]
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

    assert left.is_indexed_view

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = _dispatch_polygon_intersection_overlay_exact_batch_gpu(left, right)
    events = get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result is not None
    assert result.row_count == len(right_geoms)
    assert left.is_indexed_view
    assert not any("owned geometry host metadata" in reason for reason in runtime_reasons)
    assert not any("owned geometry device-take" in reason for reason in runtime_reasons)
    assert any(
        event.implementation == "row_isolated_polygon_capacity_partition_gpu" for event in events
    )

    got = np.asarray(GeometryArray.from_owned(result)._data, dtype=object)
    expected_left = [source_geoms[index] for index in [0, 1, 0, 2, 1]]
    expected = shapely.intersection(
        np.asarray(expected_left, dtype=object),
        np.asarray(right_geoms, dtype=object),
    )
    for got_geom, expected_geom in zip(got, expected, strict=True):
        assert got_geom.normalize().equals(expected_geom.normalize())


def test_overlay_intersection_native_builds_native_tabular_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    real_assembler = overlay_module._assemble_intersection_attributes
    assemble_calls = 0

    def _counting_assembler(*args, **kwargs):
        nonlocal assemble_calls
        assemble_calls += 1
        return real_assembler(*args, **kwargs)

    monkeypatch.setattr(
        overlay_module,
        "_assemble_intersection_attributes",
        _counting_assembler,
    )
    monkeypatch.setattr(
        PairwiseConstructiveResult,
        "to_geodataframe",
        lambda *_args, **_kwargs: pytest.fail(
            "overlay intersection native path should not export through PairwiseConstructiveResult"
        ),
    )

    native_result, used_owned = overlay_module._overlay_intersection_native(left, right)

    expected_result_type = (
        NativeTabularSelection if vibespatial.has_gpu_runtime() else NativeTabularResult
    )
    assert isinstance(native_result, expected_result_type)
    assert assemble_calls == 0

    materialized = native_result.to_geodataframe()
    wrapped, wrapped_used = overlay_module._overlay_intersection(left, right)

    assert assemble_calls == 0
    assert used_owned is wrapped_used
    assert_geodataframe_equal(materialized, wrapped, normalize=True, check_column_type=False)


def test_overlay_difference_native_builds_native_tabular_result() -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10]},
        geometry=GeoSeries([box(1, 1, 4, 4)]),
    )

    native_result, used_owned = overlay_module._overlay_difference_native(left, right)

    assert isinstance(native_result, NativeTabularResult)

    materialized = native_result.to_geodataframe()
    wrapped, wrapped_used = overlay_module._overlay_difference(left, right)

    assert used_owned is wrapped_used
    assert_geodataframe_equal(materialized, wrapped, normalize=True, check_column_type=False)


def test_overlay_intersection_export_result_returns_native_tabular_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    expected = overlay(left, right, how="intersection")

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "overlay intersection export path should not route through fragment lowering"
        )

    monkeypatch.setattr(
        native_results_module,
        "_pairwise_constructive_fragment_to_native_tabular_result",
        _fail,
    )
    monkeypatch.setattr(
        PairwiseConstructiveResult,
        "to_geodataframe",
        _fail,
    )

    export_result, used_owned = overlay_module._overlay_intersection_export_result(left, right)
    wrapped, wrapped_used = overlay_module._overlay_intersection(left, right)

    expected_result_type = (
        NativeTabularSelection if vibespatial.has_gpu_runtime() else NativeTabularResult
    )
    assert isinstance(export_result, expected_result_type)
    assert used_owned is wrapped_used
    assert_geodataframe_equal(
        export_result.to_geodataframe(),
        expected,
        normalize=True,
        check_column_type=False,
    )
    assert_geodataframe_equal(wrapped, expected, normalize=True, check_column_type=False)


def test_overlay_difference_export_result_returns_native_tabular_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10]},
        geometry=GeoSeries([box(1, 1, 4, 4)]),
    )

    expected, _ = overlay_module._overlay_difference(left, right)

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "overlay difference export path should not route through fragment lowering"
        )

    monkeypatch.setattr(
        native_results_module,
        "_left_constructive_fragment_to_native_tabular_result",
        _fail,
    )
    monkeypatch.setattr(
        LeftConstructiveResult,
        "to_geodataframe",
        _fail,
    )

    export_result, used_owned = overlay_module._overlay_difference_export_result(left, right)
    wrapped, wrapped_used = overlay_module._overlay_difference(left, right)

    assert isinstance(export_result, NativeTabularResult)
    assert used_owned is wrapped_used
    assert_geodataframe_equal(
        export_result.to_geodataframe(),
        expected,
        normalize=True,
        check_column_type=False,
    )
    assert_geodataframe_equal(wrapped, expected, normalize=True, check_column_type=False)


def test_overlay_identity_native_uses_direct_native_tabular_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    expected = overlay(left, right, how="identity")
    real_pairwise = overlay_module._pairwise_constructive_to_native_tabular_result
    real_left = overlay_module._left_constructive_to_native_tabular_result
    pairwise_calls = 0
    left_calls = 0

    def _count_pairwise(*args, **kwargs):
        nonlocal pairwise_calls
        pairwise_calls += 1
        return real_pairwise(*args, **kwargs)

    def _count_left(*args, **kwargs):
        nonlocal left_calls
        left_calls += 1
        return real_left(*args, **kwargs)

    def _fail(*_args, **_kwargs):
        raise AssertionError("identity native path should not depend on legacy wrapper lowering")

    monkeypatch.setattr(
        overlay_module,
        "_pairwise_constructive_to_native_tabular_result",
        _count_pairwise,
    )
    monkeypatch.setattr(
        overlay_module,
        "_left_constructive_to_native_tabular_result",
        _count_left,
    )
    monkeypatch.setattr(
        native_results_module,
        "_pairwise_constructive_fragment_to_native_tabular_result",
        _fail,
    )
    monkeypatch.setattr(
        native_results_module,
        "_left_constructive_fragment_to_native_tabular_result",
        _fail,
    )
    monkeypatch.setattr(PairwiseConstructiveResult, "to_geodataframe", _fail)
    monkeypatch.setattr(LeftConstructiveResult, "to_geodataframe", _fail)

    native_result, used_owned = overlay_module._overlay_identity_native(left, right)

    expected_result_type = (
        NativeTabularSelection if vibespatial.has_gpu_runtime() else NativeTabularResult
    )
    assert isinstance(native_result, expected_result_type)
    assert pairwise_calls == 1
    assert left_calls == 1

    materialized = native_result.to_geodataframe()
    wrapped, wrapped_used = overlay_module._overlay_identity(left, right)

    assert used_owned is wrapped_used
    assert_geodataframe_equal(materialized, expected, normalize=True, check_column_type=False)
    assert_geodataframe_equal(wrapped, expected, normalize=True, check_column_type=False)


def test_overlay_symmetric_difference_native_uses_direct_native_tabular_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    expected = overlay(left, right, how="symmetric_difference")
    real_left = overlay_module._left_constructive_to_native_tabular_result
    left_calls = 0

    def _count_left(*args, **kwargs):
        nonlocal left_calls
        left_calls += 1
        return real_left(*args, **kwargs)

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "symmetric-difference native path should not depend on legacy wrapper lowering"
        )

    monkeypatch.setattr(
        overlay_module,
        "_left_constructive_to_native_tabular_result",
        _count_left,
    )
    monkeypatch.setattr(
        native_results_module,
        "_left_constructive_fragment_to_native_tabular_result",
        _fail,
    )
    monkeypatch.setattr(
        native_results_module,
        "_symmetric_difference_constructive_result_to_native_tabular_result",
        _fail,
    )
    monkeypatch.setattr(LeftConstructiveResult, "to_geodataframe", _fail)

    native_result, used_owned = overlay_module._overlay_symmetric_diff_native(left, right)

    assert isinstance(native_result, NativeTabularResult)
    assert left_calls == 2

    materialized = native_result.to_geodataframe()
    wrapped, wrapped_used = overlay_module._overlay_symmetric_diff(left, right)

    assert used_owned is wrapped_used
    assert_geodataframe_equal(materialized, expected, normalize=True, check_column_type=False)
    assert_geodataframe_equal(wrapped, expected, normalize=True, check_column_type=False)


def test_overlay_union_native_uses_direct_native_tabular_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    expected = overlay(left, right, how="union")
    real_pairwise = overlay_module._pairwise_constructive_to_native_tabular_result
    real_left = overlay_module._left_constructive_to_native_tabular_result
    pairwise_calls = 0
    left_calls = 0

    def _count_pairwise(*args, **kwargs):
        nonlocal pairwise_calls
        pairwise_calls += 1
        return real_pairwise(*args, **kwargs)

    def _count_left(*args, **kwargs):
        nonlocal left_calls
        left_calls += 1
        return real_left(*args, **kwargs)

    def _fail(*_args, **_kwargs):
        raise AssertionError("union native path should not depend on legacy wrapper lowering")

    monkeypatch.setattr(
        overlay_module,
        "_pairwise_constructive_to_native_tabular_result",
        _count_pairwise,
    )
    monkeypatch.setattr(
        overlay_module,
        "_left_constructive_to_native_tabular_result",
        _count_left,
    )
    monkeypatch.setattr(
        native_results_module,
        "_pairwise_constructive_fragment_to_native_tabular_result",
        _fail,
    )
    monkeypatch.setattr(
        native_results_module,
        "_left_constructive_fragment_to_native_tabular_result",
        _fail,
    )
    monkeypatch.setattr(
        native_results_module,
        "_concat_constructive_result_to_native_tabular_result",
        _fail,
    )
    monkeypatch.setattr(PairwiseConstructiveResult, "to_geodataframe", _fail)
    monkeypatch.setattr(LeftConstructiveResult, "to_geodataframe", _fail)

    native_result, used_owned = overlay_module._overlay_union_native(left, right)

    expected_result_type = (
        NativeTabularSelection if vibespatial.has_gpu_runtime() else NativeTabularResult
    )
    assert isinstance(native_result, expected_result_type)
    assert pairwise_calls == 1
    assert left_calls == 2

    materialized = native_result.to_geodataframe()
    wrapped, wrapped_used = overlay_module._overlay_union(left, right)

    assert used_owned is wrapped_used
    assert_geodataframe_equal(materialized, expected, normalize=True, check_column_type=False)
    assert_geodataframe_equal(wrapped, expected, normalize=True, check_column_type=False)


def test_overlay_union_native_writes_without_fragment_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )
    expected = overlay(left, right, how="union")

    def _fail(*_args, **_kwargs):
        raise AssertionError("native union GeoParquet write should not require GeoDataFrame export")

    monkeypatch.setattr(
        native_results_module,
        "_concat_constructive_result_to_native_tabular_result",
        _fail,
    )
    monkeypatch.setattr(
        native_results_module,
        "_symmetric_difference_constructive_result_to_native_tabular_result",
        _fail,
    )
    monkeypatch.setattr(PairwiseConstructiveResult, "to_geodataframe", _fail)
    monkeypatch.setattr(LeftConstructiveResult, "to_geodataframe", _fail)
    native_result, _used_owned = overlay_module._overlay_union_native(left, right)

    path = tmp_path / "overlay-union-native.parquet"
    monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)
    write_geoparquet(native_result, path, geometry_encoding="geoarrow")
    monkeypatch.undo()

    result = geopandas.read_parquet(path)
    assert_geodataframe_equal(result, expected, normalize=True, check_column_type=False)


def test_overlay_union_native_builds_native_tabular_result() -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    native_result, _used_owned = overlay_module._overlay_union_native(left, right)
    tabular = to_native_tabular_result(native_result)

    expected_result_type = (
        NativeTabularSelection if vibespatial.has_gpu_runtime() else NativeTabularResult
    )
    assert isinstance(tabular, expected_result_type)
    assert_geodataframe_equal(
        tabular.to_geodataframe(),
        overlay(left, right, how="union"),
        normalize=True,
        check_column_type=False,
    )


def test_overlay_intersection_export_native_tabular_skips_pandas_attribute_assembler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )
    expected = overlay(left, right, how="intersection")

    real_pairwise = overlay_module._pairwise_constructive_to_native_tabular_result
    native_calls = 0

    def _counting_pairwise(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return real_pairwise(*args, **kwargs)

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "native overlay intersection tabular export should not require pandas attribute assembly"
        )

    monkeypatch.setattr(
        overlay_module,
        "_pairwise_constructive_to_native_tabular_result",
        _counting_pairwise,
    )
    monkeypatch.setattr(overlay_module, "_assemble_intersection_attributes", _fail)

    export_result, _used_owned = overlay_module._overlay_intersection_export_result(left, right)
    tabular = to_native_tabular_result(export_result)

    assert native_calls == 1
    assert isinstance(tabular, NativeTabularSelection)
    capacity_attributes = tabular.capacity_result.attributes
    assert isinstance(capacity_attributes, NativeAttributeTable)
    assert capacity_attributes.dataframe is None
    assert capacity_attributes.loader is None
    assert capacity_attributes.device_table is not None or capacity_attributes.parts is not None
    assert_geodataframe_equal(
        tabular.to_geodataframe(),
        expected,
        normalize=True,
        check_column_type=False,
    )


def test_overlay_difference_export_native_tabular_uses_arrow_backed_attributes() -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10]},
        geometry=GeoSeries([box(1, 1, 4, 4)]),
    )
    expected, _ = overlay_module._overlay_difference(left, right)

    export_result, _used_owned = overlay_module._overlay_difference_export_result(left, right)
    tabular = to_native_tabular_result(export_result)

    assert isinstance(tabular, NativeTabularResult)
    assert isinstance(tabular.attributes, NativeAttributeTable)
    assert tabular.attributes.arrow_table is not None
    assert_geodataframe_equal(
        tabular.to_geodataframe(),
        expected,
        normalize=True,
        check_column_type=False,
    )


def test_overlay_union_native_tabular_skips_pandas_attribute_assembler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )
    expected = overlay(left, right, how="union")

    real_pairwise = overlay_module._pairwise_constructive_to_native_tabular_result
    native_calls = 0

    def _counting_pairwise(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return real_pairwise(*args, **kwargs)

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "native overlay union tabular export should not require pandas attribute assembly"
        )

    monkeypatch.setattr(
        overlay_module,
        "_pairwise_constructive_to_native_tabular_result",
        _counting_pairwise,
    )
    monkeypatch.setattr(overlay_module, "_assemble_intersection_attributes", _fail)

    native_result, _used_owned = overlay_module._overlay_union_native(left, right)
    tabular = to_native_tabular_result(native_result)

    assert native_calls == 1
    expected_result_type = (
        NativeTabularSelection if vibespatial.has_gpu_runtime() else NativeTabularResult
    )
    assert isinstance(tabular, expected_result_type)
    capacity_result = (
        tabular.capacity_result if isinstance(tabular, NativeTabularSelection) else tabular
    )
    assert isinstance(capacity_result.attributes, NativeAttributeTable)
    assert capacity_result.attributes.arrow_table is not None
    assert_geodataframe_equal(
        tabular.to_geodataframe(),
        expected,
        normalize=True,
        check_column_type=False,
    )


def test_overlay_union_native_tabular_builds_arrow_without_frame_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    native_result, _used_owned = overlay_module._overlay_union_native(left, right)
    tabular = to_native_tabular_result(native_result)

    expected_result_type = (
        NativeTabularSelection if vibespatial.has_gpu_runtime() else NativeTabularResult
    )
    assert isinstance(tabular, expected_result_type)

    def _fail(*_args, **_kwargs):
        raise AssertionError("native Arrow export should not require GeoDataFrame export")

    expected = overlay(left, right, how="union")
    monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)
    result = pa.table(tabular.to_arrow(geometry_encoding="WKB"))

    assert result.column_names == ["col1", "col2", "geometry"]
    assert_geodataframe_equal(
        GeoDataFrame.from_arrow(result),
        expected,
        normalize=True,
        check_column_type=False,
    )


def test_overlay_union_native_tabular_writes_feather_without_frame_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    native_result, _used_owned = overlay_module._overlay_union_native(left, right)
    tabular = to_native_tabular_result(native_result)

    expected_result_type = (
        NativeTabularSelection if vibespatial.has_gpu_runtime() else NativeTabularResult
    )
    assert isinstance(tabular, expected_result_type)

    def _fail(*_args, **_kwargs):
        raise AssertionError("native Feather write should not require GeoDataFrame export")

    path = tmp_path / "overlay-union-native.feather"
    expected = overlay(left, right, how="union")
    monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)
    tabular.to_feather(path)
    monkeypatch.undo()

    result = geopandas.read_feather(path)
    assert_geodataframe_equal(
        result,
        expected,
        normalize=True,
        check_column_type=False,
    )


def test_overlay_intersection_export_result_writes_without_fragment_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )
    expected = overlay(left, right, how="intersection")

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "native intersection GeoParquet write should not require GeoDataFrame export"
        )

    monkeypatch.setattr(
        native_results_module,
        "_pairwise_constructive_fragment_to_native_tabular_result",
        _fail,
    )
    monkeypatch.setattr(PairwiseConstructiveResult, "to_geodataframe", _fail)
    export_result, _used_owned = overlay_module._overlay_intersection_export_result(left, right)

    path = tmp_path / "overlay-intersection-export.parquet"
    monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)
    write_geoparquet(export_result, path, geometry_encoding="geoarrow")
    monkeypatch.undo()

    result = geopandas.read_parquet(path)
    assert_geodataframe_equal(result, expected, normalize=True, check_column_type=False)


def test_overlay_difference_export_result_writes_without_fragment_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10]},
        geometry=GeoSeries([box(1, 1, 4, 4)]),
    )
    expected, _ = overlay_module._overlay_difference(left, right)

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "native difference GeoParquet write should not require GeoDataFrame export"
        )

    monkeypatch.setattr(
        native_results_module,
        "_left_constructive_fragment_to_native_tabular_result",
        _fail,
    )
    monkeypatch.setattr(LeftConstructiveResult, "to_geodataframe", _fail)
    export_result, _used_owned = overlay_module._overlay_difference_export_result(left, right)

    path = tmp_path / "overlay-difference-export.parquet"
    monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)
    write_geoparquet(export_result, path, geometry_encoding="geoarrow")
    monkeypatch.undo()

    result = geopandas.read_parquet(path)
    assert_geodataframe_equal(result, expected, normalize=True, check_column_type=False)


def test_overlay_difference_boundary_guard_uses_owned_family_metadata_without_geom_type_export() -> (
    None
):
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            GeometryArray.from_owned(from_shapely_geometries([box(0, 0, 2, 2), None]))
        ),
    )
    right = GeoDataFrame(
        {"col2": [10]},
        geometry=GeoSeries(GeometryArray.from_owned(from_shapely_geometries([box(1, 1, 4, 4)]))),
    )

    clear_materialization_events()
    assert overlay_module._needs_host_overlay_difference_boundary_rebuild(left, right) is False
    events = get_materialization_events(clear=True)
    assert not any(
        event.operation in {"geoseries_geom_type", "geodataframe_geom_type"} for event in events
    )

    line_left = GeoDataFrame(
        {"col1": [1]},
        geometry=GeoSeries(
            GeometryArray.from_owned(from_shapely_geometries([LineString([(0, 0), (1, 1)])]))
        ),
    )
    assert overlay_module._needs_host_overlay_difference_boundary_rebuild(line_left, right) is True


def test_overlay_difference_no_pairs_preserves_owned_left_without_device_array_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device-owned overlay difference")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=DeviceGeometryArray._from_owned(
            from_shapely_geometries(
                [box(0, 0, 1, 1), box(2, 2, 3, 3)],
                residency=Residency.DEVICE,
            )
        ),
    )
    right = GeoDataFrame(
        {"col2": [10]},
        geometry=DeviceGeometryArray._from_owned(
            from_shapely_geometries([box(10, 10, 11, 11)], residency=Residency.DEVICE)
        ),
    )

    def _fail_array_export(self, dtype=None, copy=None):
        raise AssertionError("no-pair difference should keep left geometry owned")

    monkeypatch.setattr(DeviceGeometryArray, "__array__", _fail_array_export)
    clear_materialization_events()

    result = overlay(left, right, how="difference")
    events = get_materialization_events(clear=True)

    assert len(result) == 2
    assert getattr(result.geometry.values, "_owned", None) is not None
    assert not any(event.operation == "device_geometryarray_to_numpy" for event in events)


def test_overlay_difference_full_row_grouping_preserves_unmatched_rows() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device-owned overlay difference")

    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geoms = [
        box(0.0, 0.0, 4.0, 4.0),
        box(10.0, 0.0, 14.0, 4.0),
        box(20.0, 0.0, 24.0, 4.0),
    ]
    right_geoms = [box(2.0, 0.0, 5.0, 4.0), box(22.0, 0.0, 25.0, 4.0)]
    left = GeoDataFrame(
        {"row": [0, 1, 2]},
        geometry=DeviceGeometryArray._from_owned(
            from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
        ),
    )
    right = GeoDataFrame(
        geometry=DeviceGeometryArray._from_owned(
            from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
        ),
    )

    clear_dispatch_events()
    result = overlay(left, right, how="difference")
    events = get_dispatch_events(clear=True)

    expected = [
        shapely.difference(left_geoms[0], right_geoms[0]),
        left_geoms[1],
        shapely.difference(left_geoms[2], right_geoms[1]),
    ]
    assert result["row"].tolist() == [0, 1, 2]
    assert all(
        shapely.symmetric_difference(observed, wanted).area < 1.0e-8
        for observed, wanted in zip(result.geometry, expected, strict=True)
    )
    assert any(
        event.implementation == "grouped_overlay_difference_full_row_capacity" for event in events
    )


def test_overlay_difference_index_reset_preserves_private_native_state() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device-owned overlay difference")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        index=[10, 20],
        geometry=DeviceGeometryArray._from_owned(
            from_shapely_geometries(
                [box(0, 0, 1, 1), box(2, 2, 3, 3)],
                residency=Residency.DEVICE,
            )
        ),
    )
    right = GeoDataFrame(
        {"col2": [10]},
        geometry=DeviceGeometryArray._from_owned(
            from_shapely_geometries([box(10, 10, 11, 11)], residency=Residency.DEVICE)
        ),
    )

    result = overlay(left, right, how="difference")

    from vibespatial.api._native_state import get_native_state

    state = get_native_state(result)
    assert result.index.equals(pd.RangeIndex(2))
    assert state is not None
    assert state.index_plan.kind == "range"
    assert state.column_order == tuple(result.columns)
    assert getattr(result.geometry.values, "_owned", None) is not None


def test_overlay_extract_owned_pair_keeps_large_device_owned_difference_inputs_native() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device-owned overlay dispatch")

    left = GeoDataFrame(
        {"col1": np.arange(600)},
        geometry=DeviceGeometryArray._from_owned(
            from_shapely_geometries(
                [box(float(i), 0.0, float(i) + 0.25, 0.25) for i in range(600)],
                residency=Residency.DEVICE,
            )
        ),
    )
    right = GeoDataFrame(
        {"col2": np.arange(500)},
        geometry=DeviceGeometryArray._from_owned(
            from_shapely_geometries(
                [box(float(i), 1.0, float(i) + 0.25, 1.25) for i in range(500)],
                residency=Residency.DEVICE,
            )
        ),
    )

    left_owned, right_owned = overlay_module._extract_owned_pair(
        left,
        right,
        how="difference",
    )

    assert left_owned is not None
    assert right_owned is not None
    assert left_owned.residency is Residency.DEVICE
    assert right_owned.residency is Residency.DEVICE


def test_overlay_extract_owned_pair_keeps_large_device_owned_intersection_native() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for device-owned overlay dispatch")

    left = GeoDataFrame(
        {"col1": np.arange(600)},
        geometry=DeviceGeometryArray._from_owned(
            from_shapely_geometries(
                [box(float(i), 0.0, float(i) + 0.25, 0.25) for i in range(600)],
                residency=Residency.DEVICE,
            )
        ),
    )
    right = GeoDataFrame(
        {"col2": np.arange(500)},
        geometry=DeviceGeometryArray._from_owned(
            from_shapely_geometries(
                [box(float(i), 1.0, float(i) + 0.25, 1.25) for i in range(500)],
                residency=Residency.DEVICE,
            )
        ),
    )

    left_owned, right_owned = overlay_module._extract_owned_pair(
        left,
        right,
        how="intersection",
    )

    assert left_owned is not None
    assert right_owned is not None
    assert left_owned.residency is Residency.DEVICE
    assert right_owned.residency is Residency.DEVICE


def test_overlay_symmetric_difference_native_writes_without_fragment_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )
    expected = overlay(left, right, how="symmetric_difference")

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "native symmetric-difference GeoParquet write should not require GeoDataFrame export"
        )

    monkeypatch.setattr(
        native_results_module,
        "_symmetric_difference_constructive_result_to_native_tabular_result",
        _fail,
    )
    monkeypatch.setattr(LeftConstructiveResult, "to_geodataframe", _fail)
    native_result, _used_owned = overlay_module._overlay_symmetric_diff_native(left, right)

    path = tmp_path / "overlay-symdiff-native.parquet"
    monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)
    write_geoparquet(native_result, path, geometry_encoding="geoarrow")
    monkeypatch.undo()

    result = geopandas.read_parquet(path)
    assert_geodataframe_equal(result, expected, normalize=True, check_column_type=False)


def test_overlay_symmetric_difference_native_builds_native_tabular_result() -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    native_result, _used_owned = overlay_module._overlay_symmetric_diff_native(left, right)
    tabular = to_native_tabular_result(native_result)

    assert isinstance(tabular, NativeTabularResult)
    assert_geodataframe_equal(
        tabular.to_geodataframe(),
        overlay(left, right, how="symmetric_difference"),
        normalize=True,
        check_column_type=False,
    )


def test_overlay_intersection_few_right_skips_non_polygon_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

    left = GeoDataFrame(
        {"col1": np.arange(24, dtype=np.int32)},
        geometry=GeoSeries([Point(float(i), 0.0) for i in range(24)]),
    )
    right = GeoDataFrame(
        {"zone_type": ["A", "B", "C"]},
        geometry=GeoSeries(
            [
                box(-1, -1, 8, 1),
                box(7, -1, 16, 1),
                box(15, -1, 24, 1),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    idx1 = np.arange(24, dtype=np.int32)
    idx2 = np.repeat(np.arange(3, dtype=np.int32), 8)

    binary_calls: list[int] = []

    monkeypatch.setattr(
        overlay_module,
        "_few_right_intersection_owned",
        lambda *args, **kwargs: pytest.fail(
            "few-right polygon shortcut must not run for non-polygon inputs"
        ),
    )

    def _fake_binary(op, left_arg, right_arg, **kwargs):
        assert op == "intersection"
        binary_calls.append(left_arg.row_count)
        return GeometryNativeResult.from_owned(left_arg, crs=None)

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_native",
        _fake_binary,
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=(idx1, idx2),
    )

    assert used_owned is True
    assert binary_calls == [24]
    assert result["col1"].tolist() == idx1.tolist()
    assert result["zone_type"].tolist() == ["A"] * 8 + ["B"] * 8 + ["C"] * 8


def test_overlay_intersection_few_right_uses_direct_rectangle_clip_for_rectangle_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

    polygon_rect_intersection_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_rect_intersection"
    )

    left = GeoDataFrame(
        {"col1": np.arange(24, dtype=np.int32)},
        geometry=GeoSeries([box(i, 0, i + 1, 1) for i in range(24)]),
    )
    right = GeoDataFrame(
        {"zone_type": ["A", "B", "C"]},
        geometry=GeoSeries(
            [
                Point(8.0, 1.0).buffer(9.0),
                Point(16.0, 1.0).buffer(9.0),
                Point(24.0, 1.0).buffer(9.0),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    idx1 = np.arange(24, dtype=np.int32)
    idx2 = np.repeat(np.arange(3, dtype=np.int32), 8)

    import cupy as cp

    rect_calls: list[tuple[int, int]] = []
    exact_calls: list[int] = []

    def _fake_inactive_exact(left_arg, right_arg, **_kwargs):
        exact_calls.append(left_arg.row_count)
        assert not cp.asnumpy(left_arg.device_state.validity).any()
        assert not cp.asnumpy(right_arg.device_state.validity).any()
        return from_shapely_geometries(
            [None] * left_arg.row_count,
            residency=Residency.DEVICE,
        )

    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_exact_batch_gpu",
        _fake_inactive_exact,
    )

    def _fake_polygon_rect_intersection_from_bounds(
        left_arg,
        rect_bounds,
        *,
        source_rows,
        dispatch_mode=ExecutionMode.GPU,
    ):
        assert rect_bounds.shape == (left_arg.row_count, 4)
        assert source_rows.shape == (left_arg.row_count,)
        rect_calls.append((left_arg.row_count, int(rect_bounds.shape[0])))
        geoms = [box(float(i) + 0.2, 0.2, float(i) + 0.8, 0.4) for i in range(left_arg.row_count)]
        return from_shapely_geometries(geoms, residency=Residency.DEVICE)

    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection_from_bounds",
        _fake_polygon_rect_intersection_from_bounds,
    )
    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection",
        lambda *args, **kwargs: pytest.fail(
            "few-right rectangle pairs should use row-indirected rectangle bounds"
        ),
    )
    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        lambda *args, **kwargs: pytest.fail(
            "rectangle-capable few-right intersection should use row-indirected "
            "rectangle bounds before generic constructive dispatch"
        ),
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=(idx1, idx2),
    )

    assert used_owned is True
    assert rect_calls == [(24, 24), (24, 24)]
    assert exact_calls == [24]
    assert result["col1"].tolist() == idx1.tolist()
    assert result["zone_type"].tolist() == ["A"] * 8 + ["B"] * 8 + ["C"] * 8


def test_few_right_rect_clip_accepts_right_rectangle_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    polygon_rect_intersection_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_rect_intersection"
    )

    left_pairs = from_shapely_geometries(
        [Point(float(i), 0.0).buffer(2.0) for i in range(4)],
        residency=Residency.DEVICE,
    )
    right_pairs = from_shapely_geometries(
        [box(float(i) - 0.5, -0.5, float(i) + 0.5, 0.5) for i in range(4)],
        residency=Residency.DEVICE,
    )
    sentinel = from_shapely_geometries(
        [box(float(8000 + i), 0.0, float(8000 + i + 0.5), 0.5) for i in range(4)],
        residency=Residency.DEVICE,
    )
    rect_calls: list[tuple[int, int]] = []
    rect_active_counts: list[int] = []

    def _fake_polygon_rect_intersection_from_bounds(
        left_arg,
        rect_bounds,
        *,
        source_rows,
        dispatch_mode=ExecutionMode.GPU,
    ):
        assert rect_bounds.shape == (left_arg.row_count, 4)
        assert source_rows.shape == (left_arg.row_count,)
        rect_calls.append((left_arg.row_count, int(rect_bounds.shape[0])))
        rect_active_counts.append(int(cp.count_nonzero(cp.isfinite(rect_bounds[:, 0])).item()))
        return sentinel

    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection_from_bounds",
        _fake_polygon_rect_intersection_from_bounds,
    )
    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection",
        lambda *args, **kwargs: pytest.fail(
            "few-right rectangle pairs should use row-indirected rectangle bounds"
        ),
    )

    result = overlay_module._few_right_partitioned_polygon_intersection_owned(
        left_pairs,
        right_pairs,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert result is not None
    assert rect_calls == [(4, 4), (4, 4)]
    assert rect_active_counts == [4, 0]
    for got, exp in zip(result.to_shapely(), sentinel.to_shapely(), strict=True):
        assert got.equals(exp)


def test_few_right_rect_clip_compacts_row_indirected_mixed_rectangle_rows() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    left_base = from_shapely_geometries(
        [
            box(0.0, 0.0, 4.0, 4.0),
            LineString([(100.0, 0.0), (101.0, 1.0)]),
            LineString([(110.0, 0.0), (111.0, 1.0), (112.0, 1.0)]),
            box(5.0, 0.0, 9.0, 4.0),
        ],
        residency=Residency.DEVICE,
    )
    left_indices = np.tile(np.asarray([0, 3], dtype=np.int64), 600)
    left_pairs = left_base.device_take(cp.asarray(left_indices, dtype=cp.int64))
    right_pattern = [
        Polygon([(1.0, 1.0), (3.0, 1.0), (2.0, 3.0), (1.0, 1.0)]),
        Polygon([(6.0, 1.0), (8.0, 1.0), (8.5, 2.0), (7.0, 3.0), (6.0, 2.0), (6.0, 1.0)]),
    ]
    right_base = from_shapely_geometries(
        right_pattern,
        residency=Residency.DEVICE,
    )
    right_pairs = right_base.device_take(
        cp.asarray(np.arange(left_indices.size) % 2, dtype=cp.int64)
    )

    assert left_pairs.is_indexed_view
    assert right_pairs.is_indexed_view
    assert set(left_pairs.families) == {GeometryFamily.POLYGON, GeometryFamily.LINESTRING}

    result = overlay_module._few_right_partitioned_polygon_intersection_owned(
        left_pairs,
        right_pairs,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert result is not None
    assert result.row_count == int(left_indices.size)
    expected = [right_pattern[0], right_pattern[1], right_pattern[0], right_pattern[1]]
    for got, exp in zip(
        result.take(np.arange(4, dtype=np.intp)).to_shapely(), expected, strict=True
    ):
        assert got.equals(exp)


def test_few_right_mixed_rect_bounds_clips_active_rows_without_physicalizing_batch() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.constructive import binary_constructive as constructive_module

    left_geoms = [
        box(0.0, 0.0, 4.0, 4.0),
        Polygon([(5.0, 0.0), (9.0, 0.0), (7.0, 3.5), (5.0, 0.0)]),
        box(10.0, 0.0, 14.0, 4.0),
    ]
    right_geoms = [
        Polygon([(0.5, 1.0), (2.0, 0.4), (3.5, 1.0), (3.0, 3.0), (1.0, 3.0), (0.5, 1.0)]),
        box(6.0, -1.0, 8.0, 2.0),
        Polygon([(10.5, 0.5), (13.5, 0.5), (12.0, 3.5), (10.5, 0.5)]),
    ]
    left_base = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right_base = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
    pair_rows = cp.asarray([0, 1, 2, 0], dtype=cp.int64)
    left_pairs = left_base.device_take(pair_rows)
    right_pairs = right_base.device_take(pair_rows)

    assert left_pairs.is_indexed_view
    assert right_pairs.is_indexed_view

    vibespatial.clear_dispatch_events()
    result = constructive_module._dispatch_partitioned_polygon_intersection_gpu(
        left_pairs,
        right_pairs,
        dispatch_mode=ExecutionMode.GPU,
    )
    events = vibespatial.get_dispatch_events(clear=True)

    assert result is not None
    assert result.row_count == int(pair_rows.size)
    assert any(
        event.implementation == "polygon_intersection_partitioned_capacity_gpu" for event in events
    )
    assert any(
        event.implementation == "polygon_rect_intersection_row_indirected_bounds_gpu"
        for event in events
    )
    expected = [left_geoms[int(i)].intersection(right_geoms[int(i)]) for i in cp.asnumpy(pair_rows)]
    for got, exp in zip(result.to_shapely(), expected, strict=True):
        assert got.equals(exp)


def test_mixed_rect_boundary_split_repair_stays_native_without_validity_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module
    from vibespatial.constructive import make_valid_pipeline as make_valid_module
    from vibespatial.constructive import validity as validity_module
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    coords = []
    for index in range(24):
        angle = math.pi * index / 12.0
        radius = 200.0 if index % 2 == 0 else 80.0
        coords.append(
            (
                500.0 + radius * math.cos(angle),
                500.0 + radius * math.sin(angle),
            )
        )
    star = Polygon(coords)
    cell = box(470.0, 410.0, 480.0, 420.0)
    left = from_shapely_geometries([star], residency=Residency.DEVICE)
    right = from_shapely_geometries([cell], residency=Residency.DEVICE)

    monkeypatch.setattr(
        validity_module,
        "is_valid_owned",
        lambda *_args, **_kwargs: pytest.fail(
            "mixed polygon-rectangle boundary split should not export validity"
        ),
    )
    monkeypatch.setattr(
        make_valid_module,
        "make_valid_owned",
        lambda *_args, **_kwargs: pytest.fail(
            "mixed polygon-rectangle boundary split should not use make_valid"
        ),
    )
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    result = constructive_module._dispatch_partitioned_polygon_intersection_gpu(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result is not None
    assert result.row_count == 1
    assert getattr(result, "_polygon_rect_boundary_repair_complete", False) is True
    assert not any("constructive validity" in reason for reason in runtime_reasons)
    monkeypatch.setattr(
        make_valid_module,
        "_try_device_validity_expression_rows",
        lambda *_args, **_kwargs: pytest.fail(
            "repaired polygon-rectangle boundary proof should skip validity expressions"
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_overlay_host_bool_mask_sparse_first",
        lambda *_args, **_kwargs: pytest.fail(
            "repaired polygon-rectangle boundary proof should avoid overlap row export"
        ),
    )
    series = GeoSeries(GeometryArray.from_owned(result))
    assert overlay_module._repair_invalid_polygon_output_rows(series) is series

    actual = np.asarray(result.to_shapely(), dtype=object)[0]
    expected = shapely.intersection(star, cell)
    assert actual.geom_type == "MultiPolygon"
    assert shapely.is_valid(actual)
    assert shapely.equals(shapely.normalize(actual), shapely.normalize(expected))


def test_mixed_rect_intersection_marks_polygon_complete_rows_on_device() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.constructive import binary_constructive as constructive_module

    rectangle_rows = [
        box(0.0, 0.0, 2.0, 2.0),
        box(1.0, 0.5, 4.0, 1.5),
        box(2.0, 0.0, 4.0, 2.0),
    ]
    polygon_rows = [
        Polygon([(-1.0, 1.0), (3.0, -1.0), (3.0, 3.0), (-1.0, 1.0)]),
        shapely.union_all(
            [
                box(0.0, 0.0, 2.0, 2.0),
                box(1.0, 2.0, 5.0, 3.0),
                box(4.0, 1.0, 5.0, 2.5),
            ]
        ),
        box(4.0, 0.0, 6.0, 2.0),
    ]
    left = from_shapely_geometries(rectangle_rows, residency=Residency.DEVICE)
    right = from_shapely_geometries(polygon_rows, residency=Residency.DEVICE)

    result = constructive_module._dispatch_partitioned_polygon_intersection_gpu(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    exact_polygon_only = getattr(result, "_polygon_rect_exact_polygon_only", None)
    assert exact_polygon_only is not None
    assert cp.asnumpy(exact_polygon_only).tolist() == [True, False, False]
    expected = [
        left_geom.intersection(right_geom)
        for left_geom, right_geom in zip(rectangle_rows, polygon_rows, strict=True)
    ]
    assert [geom.geom_type for geom in expected] == ["Polygon", "GeometryCollection", "LineString"]


def test_polygon_boundary_remnants_use_capacity_geometry_composition() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.constructive.boundary_remnants import (
        polygon_pair_boundary_remnants_capacity_device,
    )

    left_geom = box(1.0, 0.5, 4.0, 1.5)
    right_geom = shapely.union_all(
        [
            box(0.0, 0.0, 2.0, 2.0),
            box(1.0, 2.0, 5.0, 3.0),
            box(4.0, 1.0, 5.0, 2.5),
        ]
    )
    expected = shapely.intersection(left_geom, right_geom)
    area_parts = [
        part
        for part in shapely.get_parts(np.asarray([expected], dtype=object))
        if part.geom_type in {"Polygon", "MultiPolygon"}
    ]
    area_geom = shapely.union_all(np.asarray(area_parts, dtype=object))

    result = polygon_pair_boundary_remnants_capacity_device(
        from_shapely_geometries([left_geom], residency=Residency.DEVICE),
        from_shapely_geometries([right_geom], residency=Residency.DEVICE),
        from_shapely_geometries([area_geom], residency=Residency.DEVICE),
        crs=None,
    )

    assert result is not None
    geometry, d_keep = result
    assert cp.asnumpy(d_keep).tolist() == [True]
    assert geometry.composition is not None
    assert geometry.residency is Residency.DEVICE
    assert len(geometry.composition.parts) >= 2
    assert all(
        part.geometry.residency is Residency.DEVICE
        and hasattr(part.output_rows, "__cuda_array_interface__")
        for part in geometry.composition.parts
    )
    actual = geometry.to_geoseries(index=pd.RangeIndex(1), name="geometry").iloc[0]
    assert actual.geom_type == "GeometryCollection"
    assert shapely.equals(shapely.normalize(actual), shapely.normalize(expected))


def test_polygon_boundary_remnants_capacity_preserves_area_and_line_rows() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.constructive.boundary_remnants import (
        polygon_pair_boundary_remnants_capacity_device,
    )

    left_geoms = [
        box(1.0, 0.5, 4.0, 1.5),
        box(10.0, 0.0, 11.0, 1.0),
    ]
    right_geoms = [
        shapely.union_all(
            [
                box(0.0, 0.0, 2.0, 2.0),
                box(1.0, 2.0, 5.0, 3.0),
                box(4.0, 1.0, 5.0, 2.5),
            ]
        ),
        box(11.0, 0.0, 12.0, 1.0),
    ]
    expected = [
        shapely.intersection(left, right)
        for left, right in zip(left_geoms, right_geoms, strict=True)
    ]
    first_area = shapely.union_all(
        np.asarray(
            [
                part
                for part in shapely.get_parts(np.asarray([expected[0]], dtype=object))
                if part.geom_type in {"Polygon", "MultiPolygon"}
            ],
            dtype=object,
        )
    )
    area_owned = from_shapely_geometries(
        [first_area, Polygon()],
        residency=Residency.DEVICE,
    )

    result = polygon_pair_boundary_remnants_capacity_device(
        from_shapely_geometries(left_geoms, residency=Residency.DEVICE),
        from_shapely_geometries(right_geoms, residency=Residency.DEVICE),
        area_owned,
        crs=None,
    )

    assert result is not None
    geometry, d_keep = result
    assert cp.asnumpy(d_keep).tolist() == [True, True]
    assert geometry.composition is not None
    assert geometry.composition.row_count == 2
    assert all(part.geometry.row_count == 2 for part in geometry.composition.parts)
    assert all(
        cp.asnumpy(part.output_rows).tolist() == [0, 1] for part in geometry.composition.parts
    )
    actual = geometry.to_geoseries(index=pd.RangeIndex(2), name="geometry")
    assert [item.geom_type for item in actual] == ["GeometryCollection", "LineString"]
    assert all(
        shapely.equals(shapely.normalize(got), shapely.normalize(want))
        for got, want in zip(actual, expected, strict=True)
    )


def test_few_right_nonrectangle_rows_feed_native_boundary_composition() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.constructive.boundary_remnants import (
        polygon_pair_boundary_remnants_capacity_device,
    )

    def _pair(offset: float) -> tuple[Polygon, Polygon]:
        left = Polygon(
            [
                (offset + 1.0, 0.5),
                (offset + 4.0, 0.5),
                (offset + 4.0, 1.5),
                (offset + 1.0, 1.7),
                (offset + 1.0, 0.5),
            ]
        )
        right = shapely.union_all(
            [
                box(offset + 0.0, 0.0, offset + 2.0, 2.0),
                box(offset + 1.0, 2.0, offset + 5.0, 3.0),
                box(offset + 4.0, 1.0, offset + 5.0, 2.5),
            ]
        )
        return left, right

    pair_zero = _pair(0.0)
    pair_one = _pair(10.0)
    left_geoms = [pair_zero[0]] * 8 + [pair_one[0]] * 8
    right_geoms = [pair_zero[1], pair_one[1]]
    idx1 = np.arange(16, dtype=np.int32)
    idx2 = np.repeat(np.arange(2, dtype=np.int32), 8)
    left_owned = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right_owned = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

    vibespatial.clear_dispatch_events()
    area_owned = overlay_module._few_right_intersection_owned(
        left_owned,
        right_owned,
        idx1,
        idx2,
        dispatch_mode=ExecutionMode.GPU,
        _preserve_lower_dim_polygon_results=True,
    )
    events = vibespatial.get_dispatch_events(clear=True)

    assert area_owned is not None
    assert area_owned.row_count == 16
    assert any(
        event.implementation
        == "row_isolated_polygon_overlay_row_indirected_exact_gpu"
        for event in events
    )
    aligned_left, aligned_right = overlay_module._aligned_pair_owned_from_area(area_owned)
    composed = polygon_pair_boundary_remnants_capacity_device(
        aligned_left,
        aligned_right,
        area_owned,
        crs=None,
    )

    assert composed is not None
    geometry, d_keep = composed
    assert cp.asnumpy(d_keep).tolist() == [True] * 16
    assert geometry.composition is not None
    actual = geometry.to_geoseries(
        index=pd.RangeIndex(16),
        name="geometry",
    )
    expected = [
        left_geom.intersection(right_geoms[int(right_row)])
        for left_geom, right_row in zip(left_geoms, idx2, strict=True)
    ]
    assert all(item.geom_type == "GeometryCollection" for item in expected)
    for actual_geom, expected_geom in zip(actual, expected, strict=True):
        assert shapely.equals(
            shapely.normalize(actual_geom),
            shapely.normalize(expected_geom),
        )


def test_device_polygon_overlay_keep_geom_type_false_stays_native() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_geom = Polygon([(1.0, 0.5), (4.0, 0.5), (4.0, 1.5), (1.0, 1.7), (1.0, 0.5)])
    right_geom = shapely.union_all(
        [
            box(0.0, 0.0, 2.0, 2.0),
            box(1.0, 2.0, 5.0, 3.0),
            box(4.0, 1.0, 5.0, 2.5),
        ]
    )
    left = GeoDataFrame(
        {"left": [1]},
        geometry=GeoSeries(
            GeometryArray.from_owned(
                from_shapely_geometries([left_geom], residency=Residency.DEVICE)
            )
        ),
    )
    right = GeoDataFrame(
        {"right": [2]},
        geometry=GeoSeries(
            GeometryArray.from_owned(
                from_shapely_geometries([right_geom], residency=Residency.DEVICE)
            )
        ),
    )

    vibespatial.clear_dispatch_events()
    result = overlay(left, right, how="intersection", keep_geom_type=False)
    events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 1
    assert result.geometry.iloc[0].geom_type == "GeometryCollection"
    assert shapely.equals(
        shapely.normalize(result.geometry.iloc[0]),
        shapely.normalize(left_geom.intersection(right_geom)),
    )
    assert any(
        event.implementation == "polygon_boundary_remnant_composition_gpu" for event in events
    )
    assert not any(
        event.surface == "geopandas.overlay" and event.implementation == "shapely_host"
        for event in events
    )


def test_overlay_intersection_few_right_large_batches_keep_indexed_right_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

    left = GeoDataFrame(
        {"col1": np.arange(320, dtype=np.int32)},
        geometry=GeoSeries([box(float(i), 0.0, float(i) + 1.0, 1.0) for i in range(320)]),
    )
    right = GeoDataFrame(
        {"zone_type": ["A", "B", "C", "D"]},
        geometry=GeoSeries(
            [
                Polygon(
                    [(-20, -2), (100, -2), (100, 3), (-20, 3)],
                    holes=[[(10, 1.5), (20, 1.5), (20, 2.5), (10, 2.5)]],
                ),
                Polygon(
                    [(60, -2), (180, -2), (180, 3), (60, 3)],
                    holes=[[(90, 1.5), (100, 1.5), (100, 2.5), (90, 2.5)]],
                ),
                Polygon(
                    [(140, -2), (260, -2), (260, 3), (140, 3)],
                    holes=[[(170, 1.5), (180, 1.5), (180, 2.5), (170, 2.5)]],
                ),
                Polygon(
                    [(220, -2), (340, -2), (340, 3), (220, 3)],
                    holes=[[(250, 1.5), (260, 1.5), (260, 2.5), (250, 2.5)]],
                ),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    idx1 = np.arange(320, dtype=np.int32)
    idx2 = np.repeat(np.arange(4, dtype=np.int32), 80)

    rowwise_calls: list[tuple[int, bool, bool]] = []

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        lambda *args, **kwargs: pytest.fail(
            "large few-right batches should stay on one exact rowwise pass"
        ),
    )

    def _fake_rowwise(
        left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU, _cached_right_segments=None
    ):
        rowwise_calls.append(
            (
                left_arg.row_count,
                bool(right_arg.is_indexed_view),
                _cached_right_segments is not None,
            )
        )
        geoms = [box(float(i) + 0.2, 0.2, float(i) + 0.8, 0.4) for i in range(left_arg.row_count)]
        return from_shapely_geometries(geoms, residency=Residency.DEVICE)

    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_exact_batch_gpu",
        _fake_rowwise,
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=(idx1, idx2),
    )

    assert used_owned is True
    assert rowwise_calls == [(320, True, False)]
    assert result["col1"].tolist() == idx1.tolist()
    assert result["zone_type"].tolist() == ["A"] * 80 + ["B"] * 80 + ["C"] * 80 + ["D"] * 80


def test_overlay_intersection_few_right_fallback_preserves_exact_polygon_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

    left = GeoDataFrame(
        {"col1": np.arange(24, dtype=np.int32)},
        geometry=GeoSeries([box(i, 0, i + 1, 1) for i in range(24)]),
    )
    right = GeoDataFrame(
        {"zone_type": ["A", "B", "C"]},
        geometry=GeoSeries(
            [
                Polygon(
                    [(0, 0), (16, 0), (16, 2), (0, 2)],
                    holes=[[(4, 0.5), (6, 0.5), (6, 1.5), (4, 1.5)]],
                ),
                Polygon(
                    [(8, 0), (24, 0), (24, 2), (8, 2)],
                    holes=[[(12, 0.5), (14, 0.5), (14, 1.5), (12, 1.5)]],
                ),
                Polygon(
                    [(16, 0), (32, 0), (32, 2), (16, 2)],
                    holes=[[(20, 0.5), (22, 0.5), (22, 1.5), (20, 1.5)]],
                ),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    idx1 = np.arange(24, dtype=np.int32)
    idx2 = np.repeat(np.arange(3, dtype=np.int32), 8)

    fallback_calls: list[int] = []

    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_exact_batch_gpu",
        lambda *args, **kwargs: None,
    )

    def _fake_binary(op, left_arg, right_arg, **kwargs):
        assert op == "intersection"
        fallback_calls.append(left_arg.row_count)
        geoms = [box(float(i) + 0.2, 0.2, float(i) + 0.8, 0.4) for i in range(left_arg.row_count)]
        return from_shapely_geometries(geoms, residency=Residency.DEVICE)

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        _fake_binary,
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=(idx1, idx2),
    )

    assert used_owned is True
    assert fallback_calls == [24]
    assert result["col1"].tolist() == idx1.tolist()
    assert result["zone_type"].tolist() == ["A"] * 8 + ["B"] * 8 + ["C"] * 8


def test_overlay_intersection_exact_mode_prefers_rectangle_kernel_for_rectangles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

    envelope_module = importlib.import_module("vibespatial.constructive.envelope")
    polygon_rect_intersection_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_rect_intersection"
    )
    polygon_intersection_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_intersection"
    )

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                box(0, 0, 2, 2),
                box(2, 2, 4, 4),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries(
            [
                box(1, 1, 3, 3),
                box(3, 3, 5, 5),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    idx = np.arange(2, dtype=np.int32)

    bounds_calls: list[int] = []
    original_build_boxes = envelope_module._build_device_boxes_from_bounds

    def _record_build_boxes(device_bounds, *, row_count=None):
        bounds_calls.append(int(device_bounds.shape[0]))
        return original_build_boxes(device_bounds, row_count=row_count)

    monkeypatch.setattr(
        envelope_module,
        "_build_device_boxes_from_bounds",
        _record_build_boxes,
    )

    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection_from_bounds",
        lambda *args, **kwargs: pytest.fail(
            "rectangle pairs should construct intersections directly from bounds"
        ),
    )
    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection",
        lambda *args, **kwargs: pytest.fail(
            "exact rectangle-capable intersection should use row-indirected bounds"
        ),
    )
    monkeypatch.setattr(
        polygon_intersection_module,
        "polygon_intersection",
        lambda *args, **kwargs: pytest.fail(
            "rectangle-capable intersection should use polygon_rect_intersection before polygon_intersection"
        ),
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_exact_batch_gpu",
        lambda *args, **kwargs: pytest.fail(
            "exact rectangle-capable intersection should use polygon_rect_intersection before rowwise overlay"
        ),
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=(idx, idx),
    )

    assert used_owned is True
    assert bounds_calls == [2]
    assert result["col1"].tolist() == [1, 2]
    assert result.geometry.iloc[0].equals(box(1, 1, 2, 2))
    assert result.geometry.iloc[1].equals(box(3, 3, 4, 4))


def test_overlay_intersection_exact_mode_resolves_indexed_rectangle_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.constructive import binary_constructive as constructive_module
    from vibespatial.spatial.query_types import DeviceSpatialJoinResult

    polygon_rect_intersection_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_rect_intersection"
    )
    polygon_intersection_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_intersection"
    )

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Point(0.0, 0.0).buffer(5.0),
                Point(20.0, 0.0).buffer(5.0),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries(
            [
                box(-2.0, -2.0, 2.0, 2.0),
                box(18.0, -3.0, 24.0, 3.0),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    left_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test indexed rectangle overlay exact mode",
    )
    right_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test indexed rectangle overlay exact mode",
    )

    d_idx = cp.asarray([0, 0, 1, 1], dtype=cp.int32)
    index_result = DeviceSpatialJoinResult(d_idx, d_idx)

    rect_calls: list[int] = []
    rect_active_counts: list[int] = []
    inactive_sh_calls: list[int] = []
    inactive_exact_calls: list[int] = []

    def _fake_polygon_rect_intersection_from_bounds(
        left_arg,
        rect_bounds,
        *,
        source_rows,
        dispatch_mode=ExecutionMode.GPU,
    ):
        assert rect_bounds.shape == (left_arg.row_count, 4)
        assert source_rows.shape == (left_arg.row_count,)
        rect_calls.append(left_arg.row_count)
        rect_active_counts.append(int(cp.count_nonzero(cp.isfinite(rect_bounds[:, 0])).item()))
        return from_shapely_geometries(
            [
                box(-1.0, -1.0, 1.0, 1.0),
                box(-1.0, -1.0, 1.0, 1.0),
                box(19.0, -1.0, 21.0, 1.0),
                box(19.0, -1.0, 21.0, 1.0),
            ],
            residency=Residency.DEVICE,
        )

    def _fake_inactive_partition(left_arg, right_arg, **_kwargs):
        inactive_sh_calls.append(left_arg.row_count)
        assert not cp.asnumpy(left_arg.device_state.validity).any()
        assert not cp.asnumpy(right_arg.device_state.validity).any()
        return from_shapely_geometries(
            [None] * left_arg.row_count,
            residency=Residency.DEVICE,
        )

    def _fake_inactive_exact(left_arg, right_arg, **_kwargs):
        inactive_exact_calls.append(left_arg.row_count)
        assert not cp.asnumpy(left_arg.device_state.validity).any()
        assert not cp.asnumpy(right_arg.device_state.validity).any()
        return from_shapely_geometries(
            [None] * left_arg.row_count,
            residency=Residency.DEVICE,
        )

    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection_from_bounds",
        _fake_polygon_rect_intersection_from_bounds,
    )
    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection",
        lambda *args, **kwargs: pytest.fail(
            "indexed rectangle-capable batches should use row-indirected bounds"
        ),
    )
    monkeypatch.setattr(
        polygon_intersection_module,
        "polygon_intersection",
        _fake_inactive_partition,
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_exact_batch_gpu",
        _fake_inactive_exact,
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=index_result,
    )

    assert used_owned is True
    assert rect_calls == [4, 4]
    assert rect_active_counts == [4, 0]
    assert inactive_sh_calls == [4, 4]
    assert inactive_exact_calls == [4]
    assert result["col1"].tolist() == [1, 1, 2, 2]
    assert result["col2"].tolist() == [10, 10, 20, 20]
    assert [geom.bounds for geom in result.geometry] == [
        (-1.0, -1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0, 1.0),
        (19.0, -1.0, 21.0, 1.0),
        (19.0, -1.0, 21.0, 1.0),
    ]


def test_overlay_intersection_many_vs_one_device_pairs_skip_gather_sizing_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.spatial.query_types import DeviceSpatialJoinResult

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                box(0.0, 0.0, 2.0, 2.0),
                box(3.0, 0.0, 5.0, 2.0),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [10]},
        geometry=GeoSeries([box(-1.0, -1.0, 6.0, 3.0)]),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    left_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test many-vs-one overlay left input",
    )
    right_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test many-vs-one overlay right input",
    )
    index_result = DeviceSpatialJoinResult(
        cp.asarray([0, 1], dtype=cp.int32),
        cp.asarray([0, 0], dtype=cp.int32),
    )

    def _fail_right_to_host(*_args, **_kwargs):
        raise AssertionError("many-vs-one overlay should synthesize right pair rows")

    monkeypatch.setattr(DeviceSpatialJoinResult, "right_to_host", _fail_right_to_host)
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=index_result,
    )
    events = get_materialization_events(clear=True)
    d2h_events = get_d2h_transfer_events(clear=True)

    assert used_owned is True
    assert result["col1"].tolist() == [1, 2]
    assert result["col2"].tolist() == [10, 10]
    assert not result.geometry.is_empty.any()
    assert not any(
        event.reason == "overlay many-vs-one unique right-row scalar host boundary"
        for event in d2h_events
    )
    assert not any(
        event.surface == "vibespatial.spatial.query_types.DeviceSpatialJoinResult.left_to_host"
        for event in events
    )
    assert not any(
        event.operation == "pairwise_constructive_relation_indices_to_host" for event in events
    )
    assert not any(
        event.reason == "overlay terminal pairwise left attribute rows export"
        for event in d2h_events
    )
    assert not any(
        event.operation == "pairwise_constructive_relation_indices_to_host"
        and "side=right" in event.detail
        for event in events
    )
    assert not any("many-vs-one overlay gather sizing" in event.reason for event in events)
    assert not any(
        event.surface == "vibespatial.spatial.query_types.DeviceSpatialJoinResult.right_to_host"
        for event in events
    )


def test_overlay_intersection_general_device_pairs_construct_before_host_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.constructive import binary_constructive as constructive_module
    from vibespatial.spatial.query_types import DeviceSpatialJoinResult

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                box(0.0, 0.0, 3.0, 3.0),
                box(10.0, 0.0, 13.0, 3.0),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries(
            [
                box(1.0, 1.0, 2.0, 2.0),
                box(11.0, 1.0, 12.0, 2.0),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    left_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test general overlay left input",
    )
    right_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test general overlay right input",
    )
    index_result = DeviceSpatialJoinResult(
        cp.asarray([0, 1], dtype=cp.int32),
        cp.asarray([0, 1], dtype=cp.int32),
    )

    constructive_started = False

    def _fail_device_pair_export(*_args, **_kwargs):
        raise AssertionError("general overlay should carry pair context natively")

    def _fake_binary_constructive(*_args, **_kwargs):
        nonlocal constructive_started
        constructive_started = True
        return GeometryNativeResult.from_owned(
            from_shapely_geometries(
                [box(1.0, 1.0, 2.0, 2.0), box(11.0, 1.0, 12.0, 2.0)],
                residency=Residency.DEVICE,
            ),
            crs=None,
        )

    monkeypatch.setattr(DeviceSpatialJoinResult, "left_to_host", _fail_device_pair_export)
    monkeypatch.setattr(DeviceSpatialJoinResult, "right_to_host", _fail_device_pair_export)
    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_native",
        _fake_binary_constructive,
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=index_result,
    )

    assert constructive_started is True
    assert used_owned is True
    assert result["col1"].tolist() == [1, 2]
    assert result["col2"].tolist() == [10, 20]


def test_overlay_intersection_general_pairs_preserve_mixed_native_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.api._native_result_core import (
        NativeGeometryComposition,
        NativeGeometryCompositionPart,
    )
    from vibespatial.constructive import binary_constructive as constructive_module
    from vibespatial.spatial.query_types import DeviceSpatialJoinResult

    left = GeoDataFrame(
        {"left_value": [1, 2]},
        geometry=GeoSeries([box(0, 0, 3, 3), box(10, 0, 13, 3)]),
    )
    right = GeoDataFrame(
        {"right_value": [10, 20]},
        geometry=GeoSeries([box(1, 1, 2, 2), box(11, 1, 12, 2)]),
    )
    left_owned = from_shapely_geometries(list(left.geometry), residency=Residency.DEVICE)
    right_owned = from_shapely_geometries(list(right.geometry), residency=Residency.DEVICE)
    point_part = GeometryNativeResult.from_owned(
        from_shapely_geometries([Point(1.0, 1.0)], residency=Residency.DEVICE),
        crs=None,
    )
    polygon_part = GeometryNativeResult.from_owned(
        from_shapely_geometries([box(11.0, 1.0, 12.0, 2.0)], residency=Residency.DEVICE),
        crs=None,
    )
    native_composition = GeometryNativeResult.from_composition(
        NativeGeometryComposition(
            parts=(
                NativeGeometryCompositionPart(
                    geometry=point_part,
                    output_rows=cp.asarray([0], dtype=cp.int64),
                ),
                NativeGeometryCompositionPart(
                    geometry=polygon_part,
                    output_rows=cp.asarray([1], dtype=cp.int64),
                ),
            ),
            row_count=2,
            crs=None,
        ),
        crs=None,
    )

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_native",
        lambda *_args, **_kwargs: native_composition,
    )
    native_result, used_owned = overlay_module._overlay_intersection_native(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _preserve_lower_dim_polygon_results=True,
        _index_result=DeviceSpatialJoinResult(
            cp.asarray([0, 1], dtype=cp.int32),
            cp.asarray([0, 1], dtype=cp.int32),
        ),
    )

    assert used_owned is True
    assert isinstance(native_result, NativeTabularSelection)
    assert native_result.capacity_result.geometry.composition is not None
    exported = native_result.to_geodataframe()
    assert exported["left_value"].tolist() == [1, 2]
    assert exported["right_value"].tolist() == [10, 20]
    assert exported.geometry.geom_type.tolist() == ["Point", "Polygon"]


def test_overlay_intersection_few_right_device_pairs_stay_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.spatial.query_types import DeviceSpatialJoinResult

    left = GeoDataFrame(
        {"col1": np.arange(24, dtype=np.int32)},
        geometry=GeoSeries([box(float(i), 0.0, float(i) + 0.5, 0.5) for i in range(24)]),
    )
    right = GeoDataFrame(
        {"col2": np.asarray([10, 20, 30], dtype=np.int32)},
        geometry=GeoSeries(
            [
                box(0.0, -1.0, 8.0, 1.0),
                box(8.0, -1.0, 16.0, 1.0),
                box(16.0, -1.0, 24.0, 1.0),
            ]
        ),
    )
    left_state = _attach_owned_overlay_state(left)
    right_state = _attach_owned_overlay_state(right)
    left_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test few-right overlay native left input",
    )
    right_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test few-right overlay native right input",
    )
    index_result = DeviceSpatialJoinResult(
        cp.asarray(np.arange(24, dtype=np.int32)),
        cp.asarray(np.repeat(np.arange(3, dtype=np.int32), 8)),
    )

    def _fail_device_pair_export(*_args, **_kwargs):
        raise AssertionError("few-right overlay should consume device pair rows")

    def _fail_relation_pair_export(*_args, **_kwargs):
        raise AssertionError("few-right native attributes should not export pair rows")

    monkeypatch.setattr(DeviceSpatialJoinResult, "left_to_host", _fail_device_pair_export)
    monkeypatch.setattr(DeviceSpatialJoinResult, "right_to_host", _fail_device_pair_export)
    monkeypatch.setattr(RelationIndexResult, "to_host", _fail_relation_pair_export)
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()

    with strict_native_environment():
        tabular, used_owned = overlay_module._overlay_intersection_export_result(
            left,
            right,
            left_owned=left_state.geometry.owned,
            right_owned=right_state.geometry.owned,
            _prefer_exact_polygon_gpu=True,
            _index_result=index_result,
        )

    assert used_owned is True
    assert isinstance(tabular, NativeTabularSelection)
    capacity_attributes = tabular.capacity_result.attributes
    assert tuple(capacity_attributes.columns) == ("col1", "col2")
    assert capacity_attributes.loader is None
    assert capacity_attributes.device_table is not None or capacity_attributes.parts is not None
    events = get_materialization_events(clear=True)
    assert not any(
        event.surface
        in {
            "vibespatial.spatial.query_types.DeviceSpatialJoinResult.left_to_host",
            "vibespatial.spatial.query_types.DeviceSpatialJoinResult.right_to_host",
            "vibespatial.api._native_results._pairwise_constructive_to_native_tabular_result",
        }
        for event in events
    )
    d2h_events = get_d2h_transfer_events(clear=True)
    assert not any("device spatial join" in event.reason for event in d2h_events)
    assert not any("pairwise constructive" in event.reason for event in d2h_events)

    exported = tabular.to_geodataframe()
    assert exported["col1"].tolist() == list(range(24))
    assert exported["col2"].tolist() == [10] * 8 + [20] * 8 + [30] * 8
    reset_d2h_transfer_count()


def test_overlay_intersection_device_pairs_build_native_tabular_without_pair_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.spatial.query_types import DeviceSpatialJoinResult

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                box(0.0, 0.0, 3.0, 3.0),
                box(10.0, 0.0, 13.0, 3.0),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries(
            [
                box(1.0, 1.0, 2.0, 2.0),
                box(11.0, 1.0, 12.0, 2.0),
            ]
        ),
    )
    left_state = _attach_owned_overlay_state(left)
    right_state = _attach_owned_overlay_state(right)
    left_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test overlay native tabular left input",
    )
    right_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test overlay native tabular right input",
    )
    index_result = DeviceSpatialJoinResult(
        cp.asarray([0, 1], dtype=cp.int32),
        cp.asarray([0, 1], dtype=cp.int32),
    )

    def _fail_device_pair_export(*_args, **_kwargs):
        raise AssertionError("overlay native tabular assembly should consume device pair rows")

    def _fail_relation_pair_export(*_args, **_kwargs):
        raise AssertionError(
            "pairwise constructive native attributes should defer host row positions"
        )

    monkeypatch.setattr(DeviceSpatialJoinResult, "left_to_host", _fail_device_pair_export)
    monkeypatch.setattr(DeviceSpatialJoinResult, "right_to_host", _fail_device_pair_export)
    monkeypatch.setattr(RelationIndexResult, "to_host", _fail_relation_pair_export)
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()

    with strict_native_environment():
        tabular, used_owned = overlay_module._overlay_intersection_export_result(
            left,
            right,
            left_owned=left_state.geometry.owned,
            right_owned=right_state.geometry.owned,
            _index_result=index_result,
        )

    assert used_owned is True
    assert isinstance(tabular, NativeTabularSelection)
    capacity_attributes = tabular.capacity_result.attributes
    assert capacity_attributes.loader is None
    assert capacity_attributes.device_table is not None or capacity_attributes.parts is not None
    assert tuple(capacity_attributes.columns) == ("col1", "col2")
    events = get_materialization_events(clear=True)
    assert not any(
        event.surface
        in {
            "vibespatial.spatial.query_types.DeviceSpatialJoinResult.left_to_host",
            "vibespatial.spatial.query_types.DeviceSpatialJoinResult.right_to_host",
            "vibespatial.api._native_results._pairwise_constructive_to_native_tabular_result",
        }
        for event in events
    )
    d2h_events = get_d2h_transfer_events(clear=True)
    assert not any("device spatial join" in event.reason for event in d2h_events)
    assert not any("pairwise constructive" in event.reason for event in d2h_events)

    exported = tabular.to_geodataframe()
    assert exported["col1"].tolist() == [1, 2]
    assert exported["col2"].tolist() == [10, 20]
    reset_d2h_transfer_count()


def test_overlay_intersection_host_polygon_boundary_prefers_pair_owned_gpu_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                box(0, 0, 2, 2),
                box(2, 2, 4, 4),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries(
            [
                box(1, 1, 3, 3),
                box(3, 3, 5, 5),
            ]
        ),
    )
    idx = np.arange(2, dtype=np.int32)

    kernel_calls: list[int] = []

    def _fake_binary(*args, **kwargs):
        kernel_calls.append(args[1].row_count)
        return from_shapely_geometries(
            [
                box(1.0, 1.0, 2.0, 2.0),
                box(3.0, 3.0, 4.0, 4.0),
            ],
            residency=Residency.DEVICE,
        )

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        _fake_binary,
    )
    monkeypatch.setattr(
        overlay_module,
        "_take_geoseries_object_values",
        lambda *_args, **_kwargs: pytest.fail(
            "small polygon pair batches should prefer pair-owned GPU exact path before host exact intersection"
        ),
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        _index_result=(idx, idx),
    )

    assert used_owned is True
    assert kernel_calls == [2]
    assert result["col1"].tolist() == [1, 2]
    assert result["col2"].tolist() == [10, 20]


def test_overlay_intersection_warning_path_prefers_pair_owned_gpu_boundary_for_small_device_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.constructive import binary_constructive as constructive_module

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                box(0, 0, 2, 2),
                box(2, 2, 4, 4),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries(
            [
                box(1, 1, 3, 3),
                box(3, 3, 5, 5),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    left_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test overlay warning path prefers pair-owned exact boundary",
    )
    right_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test overlay warning path prefers pair-owned exact boundary",
    )
    kernel_calls: list[int] = []

    def _fake_binary(*args, **kwargs):
        kernel_calls.append(args[1].row_count)
        result = from_shapely_geometries(
            [
                box(1.0, 1.0, 2.0, 2.0),
                box(3.0, 3.0, 4.0, 4.0),
            ],
            residency=Residency.DEVICE,
        )
        result._polygon_rect_boundary_overlap = cp.zeros(2, dtype=cp.bool_)
        return GeometryNativeResult.from_owned(result, crs=None)

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_native",
        _fake_binary,
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_exact_batch_gpu",
        lambda *args, **kwargs: pytest.fail(
            "small device-backed warning batches should stay on the pair-owned boundary path"
        ),
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _warn_on_dropped_lower_dim_polygon_results=True,
        _index_result=(np.arange(2, dtype=np.int32), np.arange(2, dtype=np.int32)),
    )

    assert used_owned is True
    assert kernel_calls == [2]
    assert result["col1"].tolist() == [1, 2]
    assert result["col2"].tolist() == [10, 20]


def test_overlay_intersection_rect_overlap_metadata_skips_generic_make_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.constructive import binary_constructive as constructive_module

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                box(0, 0, 2, 2),
                box(2, 2, 4, 4),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries(
            [
                box(1, 1, 3, 3),
                box(3, 3, 5, 5),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    left_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test overlay rect overlap metadata skips generic make_valid",
    )
    right_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test overlay rect overlap metadata skips generic make_valid",
    )

    def _fake_binary(*args, **kwargs):
        result = from_shapely_geometries(
            [
                box(1.0, 1.0, 2.0, 2.0),
                box(3.0, 3.0, 4.0, 4.0),
            ],
            residency=Residency.DEVICE,
        )
        result._polygon_rect_boundary_overlap = cp.zeros(2, dtype=cp.bool_)
        return result

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        _fake_binary,
    )
    monkeypatch.setattr(
        overlay_module,
        "_make_valid_geoseries",
        lambda *_args, **_kwargs: pytest.fail(
            "rectangle-overlap-tagged intersection batches should defer generic make_valid"
        ),
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _warn_on_dropped_lower_dim_polygon_results=True,
        _index_result=(np.arange(2, dtype=np.int32), np.arange(2, dtype=np.int32)),
    )

    assert used_owned is True
    assert result["col1"].tolist() == [1, 2]
    assert result["col2"].tolist() == [10, 20]


def test_overlay_intersection_exact_mode_splits_mixed_rectangle_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

    polygon_rect_intersection_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_rect_intersection"
    )
    polygon_intersection_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_intersection"
    )

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Point(0.0, 0.0).buffer(5.0),
                Polygon(
                    [
                        (18.0, -5.0),
                        (26.0, -5.0),
                        (26.0, 5.0),
                        (18.0, 5.0),
                        (18.0, -5.0),
                    ],
                    holes=[[(20.0, -1.0), (24.0, -1.0), (24.0, 1.0), (20.0, 1.0), (20.0, -1.0)]],
                ),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries(
            [
                box(-2.0, -2.0, 2.0, 2.0),
                box(18.0, -3.0, 24.0, 3.0),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()

    rect_calls: list[int] = []
    sh_calls: list[int] = []
    rowwise_calls: list[int] = []

    def _fake_polygon_rect_intersection_from_bounds(
        left_arg,
        rect_bounds,
        *,
        source_rows,
        dispatch_mode=ExecutionMode.GPU,
    ):
        assert rect_bounds.shape == (left_arg.row_count, 4)
        assert source_rows.shape == (left_arg.row_count,)
        rect_calls.append(left_arg.row_count)
        return from_shapely_geometries(
            [
                box(-1.0, -1.0, 1.0, 1.0),
                None,
            ],
            residency=Residency.DEVICE,
        )

    def _fake_rowwise(
        left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU, _cached_right_segments=None
    ):
        rowwise_calls.append(left_arg.row_count)
        return from_shapely_geometries(
            [
                box(18.5, -2.5, 19.5, -1.5),
                None,
            ],
            residency=Residency.DEVICE,
        )

    def _fake_polygon_intersection(
        left_arg,
        right_arg,
        *,
        dispatch_mode=ExecutionMode.GPU,
    ):
        sh_calls.append(left_arg.row_count)
        return from_shapely_geometries(
            [None, None],
            residency=Residency.DEVICE,
        )

    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection_from_bounds",
        _fake_polygon_rect_intersection_from_bounds,
    )
    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection",
        lambda *args, **kwargs: pytest.fail(
            "mixed rectangle batches should use row-indirected rectangle bounds"
        ),
    )
    monkeypatch.setattr(
        polygon_intersection_module,
        "polygon_intersection",
        _fake_polygon_intersection,
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_exact_batch_gpu",
        _fake_rowwise,
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=(np.arange(2, dtype=np.int32), np.arange(2, dtype=np.int32)),
    )

    assert used_owned is True
    assert rect_calls == [2, 2]
    assert sh_calls == [2, 2]
    assert rowwise_calls == [2]
    assert result["col1"].tolist() == [1, 2]
    assert result["col2"].tolist() == [10, 20]
    assert [geom.bounds for geom in result.geometry] == [
        (-1.0, -1.0, 1.0, 1.0),
        (18.5, -2.5, 19.5, -1.5),
    ]


def test_overlay_intersection_exact_mode_recovers_all_handled_rect_batches_when_batch_probe_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

    polygon_rect_intersection_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_rect_intersection"
    )
    polygon_intersection_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_intersection"
    )

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Point(0.0, 0.0).buffer(5.0),
                Point(20.0, 0.0).buffer(5.0),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries(
            [
                box(-2.0, -2.0, 2.0, 2.0),
                box(18.0, -3.0, 24.0, 3.0),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()

    rect_calls: list[int] = []
    sh_calls: list[int] = []
    rowwise_calls: list[int] = []

    def _fake_polygon_rect_intersection_from_bounds(
        left_arg,
        rect_bounds,
        *,
        source_rows,
        dispatch_mode=ExecutionMode.GPU,
    ):
        assert rect_bounds.shape == (left_arg.row_count, 4)
        assert source_rows.shape == (left_arg.row_count,)
        rect_calls.append(left_arg.row_count)
        return from_shapely_geometries(
            [
                box(-1.0, -1.0, 1.0, 1.0),
                box(19.0, -1.0, 21.0, 1.0),
            ],
            residency=Residency.DEVICE,
        )

    def _fake_polygon_intersection(
        left_arg,
        right_arg,
        *,
        dispatch_mode=ExecutionMode.GPU,
    ):
        sh_calls.append(left_arg.row_count)
        return from_shapely_geometries(
            [None, None],
            residency=Residency.DEVICE,
        )

    def _fake_rowwise(
        left_arg,
        right_arg,
        *,
        dispatch_mode=ExecutionMode.GPU,
        _cached_right_segments=None,
    ):
        rowwise_calls.append(left_arg.row_count)
        return from_shapely_geometries(
            [None, None],
            residency=Residency.DEVICE,
        )

    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection_can_handle",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection_from_bounds",
        _fake_polygon_rect_intersection_from_bounds,
    )
    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection",
        lambda *args, **kwargs: pytest.fail(
            "batch-probe misses should recover through row-indirected rectangle bounds"
        ),
    )
    monkeypatch.setattr(
        polygon_intersection_module,
        "polygon_intersection",
        _fake_polygon_intersection,
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_exact_batch_gpu",
        _fake_rowwise,
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=(np.arange(2, dtype=np.int32), np.arange(2, dtype=np.int32)),
    )

    assert used_owned is True
    assert rect_calls == [2, 2]
    assert sh_calls == [2, 2]
    assert rowwise_calls == [2]
    assert result["col1"].tolist() == [1, 2]
    assert result["col2"].tolist() == [10, 20]
    assert [geom.bounds for geom in result.geometry] == [
        (-1.0, -1.0, 1.0, 1.0),
        (19.0, -1.0, 21.0, 1.0),
    ]


def test_overlay_intersection_default_keep_geom_type_drops_touch_only_rectangle_rows() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1]},
        geometry=GeoSeries([box(0.0, 0.0, 2.0, 2.0)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries(
            [
                box(1.0, 0.0, 3.0, 2.0),
                box(2.0, 0.0, 4.0, 2.0),
            ]
        ),
    )

    with strict_native_environment():
        with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay"):
            result = overlay(left, right, how="intersection", keep_geom_type=None)

    assert len(result) == 1
    assert result["col1"].tolist() == [1]
    assert result["col2"].tolist() == [10]
    assert result.geometry.iloc[0].equals(box(1.0, 0.0, 2.0, 2.0))


def test_overlay_difference_survives_strict_native_mode_for_small_overlap_polygons() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [Point(0, 0).buffer(1, quad_segs=2), Point(1.5, 0).buffer(1, quad_segs=2)]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [Point(1, 1).buffer(1, quad_segs=2), Point(2, 2).buffer(1, quad_segs=2)]
        ),
    )

    with strict_native_environment():
        result = overlay(left, right, how="difference")

    assert len(result) == 2
    assert result.geometry.notna().all()


def test_grouped_overlay_difference_does_not_mutate_source_validity() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")

    polygons = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    lines = GeoDataFrame(
        {"col3": [1, 2]},
        geometry=GeoSeries(
            [
                LineString([(2, 0), (2, 4), (6, 4)]),
                LineString([(0, 3), (6, 3)]),
            ]
        ),
    )

    with strict_native_environment():
        polygon_owned, line_owned = overlay_module._coerce_owned_pair_for_strict_overlay(
            polygons,
            lines,
            None,
            None,
        )
        polygon_validity = cp.asarray(
            polygon_owned._ensure_device_state(preserve_indexed_view=True).validity,
            dtype=cp.bool_,
        ).copy()
        line_validity = cp.asarray(
            line_owned._ensure_device_state(preserve_indexed_view=True).validity,
            dtype=cp.bool_,
        ).copy()
        result, used_owned = overlay_module._overlay_symmetric_diff_native(
            polygons,
            lines,
            polygon_owned,
            line_owned,
        )

    assert used_owned is True
    assert bool(cp.array_equal(polygon_owned.device_state.validity, polygon_validity))
    assert bool(cp.array_equal(line_owned.device_state.validity, line_validity))
    exported = result.to_geodataframe()
    line_results = exported.loc[exported["col1"].isna(), "geometry"]
    assert len(line_results) == 2
    assert all(geometry.geom_type == "MultiLineString" for geometry in line_results)


@pytest.mark.gpu
def test_mixed_overlay_union_normalizes_assembled_polygons_like_geos() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col4": [1, 2]},
        geometry=GeoSeries([Point(2, 2), Point(3, 3)]),
    )

    with strict_native_environment():
        result = overlay(left, right, how="union", keep_geom_type=False)
        expected_frame = read_file(
            Path(
                "tests/upstream/geopandas/tests/data/overlay/strict/"
                "poly_point_union_False.geojson"
            )
        )
        sort_columns = list(set(result.columns) - {"geometry"})
        result = result.sort_values(sort_columns).reset_index(drop=True)
        expected_frame = expected_frame.sort_values(sort_columns).reset_index(drop=True)
        normalized_array = result.geometry.array.normalize()
        expected_array = expected_frame.geometry.array.normalize()
        normalized = np.asarray(normalized_array, dtype=object)
        expected = np.asarray(expected_array, dtype=object)

    assert shapely.equals_exact(normalized, expected, tolerance=0.0).all()


def test_overlay_difference_preserves_left_geometry_name() -> None:
    left = (
        GeoDataFrame(
            {"col1": [1, 2]},
            geometry=GeoSeries([box(0, 0, 2, 2), box(2, 2, 4, 4)]),
        )
        .rename(columns={"geometry": "polygons"})
        .set_geometry("polygons")
    )
    right = GeoDataFrame(
        {"col2": [10]},
        geometry=GeoSeries([box(1, 1, 3, 3)]),
    )

    result = overlay(left, right, how="difference")

    assert result.geometry.name == "polygons"
    assert result._geometry_column_name == "polygons"


def test_overlay_strict_nybb_single_pair_intersection_matches_host() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    data = os.path.join(
        os.path.dirname(__file__),
        "upstream",
        "geopandas",
        "tests",
        "data",
    )
    overlay_data = os.path.join(data, "overlay", "nybb_qgis")
    left = read_file(f"zip://{os.path.join(data, 'nybb_16a.zip')}").iloc[[4]].copy()
    right = read_file(os.path.join(overlay_data, "polydf2.shp")).iloc[[8]].copy()

    with strict_native_environment():
        result = overlay(left, right, how="intersection")

    expected = left.geometry.iloc[0].intersection(right.geometry.iloc[0])
    assert len(result) == 1
    assert result.geometry.iloc[0].geom_type == expected.geom_type
    assert result.geometry.iloc[0].normalize().equals_exact(expected.normalize(), tolerance=1e-6)


@pytest.mark.parametrize(
    ("how", "keep_geom_type"),
    [
        ("union", True),
        ("union", False),
        ("symmetric_difference", True),
        ("symmetric_difference", False),
    ],
)
def test_overlay_touching_polygon_groups_do_not_merge_owned_difference_rows(
    how: str,
    keep_geom_type: bool,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    df1 = GeoDataFrame(
        {
            "col1": [1, 2],
            "geometry": GeoSeries(
                [
                    Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                    Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
                ]
            ),
        }
    )
    df2 = GeoDataFrame(
        {
            "col2": [1, 2, 3],
            "geometry": GeoSeries(
                [
                    Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                    Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                    Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
                ]
            ),
        }
    )

    result = overlay(df1, df2, how=how, keep_geom_type=keep_geom_type)
    polygon_only = result.loc[result["col1"].isna() & (result["col2"] == 2), "geometry"]

    assert len(polygon_only) == 1
    assert shapely.equals(
        polygon_only.iloc[0],
        Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
    )
@pytest.mark.gpu
def test_reverse_device_relation_gather_is_sorted_and_independent() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.spatial.query_types import DeviceSpatialJoinResult

    forward_left = cp.asarray([1, 0, 1], dtype=cp.int32)
    forward_right = cp.asarray([2, 3, 1], dtype=cp.int32)
    reverse = overlay_module._reverse_intersecting_index_pairs(
        DeviceSpatialJoinResult(forward_left, forward_right)
    )

    np.testing.assert_array_equal(reverse.d_left_idx.get(), np.asarray([1, 2, 3]))
    np.testing.assert_array_equal(reverse.d_right_idx.get(), np.asarray([1, 1, 0]))
    assert reverse.d_left_idx.data.ptr != forward_right.data.ptr
    assert reverse.d_right_idx.data.ptr != forward_left.data.ptr
