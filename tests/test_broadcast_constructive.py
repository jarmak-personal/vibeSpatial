"""Tests for broadcast-right support in binary constructive operations.

Validates that scalar (1-row) right operands produce correct results for
intersection, union, difference, and symmetric_difference.  Oracle: Shapely
with tiled-pairwise comparison.

Covers nsf.3: elimination of [other]*len(self) materialization.
"""

from __future__ import annotations

import ast
import importlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

from benchmarks.shootout._data import setup_fixtures
from vibespatial.api import read_file
from vibespatial.constructive import binary_constructive as binary_constructive_module
from vibespatial.constructive.binary_constructive import binary_constructive_owned
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    from_shapely_geometries,
    materialize_broadcast,
    tile_single_row,
)
from vibespatial.kernels.constructive.segmented_union import segmented_union_all
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.hotpath_trace import reset_hotpath_trace, summarize_hotpath_trace
from vibespatial.runtime.residency import Residency
from vibespatial.testing import strict_native_environment

# The four constructive operations.
_CONSTRUCTIVE_OPS = ("intersection", "union", "difference", "symmetric_difference")

requires_gpu = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")


@requires_gpu
def test_owned_boundary_physicalizes_threshold_crossing_chunk_composition() -> None:
    """The first bounded row count remains consumable by legacy owned callers."""
    cp = pytest.importorskip("cupy")
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    chunk_rows = binary_constructive_module._polygon_constructive_chunk_rows(1 << 30)
    row_count = chunk_rows + 1
    left = tile_single_row(
        from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)], residency=Residency.DEVICE),
        row_count,
    )
    right = tile_single_row(
        from_shapely_geometries([box(1.0, 0.0, 3.0, 2.0)], residency=Residency.DEVICE),
        row_count,
    )

    clear_dispatch_events()
    result = binary_constructive_owned(
        "intersection",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    events = get_dispatch_events(clear=True)

    assert result.row_count == row_count
    assert result.residency is Residency.DEVICE
    assert cp.asarray(result.device_state.validity).shape == (row_count,)
    sample = result._device_indexed_take(
        cp.asarray([0, row_count - 1], dtype=cp.int64),
        assume_unique_indices=True,
    ).to_shapely()
    expected = shapely.normalize(box(1.0, 0.0, 2.0, 2.0))
    assert all(
        shapely.normalize(geometry).equals_exact(expected, tolerance=1e-12)
        for geometry in sample
    )
    assert any(
        event.implementation == "chunked_polygon_intersection_composition_gpu"
        and f"chunk_rows={chunk_rows}" in event.detail
        and "chunks=2" in event.detail
        for event in events
    )
    assert any(
        event.implementation
        == "binary_constructive_owned_composition_physicalization"
        for event in events
    )


def test_row_isolated_polygon_overlay_has_no_per_row_fallback_executor() -> None:
    source = Path(binary_constructive_module.__file__).read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_dispatch_polygon_overlay_row_isolated_batch_gpu"
    )
    function_source = ast.unparse(function)

    assert "for row_index in range" not in function_source
    assert "_resolve_indexed_polygon_fast_path_candidate" not in function_source
    assert "_row_isolated=True" in function_source
    assert "_expand_right_segments_for_pair_rows_legacy" not in source
    assert "_expand_right_segments_for_pair_rows" not in source
    assert "expanded right-segment allocation fence" not in source
    assert "_dispatch_exact_rowwise_polygon_union_rows_gpu" not in source
    assert "row_aligned_union_area_proof_cpu_subset_rescue" not in source
    assert "allow_segmented_union_fallback" not in source
    assert "multipart-union group-offset host export" not in source
    assert "constructive.intersection.multipart_union.fallback" not in source
    assert "_dispatch_polygonal_multipart_difference_gpu" not in source
    assert 'reason="polygonal difference polygon-part depth scalar fence"' not in source
    assert "polygonal_multipart_grouped_difference_gpu" not in source
    assert "_pack_line_parts_by_source_rows_gpu" not in source
    assert "_lineal_polygon_rows_difference_gpu" not in source
    assert "lineal-polygonal difference polygon-part depth scalar fence" not in source
    assert "for part_index in range(max_parts)" not in source
    assert "def _explode_polygonal_rows_to_polygons_gpu(" not in source
    assert "def _explode_multipolygon_rows_to_polygons_gpu(" not in source
    assert "_row_isolated_overlay_operation_area_gpu" not in source
    assert "def _host_single_ring_polygon_mask(" not in source
    assert "def _host_rectangle_polygon_mask(" not in source
    assert "def _host_single_ring_and_rectangle_polygon_masks(" not in source
    assert "def _sh_kernel_can_handle(" not in source
    assert "def _aligned_sh_eligible_polygon_rows(" not in source
    assert "def _dispatch_polygon_contraction_gpu(" not in source
    assert "_MIXED_POLYGON_INTERSECTION_ROWWISE_MAX" not in source
    assert "_POLYGON_CONTRACTION_MIN_ROWS" not in source

    binary_dispatch = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_binary_constructive_gpu"
    )
    binary_dispatch_source = ast.unparse(binary_dispatch)
    assert not any(isinstance(node, ast.Try) for node in ast.walk(binary_dispatch))
    assert "_dispatch_partitioned_polygon_intersection_gpu" in binary_dispatch_source
    assert "_dispatch_polygon_difference_overlay_batched_gpu" in binary_dispatch_source

    difference_capacity_start = source.index(
        "def _dispatch_polygon_difference_overlay_batched_gpu("
    )
    difference_capacity_end = source.index("\ndef ", difference_capacity_start + 1)
    difference_capacity_source = source[difference_capacity_start:difference_capacity_end]
    assert difference_capacity_source.count("NativeDeviceSelection.from_mask") == 2
    assert "build_empty_polygon_rows_device" in difference_capacity_source
    assert "device_scatter_owned_capacity_selection" in difference_capacity_source
    assert "cp.flatnonzero" not in difference_capacity_source
    assert "except Exception" not in difference_capacity_source
    assert "partition_counts=device-resident" in difference_capacity_source

    mixed_dispatch = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_dispatch_mixed_binary_constructive_gpu"
    )
    mixed_dispatch_source = ast.unparse(mixed_dispatch)
    assert "NativeDeviceSelection.from_mask" in mixed_dispatch_source
    assert "device_take_owned_family_capacity_selection" in mixed_dispatch_source
    assert "device_select_owned_capacity_partitions" in mixed_dispatch_source
    assert "cp.flatnonzero" not in mixed_dispatch_source
    assert ".take(" not in mixed_dispatch_source
    assert "int(d_sub_rows.size)" not in mixed_dispatch_source
    assert "_concat_device_family_buffers" not in mixed_dispatch_source

    line_capacity_start = source.index("def _explode_lineal_rows_to_line_capacity_gpu(")
    line_capacity_end = source.index("\ndef ", line_capacity_start + 1)
    line_capacity_source = source[line_capacity_start:line_capacity_end]
    line_compose_start = source.index("def _line_part_selection_from_capacities(")
    line_compose_end = source.index("\ndef ", line_compose_start + 1)
    line_compose_source = source[line_compose_start:line_compose_end]
    assert "_indexed_lineal_part_capacities" in line_capacity_source
    assert "preserve_indexed_view=True" in line_capacity_source
    assert "NativeDeviceSelection.from_mask" in line_compose_source
    assert "safe_capacity_positions" in line_compose_source
    assert "gather_capacity" in line_compose_source
    assert "coord_capacity=" in line_compose_source
    assert "cp.flatnonzero" not in line_capacity_source
    assert "count_scatter_total" not in line_capacity_source

    point_capacity_start = source.index("def _explode_point_rows_to_point_capacity_gpu(")
    point_capacity_end = source.index("\ndef ", point_capacity_start + 1)
    point_capacity_source = source[point_capacity_start:point_capacity_end]
    point_compose_start = source.index("def _point_part_selection_from_capacities(")
    point_compose_end = source.index("\ndef ", point_compose_start + 1)
    point_compose_source = source[point_compose_start:point_compose_end]
    assert "_indexed_point_part_capacities" in point_capacity_source
    assert "preserve_indexed_view=True" in point_capacity_source
    assert "NativeDeviceSelection.from_mask" in point_compose_source
    assert "partition_capacity_positions" in point_compose_source
    assert "gather_capacity" in point_compose_source
    assert "cp.flatnonzero" not in point_capacity_source
    assert "count_scatter_total" not in point_capacity_source

    boundary_capacity_start = source.index("def _polygon_part_capacity_boundary_segments_gpu(")
    boundary_capacity_end = source.index(
        "\ndef ",
        boundary_capacity_start + 1,
    )
    boundary_capacity_source = source[boundary_capacity_start:boundary_capacity_end]
    assert "preserve_indexed_view=True" in boundary_capacity_source
    assert "polygon_parts.ring_capacity" in boundary_capacity_source
    assert "polygon_parts.coord_capacity" in boundary_capacity_source
    assert "d_part_family_rows" in boundary_capacity_source
    assert "d_physical_ring_rows" in boundary_capacity_source
    assert "physicalize_device_rows" not in boundary_capacity_source

    for function_name in (
        "_dispatch_row_aligned_polygon_known_coverage_union_gpu",
        "_dispatch_grouped_polygon_known_coverage_union_gpu",
        "_dispatch_single_row_polygon_known_coverage_union_gpu",
    ):
        function_start = source.index(f"def {function_name}(")
        function_end = source.index("\ndef ", function_start + 1)
        function_source = source[function_start:function_end]
        assert "cp.flatnonzero" not in function_source
        assert "physicalize_device_rows" not in function_source

    noded_coverage_start = source.index(
        "def _dispatch_grouped_polygon_noded_coverage_union_gpu("
    )
    noded_coverage_end = source.index("\ndef ", noded_coverage_start + 1)
    noded_coverage_source = source[noded_coverage_start:noded_coverage_end]
    assert "build_gpu_split_events" in noded_coverage_source
    assert "_assemble_noded_polygon_coverage_split_events_gpu" in noded_coverage_source
    assert "right_geometry_source_rows=d_source_rows" in noded_coverage_source
    assert "cp.flatnonzero" not in noded_coverage_source
    assert "physicalize_device_rows" not in noded_coverage_source

    noded_assembly_start = source.index(
        "def _assemble_noded_polygon_coverage_split_events_gpu("
    )
    noded_assembly_end = source.index("\ndef ", noded_assembly_start + 1)
    noded_assembly_source = source[noded_assembly_start:noded_assembly_end]
    assert "noded_boundary_segments_from_split_events_gpu" in noded_assembly_source
    assert "undirected_boundary_segment_orders_gpu" in noded_assembly_source
    assert "build_polygon_output_from_boundary_segments_gpu" in noded_assembly_source
    assert "cp.flatnonzero" not in noded_assembly_source

    line_difference_source = (
        Path(binary_constructive_module.__file__)
        .with_name("line_polygon_difference.py")
        .read_text()
    )
    assert "build_gpu_split_events" in line_difference_source
    assert "NativeDeviceSelection.from_mask" in line_difference_source
    assert "binary_predicate_expression" in line_difference_source
    assert "sort_pairs" in line_difference_source
    assert "cp.flatnonzero" not in line_difference_source
    assert "count_scatter_total" not in line_difference_source
    assert "_device_scalar_int" not in line_difference_source

    multipart_start = source.index("def _dispatch_oriented_multipolygon_polygon_intersection_gpu(")
    multipart_end = source.index("\ndef ", multipart_start + 1)
    multipart_source = source[multipart_start:multipart_end]
    assert "_explode_polygonal_rows_to_polygon_capacity_gpu" in multipart_source
    assert "d_valid_rows_mask=d_valid_parts" in multipart_source
    assert "_explode_multipolygon_rows_to_polygons_gpu" not in multipart_source
    assert "cp.flatnonzero" not in multipart_source

    orientation_start = source.index("def _oriented_multipolygon_polygon_source_rows_gpu(")
    orientation_end = source.index("\ndef ", orientation_start + 1)
    orientation_source = source[orientation_start:orientation_end]
    assert "_device_single_family_covering_all_rows" in orientation_source
    assert "cp.arange" in orientation_source
    assert "cp.flatnonzero" not in orientation_source

    capacity_pack_start = source.index("def _pack_disjoint_multipart_intersection_capacity_gpu(")
    capacity_pack_end = source.index("\ndef ", capacity_pack_start + 1)
    capacity_pack_source = source[capacity_pack_start:capacity_pack_end]
    assert "NativeGroupedSelection" in capacity_pack_source
    assert "d_output_validity" in capacity_pack_source
    assert "d_valid_empty_rows=d_output_validity" in capacity_pack_source
    assert "_explode_polygonal_rows_to_polygons_gpu" not in capacity_pack_source
    assert "physicalize_device_rows" not in capacity_pack_source
    assert "ring_capacity=polygon_parts.ring_capacity" in capacity_pack_source
    assert "coord_capacity=polygon_parts.coord_capacity" in capacity_pack_source

    grouped_pack_start = source.index("def _pack_native_grouped_disjoint_polygon_parts_gpu(")
    grouped_pack_end = source.index("\ndef ", grouped_pack_start + 1)
    grouped_pack_source = source[grouped_pack_start:grouped_pack_end]
    assert "NativeGroupedSelection" in grouped_pack_source
    assert "_assemble_sorted_polygon_part_capacity_gpu" in grouped_pack_source
    assert "cp.argsort" not in grouped_pack_source
    assert "cp.flatnonzero" not in grouped_pack_source
    assert "physicalize_device_rows" not in grouped_pack_source
    assert "preserve_indexed_view=True" in grouped_pack_source

    disjoint_proof_start = source.index(
        "def _sorted_polygon_parts_have_strictly_disjoint_group_bounds("
    )
    disjoint_proof_end = source.index("\ndef ", disjoint_proof_start + 1)
    disjoint_proof_source = source[disjoint_proof_start:disjoint_proof_end]
    assert "NativeDeviceSelection.from_mask" in disjoint_proof_source
    assert "d_pair_count=d_refine_count" in disjoint_proof_source
    assert "pair_capacity=refine_capacity" in disjoint_proof_source
    assert "candidate_pair_count" not in disjoint_proof_source

    coverage_start = source.index("def _dispatch_grouped_polygon_known_coverage_union_gpu(")
    coverage_end = source.index("\ndef ", coverage_start + 1)
    coverage_source = source[coverage_start:coverage_end]
    assert "_dispatch_grouped_polygon_noded_coverage_union_gpu" in coverage_source
    assert "cp.flatnonzero(valid_group_mask)" not in coverage_source

    union_batch = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_dispatch_polygon_partition_union_gpu"
    )
    union_batch_source = ast.unparse(union_batch)
    assert "for row_index in range" not in union_batch_source
    assert "_include_same_side_splits=True" in union_batch_source
    assert "NativeDeviceSelection.from_mask" in union_batch_source
    assert "device_scatter_owned_capacity_selection" in union_batch_source
    assert "cp.flatnonzero" not in union_batch_source
    assert not any(isinstance(node, ast.Try) for node in ast.walk(union_batch))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shapely_oracle(
    op: str,
    left_geoms: list[object | None],
    right_geom: object | None,
) -> list[object | None]:
    """Compute expected results by tiling right_geom and using Shapely."""
    left_arr = np.asarray(left_geoms, dtype=object)
    right_arr = np.asarray([right_geom] * len(left_geoms), dtype=object)
    result = getattr(shapely, op)(left_arr, right_arr)
    out: list[object | None] = []
    for left_val, right_val, val in zip(
        left_geoms,
        [right_geom] * len(left_geoms),
        result.tolist(),
        strict=True,
    ):
        if left_val is None or right_val is None:
            out.append(None)
        else:
            out.append(val)
    return out


def _assert_constructive_matches_oracle(
    op: str,
    left_geoms: list[object | None],
    right_geom: object,
    *,
    tolerance: float = 1e-9,
) -> None:
    """Assert broadcast constructive result matches tiled Shapely oracle."""
    left_owned = from_shapely_geometries(list(left_geoms))
    right_owned = from_shapely_geometries([right_geom])

    result_owned = binary_constructive_owned(op, left_owned, right_owned)
    result_geoms = result_owned.to_shapely()

    expected = _shapely_oracle(op, left_geoms, right_geom)

    assert len(result_geoms) == len(expected), (
        f"Length mismatch: got {len(result_geoms)}, expected {len(expected)}"
    )

    for i, (got, exp) in enumerate(zip(result_geoms, expected, strict=True)):
        if exp is None:
            assert got is None, f"Row {i}: expected None, got {got}"
        elif shapely.is_empty(exp):
            assert shapely.is_empty(got), f"Row {i}: expected empty, got {got}"
        else:
            assert got.equals_exact(exp, tolerance), (
                f"Row {i}: {op} mismatch.\n  got={got}\n  exp={exp}"
            )


# ---------------------------------------------------------------------------
# 1. Oracle: broadcast result == tiled-pairwise Shapely result
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", _CONSTRUCTIVE_OPS)
def test_broadcast_polygon_polygon_matches_oracle(op: str) -> None:
    """Polygon x scalar Polygon broadcast matches tiled Shapely."""
    left = [
        box(0, 0, 2, 2),
        box(1, 1, 3, 3),
        box(5, 5, 7, 7),
        box(0, 0, 1, 1),
    ]
    right = box(0.5, 0.5, 2.5, 2.5)
    _assert_constructive_matches_oracle(op, left, right)


@pytest.mark.parametrize("op", _CONSTRUCTIVE_OPS)
def test_broadcast_point_point_matches_oracle(op: str) -> None:
    """Point x scalar Point broadcast matches tiled Shapely."""
    left = [
        Point(0, 0),
        Point(1, 1),
        Point(2, 2),
    ]
    right = Point(1, 1)
    _assert_constructive_matches_oracle(op, left, right)


@pytest.mark.parametrize("op", ("intersection", "difference"))
def test_broadcast_point_polygon_matches_oracle(op: str) -> None:
    """Point x scalar Polygon broadcast matches tiled Shapely."""
    left = [
        Point(1, 1),
        Point(0, 0),
        Point(3, 3),
        Point(0.5, 0.5),
    ]
    right = box(0, 0, 2, 2)
    _assert_constructive_matches_oracle(op, left, right)


# ---------------------------------------------------------------------------
# 2. All 4 ops: intersection, union, difference, symmetric_difference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", _CONSTRUCTIVE_OPS)
def test_all_four_ops_polygon(op: str) -> None:
    """All 4 constructive ops work with broadcast polygon."""
    left = [box(0, 0, 2, 2), box(1, 1, 3, 3)]
    right = box(0.5, 0.5, 2.5, 2.5)
    _assert_constructive_matches_oracle(op, left, right)


# ---------------------------------------------------------------------------
# 3. Null broadcast: single null geometry -> all-null output
# ---------------------------------------------------------------------------


def test_null_broadcast_right() -> None:
    """Broadcast of a null right geometry produces all-null output."""
    left = [box(0, 0, 2, 2), box(1, 1, 3, 3), box(5, 5, 7, 7)]
    right_owned = from_shapely_geometries([None])
    left_owned = from_shapely_geometries(left)

    result = binary_constructive_owned("intersection", left_owned, right_owned)
    result_geoms = result.to_shapely()

    for i, g in enumerate(result_geoms):
        assert g is None, f"Row {i}: expected None for null broadcast, got {g}"


# ---------------------------------------------------------------------------
# 4. Empty broadcast: single empty geometry -> all-empty output
# ---------------------------------------------------------------------------


def test_empty_broadcast_right() -> None:
    """Broadcast of an empty right geometry produces appropriate output."""
    left = [box(0, 0, 2, 2), box(1, 1, 3, 3)]
    # Create a valid empty polygon (e.g., intersection of disjoint polys).
    empty_geom = shapely.intersection(box(0, 0, 1, 1), box(10, 10, 11, 11))
    assert shapely.is_empty(empty_geom)

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries([empty_geom])

    result = binary_constructive_owned("intersection", left_owned, right_owned)
    result_geoms = result.to_shapely()

    for i, g in enumerate(result_geoms):
        assert shapely.is_empty(g), (
            f"Row {i}: expected empty for empty broadcast intersection, got {g}"
        )


@requires_gpu
@pytest.mark.parametrize("operation", _CONSTRUCTIVE_OPS)
def test_gpu_binary_constructive_preserves_valid_empty_identity_rows(
    operation: str,
) -> None:
    left_geometries = [box(0, 0, 1, 1), Polygon(), None, Polygon()]
    right_geometries = [Polygon(), box(2, 2, 3, 3), box(0, 0, 1, 1), Polygon()]
    left = from_shapely_geometries(left_geometries)
    right = from_shapely_geometries(right_geometries)

    with strict_native_environment():
        result = binary_constructive_owned(
            operation,
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
        )

    actual = result.to_shapely()
    expected = getattr(shapely, operation)(
        np.asarray(left_geometries, dtype=object),
        np.asarray(right_geometries, dtype=object),
    )
    for got, oracle in zip(actual, expected, strict=True):
        if oracle is None:
            assert got is None
        else:
            assert got is not None
            assert got.geom_type == oracle.geom_type
            assert shapely.equals(got, oracle)


# ---------------------------------------------------------------------------
# 5. Verify 1-row right (not N copies) via row_count check
# ---------------------------------------------------------------------------


def test_right_is_single_row() -> None:
    """After _coerce_other_to_owned change, right should be 1-row."""
    right_geom = box(0, 0, 2, 2)
    right_owned = from_shapely_geometries([right_geom])
    assert right_owned.row_count == 1, f"Expected 1-row right, got {right_owned.row_count}"


# ---------------------------------------------------------------------------
# 6. Tiling equivalence regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", _CONSTRUCTIVE_OPS)
def test_tiling_equivalence(op: str) -> None:
    """Broadcast result must match the old N-copy tiling approach."""
    left = [
        box(0, 0, 2, 2),
        box(1, 1, 3, 3),
        box(5, 5, 7, 7),
    ]
    right_geom = box(0.5, 0.5, 2.5, 2.5)

    # New broadcast path: 1-row right
    left_owned = from_shapely_geometries(left)
    right_owned_broadcast = from_shapely_geometries([right_geom])
    result_broadcast = binary_constructive_owned(op, left_owned, right_owned_broadcast)

    # Old tiling path: N-copy right
    right_owned_tiled = from_shapely_geometries([right_geom] * len(left))
    result_tiled = binary_constructive_owned(op, left_owned, right_owned_tiled)

    broadcast_geoms = result_broadcast.to_shapely()
    tiled_geoms = result_tiled.to_shapely()

    for i, (bg, tg) in enumerate(zip(broadcast_geoms, tiled_geoms, strict=True)):
        if tg is None:
            assert bg is None, f"Row {i}: broadcast={bg}, tiled=None"
        elif shapely.is_empty(tg):
            assert shapely.is_empty(bg), f"Row {i}: expected empty"
        else:
            assert bg.equals_exact(tg, 1e-9), (
                f"Row {i}: broadcast != tiled\n  broadcast={bg}\n  tiled={tg}"
            )


@requires_gpu
def test_indexed_right_segment_physicalization_matches_segment_oracle() -> None:
    import cupy as cp

    from vibespatial.spatial.segment_primitives import _extract_segments_gpu

    right = from_shapely_geometries(
        [
            Polygon(
                [(0, 0), (8, 0), (8, 4), (0, 4)],
                holes=[[(2, 1), (3, 1), (3, 2), (2, 2)]],
            ),
            Polygon(
                [(10, 0), (18, 0), (18, 4), (10, 4)],
                holes=[[(12, 1), (13, 1), (13, 2), (12, 2)]],
            ),
            Polygon(
                [(20, 0), (28, 0), (28, 4), (20, 4)],
                holes=[[(22, 1), (23, 1), (23, 2), (22, 2)]],
            ),
        ],
        residency=Residency.DEVICE,
    )
    source_rows = np.asarray([2, 0, 2, 1, 0, 2, 1, 1], dtype=np.int32)

    actual = _extract_segments_gpu(right.device_take(cp.asarray(source_rows)))

    base = _extract_segments_gpu(right)
    base_rows = cp.asnumpy(base.row_indices)
    expected_slots = np.concatenate(
        [np.flatnonzero(base_rows == source_row) for source_row in source_rows]
    )
    expected_rows = np.concatenate(
        [
            np.full(
                np.count_nonzero(base_rows == source_row),
                pair_row,
                dtype=np.int32,
            )
            for pair_row, source_row in enumerate(source_rows)
        ]
    )

    assert int(actual.count) == int(expected_slots.size)
    np.testing.assert_array_equal(cp.asnumpy(actual.row_indices), expected_rows)
    np.testing.assert_array_equal(
        cp.asnumpy(actual.segment_indices),
        cp.asnumpy(base.segment_indices)[expected_slots],
    )
    np.testing.assert_allclose(cp.asnumpy(actual.x0), cp.asnumpy(base.x0)[expected_slots])
    np.testing.assert_allclose(cp.asnumpy(actual.y0), cp.asnumpy(base.y0)[expected_slots])
    np.testing.assert_allclose(cp.asnumpy(actual.x1), cp.asnumpy(base.x1)[expected_slots])
    np.testing.assert_allclose(cp.asnumpy(actual.y1), cp.asnumpy(base.y1)[expected_slots])
    if actual.part_indices is None or base.part_indices is None:
        assert actual.part_indices is None and base.part_indices is None
    else:
        np.testing.assert_array_equal(
            cp.asnumpy(actual.part_indices),
            cp.asnumpy(base.part_indices)[expected_slots],
        )
    if actual.ring_indices is None or base.ring_indices is None:
        assert actual.ring_indices is None and base.ring_indices is None
    else:
        np.testing.assert_array_equal(
            cp.asnumpy(actual.ring_indices),
            cp.asnumpy(base.ring_indices)[expected_slots],
        )


@requires_gpu
def test_strict_broadcast_polygon_intersection_preserves_row_cardinality_for_complex_polygons() -> (
    None
):
    """Strict scalar-right polygon intersection keeps one output slot per input row.

    Buffered polygons exceed the SH kernel's vertex workspace, so this exercises
    the row-preserving overlay fallback instead of the direct polygon clip
    kernel.
    """
    left = [
        Point(2, 2).buffer(4),
        Point(3, 4).buffer(4),
        Point(9, 8).buffer(4),
        Point(-12, -15).buffer(4),
    ]
    right = box(0, 0, 10, 10)

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries([right])

    with strict_native_environment():
        result = binary_constructive_owned("intersection", left_owned, right_owned)

    got = result.to_shapely()
    expected = _shapely_oracle("intersection", left, right)

    assert len(got) == len(expected) == 4
    for i, (got_geom, expected_geom) in enumerate(zip(got, expected, strict=True)):
        if expected_geom is None:
            assert got_geom is None, f"Row {i}: expected None, got {got_geom}"
        elif shapely.is_empty(expected_geom):
            assert got_geom is None or shapely.is_empty(got_geom), (
                f"Row {i}: expected empty or null, got {got_geom}"
            )
        else:
            assert got_geom is not None, f"Row {i}: expected non-null polygon result"
            assert got_geom.normalize().equals_exact(expected_geom.normalize(), 1e-9), (
                f"Row {i}: broadcast complex polygon intersection mismatch\n"
                f"  got={got_geom}\n"
                f"  exp={expected_geom}"
            )


def test_polygon_intersection_prefers_canonical_capacity_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aligned polygon intersection enters the canonical capacity router first."""
    left = from_shapely_geometries([Point(0, 0).buffer(2.0), Point(5, 5).buffer(2.0)])
    right = from_shapely_geometries([Point(1, 0).buffer(1.5), Point(5, 5).buffer(1.0)])
    sentinel = from_shapely_geometries([box(1, 1, 2, 2), box(4, 4, 5, 5)])

    monkeypatch.setattr(
        binary_constructive_module,
        "_dispatch_partitioned_polygon_intersection_gpu",
        lambda *args, **kwargs: sentinel,
    )
    monkeypatch.setattr(
        binary_constructive_module,
        "_dispatch_overlay_gpu",
        lambda *args, **kwargs: pytest.fail("bulk overlay should not precede capacity routing"),
    )
    monkeypatch.setattr(
        binary_constructive_module,
        "_dispatch_polygon_intersection_overlay_exact_batch_gpu",
        lambda *args, **kwargs: pytest.fail("rowwise overlay should not precede capacity routing"),
    )

    result = binary_constructive_module._binary_constructive_gpu(
        "intersection",
        left,
        right,
    )

    assert result is sentinel


def test_polygon_difference_propagates_batched_topology_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = from_shapely_geometries([box(0, 0, 2, 2), box(4, 4, 6, 6)])
    right = from_shapely_geometries([box(1, 1, 3, 3), box(4, 4, 5, 5)])

    def _raise_batched(*args, **kwargs):
        raise RuntimeError("batched boom")

    monkeypatch.setattr(
        binary_constructive_module,
        "_dispatch_polygon_difference_overlay_batched_gpu",
        _raise_batched,
    )

    with pytest.raises(RuntimeError, match="batched boom"):
        binary_constructive_module._binary_constructive_gpu(
            "difference",
            left,
            right,
        )
    source = Path(binary_constructive_module.__file__).read_text()
    assert "_dispatch_polygon_difference_overlay_rowwise_gpu_legacy" not in source


def test_polygon_symmetric_difference_propagates_overlay_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = from_shapely_geometries([box(0, 0, 2, 2)])
    right = from_shapely_geometries([box(1, 1, 3, 3)])

    monkeypatch.setattr(
        binary_constructive_module,
        "_dispatch_overlay_gpu",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("overlay boom")),
    )

    with pytest.raises(RuntimeError, match="overlay boom"):
        binary_constructive_module._binary_constructive_gpu(
            "symmetric_difference",
            left,
            right,
        )


@requires_gpu
def test_single_pair_polygon_intersection_uses_exact_overlay_path_for_complex_polygons() -> None:
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
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    expected = left.geometry.iloc[0].intersection(right.geometry.iloc[0]).normalize()

    result = binary_constructive_owned(
        "intersection",
        left_owned,
        right_owned,
        dispatch_mode=ExecutionMode.GPU,
    )

    got = result.to_shapely()[0]
    assert got is not None
    assert got.geom_type == expected.geom_type
    assert got.normalize().equals_exact(expected, tolerance=1e-6)


@requires_gpu
def test_single_pair_polygon_difference_preserves_touch_only_left_polygon() -> None:
    left = from_shapely_geometries([box(-1, 1, 1, 3)])
    right = from_shapely_geometries([box(1, 1, 3, 3)])

    result = binary_constructive_owned(
        "difference",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    got = result.to_shapely()[0]
    expected = box(-1, 1, 1, 3)
    assert got is not None
    assert got.equals_exact(expected, tolerance=1e-9)


@requires_gpu
def test_polygon_difference_capacity_partition_preserves_null_and_empty_semantics() -> None:
    left_geometries = [
        box(0, 0, 2, 2),
        box(0, 0, 2, 2),
        None,
        Polygon(),
        box(0, 0, 3, 3),
    ]
    right_geometries = [
        box(-1, -1, 3, 3),
        box(3, 3, 4, 4),
        box(0, 0, 1, 1),
        box(0, 0, 1, 1),
        box(1, 1, 2, 2),
    ]
    left = from_shapely_geometries(left_geometries)
    right = from_shapely_geometries(right_geometries)

    with strict_native_environment():
        result = binary_constructive_module._dispatch_polygon_difference_overlay_batched_gpu(
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
        )

    assert result is not None
    got = list(result.to_shapely())
    expected = [
        None if lhs is None or rhs is None else lhs.difference(rhs)
        for lhs, rhs in zip(left_geometries, right_geometries, strict=True)
    ]
    for actual, oracle in zip(got, expected, strict=True):
        if oracle is None:
            assert actual is None
        else:
            assert actual is not None
            assert actual.geom_type == oracle.geom_type
            assert shapely.normalize(actual).equals(shapely.normalize(oracle))


@requires_gpu
def test_grouped_polygon_difference_rowwise_matches_oracle_on_redevelopment_rows() -> None:
    """Redevelopment grouped difference wrapper matches the exact Shapely oracle."""
    tmpdir = Path(tempfile.mkdtemp(prefix="test_redev_grouped_diff_"))
    fixtures = setup_fixtures(tmpdir)

    import geopandas as gpd

    parcels = gpd.read_parquet(fixtures["parcels"])
    exclusions = gpd.read_parquet(fixtures["exclusion_zones"])

    bounds = parcels.total_bounds
    dx = (bounds[2] - bounds[0]) * 0.15
    dy = (bounds[3] - bounds[1]) * 0.15
    clip_box = box(bounds[0] + dx, bounds[1] + dy, bounds[2] - dx, bounds[3] - dy)
    study = gpd.clip(parcels, clip_box)
    study = study[study.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    left = study.geometry.values.to_owned()
    right = exclusions.geometry.values.to_owned()

    idx1, idx2 = exclusions.sindex.query(study.geometry, predicate="intersects", sort=True)
    idx1 = np.asarray(idx1)
    idx2 = np.asarray(idx2)
    idx1_unique, idx1_split_at = np.unique(idx1, return_index=True)
    group_offsets = np.concatenate([idx1_split_at, np.asarray([len(idx2)])])
    right_grouped = right.take(idx2)
    right_unioned = segmented_union_all(right_grouped, group_offsets)
    left_grouped = left.take(idx1_unique)

    row_subset = np.asarray([2, 17, 27, 28, 29], dtype=np.int64)
    left_subset = left_grouped.take(row_subset)
    right_subset = right_unioned.take(row_subset)

    result = binary_constructive_module._dispatch_polygon_difference_overlay_rowwise_gpu(
        left_subset,
        right_subset,
        dispatch_mode=ExecutionMode.GPU,
    )
    assert result is not None

    got = list(result.to_shapely())
    expected = [
        left_geom.difference(right_geom)
        for left_geom, right_geom in zip(
            left_subset.to_shapely(),
            right_subset.to_shapely(),
            strict=True,
        )
    ]

    assert len(got) == len(expected)
    for row, (actual, oracle) in enumerate(zip(got, expected, strict=True)):
        if actual is None or oracle is None:
            assert actual is None and oracle is None, f"row {row} null mismatch"
            continue
        assert shapely.normalize(actual).equals(shapely.normalize(oracle)), (
            f"row {row} grouped rowwise difference mismatch"
        )


@requires_gpu
def test_grouped_polygon_difference_batched_matches_oracle_on_redevelopment_prefix() -> None:
    """The batched row-isolated difference path stays exact on the first failing prefix."""
    tmpdir = Path(tempfile.mkdtemp(prefix="test_redev_grouped_diff_prefix_"))
    fixtures = setup_fixtures(tmpdir)

    import geopandas as gpd

    parcels = gpd.read_parquet(fixtures["parcels"])
    exclusions = gpd.read_parquet(fixtures["exclusion_zones"])

    bounds = parcels.total_bounds
    dx = (bounds[2] - bounds[0]) * 0.15
    dy = (bounds[3] - bounds[1]) * 0.15
    clip_box = box(bounds[0] + dx, bounds[1] + dy, bounds[2] - dx, bounds[3] - dy)
    study = gpd.clip(parcels, clip_box)
    study = study[study.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    left = study.geometry.values.to_owned()
    right = exclusions.geometry.values.to_owned()

    idx1, idx2 = exclusions.sindex.query(study.geometry, predicate="intersects", sort=True)
    idx1 = np.asarray(idx1)
    idx2 = np.asarray(idx2)
    idx1_unique, idx1_split_at = np.unique(idx1, return_index=True)
    group_offsets = np.concatenate([idx1_split_at, np.asarray([len(idx2)])])
    right_grouped = right.take(idx2)
    right_unioned = segmented_union_all(right_grouped, group_offsets)
    left_grouped = left.take(idx1_unique)

    prefix = np.arange(16, dtype=np.int64)
    left_prefix = left_grouped.take(prefix)
    right_prefix = right_unioned.take(prefix)

    with strict_native_environment():
        result = binary_constructive_module._dispatch_polygon_difference_overlay_rowwise_gpu(
            left_prefix,
            right_prefix,
            dispatch_mode=ExecutionMode.GPU,
        )

    assert result is not None
    got = list(result.to_shapely())
    expected = [
        left_geom.difference(right_geom)
        for left_geom, right_geom in zip(
            left_prefix.to_shapely(),
            right_prefix.to_shapely(),
            strict=True,
        )
    ]

    assert len(got) == len(expected)
    for row, (actual, oracle) in enumerate(zip(got, expected, strict=True)):
        if actual is None or oracle is None:
            assert actual is None and oracle is None, f"row {row} null mismatch"
            continue
        assert shapely.normalize(actual).equals(shapely.normalize(oracle)), (
            f"row {row} batched grouped difference mismatch"
        )


@requires_gpu
def test_grouped_polygon_difference_strict_matches_oracle_on_redevelopment_prefix_16() -> None:
    """The 16-row redevelopment prefix stays exact under strict row isolation."""
    tmpdir = Path(tempfile.mkdtemp(prefix="test_redev_grouped_diff_prefix16_"))
    fixtures = setup_fixtures(tmpdir)

    import geopandas as gpd

    parcels = gpd.read_parquet(fixtures["parcels"])
    exclusions = gpd.read_parquet(fixtures["exclusion_zones"])

    bounds = parcels.total_bounds
    dx = (bounds[2] - bounds[0]) * 0.15
    dy = (bounds[3] - bounds[1]) * 0.15
    clip_box = box(bounds[0] + dx, bounds[1] + dy, bounds[2] - dx, bounds[3] - dy)
    study = gpd.clip(parcels, clip_box)
    study = study[study.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    left = study.geometry.values.to_owned()
    right = exclusions.geometry.values.to_owned()

    idx1, idx2 = exclusions.sindex.query(study.geometry, predicate="intersects", sort=True)
    idx1 = np.asarray(idx1)
    idx2 = np.asarray(idx2)
    idx1_unique, idx1_split_at = np.unique(idx1, return_index=True)
    group_offsets = np.concatenate([idx1_split_at, np.asarray([len(idx2)])])
    right_grouped = right.take(idx2)
    right_unioned = segmented_union_all(right_grouped, group_offsets)
    left_grouped = left.take(idx1_unique)

    prefix = np.arange(16, dtype=np.int64)
    left_prefix = left_grouped.take(prefix)
    right_prefix = right_unioned.take(prefix)

    with strict_native_environment():
        result = binary_constructive_module._dispatch_polygon_difference_overlay_rowwise_gpu(
            left_prefix,
            right_prefix,
            dispatch_mode=ExecutionMode.GPU,
        )
    assert result is not None

    got = list(result.to_shapely())
    expected = [
        left_geom.difference(right_geom)
        for left_geom, right_geom in zip(
            left_prefix.to_shapely(),
            right_prefix.to_shapely(),
            strict=True,
        )
    ]

    assert len(got) == len(expected)
    for row, (actual, oracle) in enumerate(zip(got, expected, strict=True)):
        if actual is None or oracle is None:
            assert actual is None and oracle is None, f"row {row} null mismatch"
            continue
        assert shapely.normalize(actual).equals(shapely.normalize(oracle)), (
            f"row {row} grouped strict difference mismatch"
        )


@requires_gpu
@pytest.mark.parametrize(
    ("right_geom", "label"),
    [
        (Point(1, 1), "point"),
        (MultiPoint([(1, 1), (3, 3)]), "multipoint"),
    ],
)
def test_single_pair_polygon_difference_preserves_left_for_lower_dim_right(
    right_geom,
    label: str,
) -> None:
    left = from_shapely_geometries([box(0, 0, 4, 4)])
    right = from_shapely_geometries([right_geom])

    result = binary_constructive_owned(
        "difference",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    got = result.to_shapely()[0]
    expected = box(0, 0, 4, 4)
    assert got is not None, f"{label} difference unexpectedly returned null"
    assert got.equals_exact(expected, tolerance=1e-9), (
        f"{label} difference should preserve the left polygon exactly"
    )


@requires_gpu
def test_polygon_union_batches_aligned_overlay_candidate_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = [
        Point(0, 0).buffer(10, resolution=32),
        Point(40, 0).buffer(10, resolution=32),
        Point(0, 40).buffer(10, resolution=32),
        Point(40, 40).buffer(10, resolution=32),
    ]
    right = [
        Point(5, 0).buffer(10, resolution=32),
        Point(45, 0).buffer(10, resolution=32),
        Point(5, 40).buffer(10, resolution=32),
        Point(45, 40).buffer(10, resolution=32),
    ]

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()

    result = binary_constructive_owned(
        "union",
        from_shapely_geometries(left),
        from_shapely_geometries(right),
        dispatch_mode=ExecutionMode.GPU,
    )

    got = result.to_shapely()
    expected = shapely.union(
        np.asarray(left, dtype=object), np.asarray(right, dtype=object)
    ).tolist()
    assert len(got) == len(expected) == 4
    for got_geom, expected_geom in zip(got, expected, strict=True):
        assert got_geom is not None
        assert got_geom.normalize().equals_exact(expected_geom.normalize(), tolerance=1e-9)

    summary = {entry["name"]: entry["calls"] for entry in summarize_hotpath_trace()}
    # The aligned batched helper should still keep candidate generation
    # bounded. Degenerate touch/disjoint rows now take the corrected
    # single-row fallback path, so this mixed workload legitimately does
    # more than the ideal all-area-overlap case, but it should remain well
    # below a naive overlay-per-row-everything shape.
    assert summary.get("segment.classify.generate_candidates", 0) <= 5
    assert summary.get("overlay.split.classify_intersections", 0) <= 4


@requires_gpu
def test_polygon_difference_batches_aligned_overlay_candidate_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = [
        box(0, 0, 4, 4),  # partial overlap
        box(10, 0, 14, 4),  # touch-only
        box(20, 0, 24, 4),  # full overlap
        box(30, 0, 34, 4),  # disjoint
    ]
    right = [
        box(2, 0, 6, 4),
        box(14, 0, 18, 4),
        box(20, 0, 24, 4),
        box(40, 0, 44, 4),
    ]

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()

    result = binary_constructive_owned(
        "difference",
        from_shapely_geometries(left),
        from_shapely_geometries(right),
        dispatch_mode=ExecutionMode.GPU,
    )

    got = result.to_shapely()
    expected = shapely.difference(
        np.asarray(left, dtype=object), np.asarray(right, dtype=object)
    ).tolist()
    assert len(got) == len(expected) == 4
    for got_geom, expected_geom in zip(got, expected, strict=True):
        if expected_geom is None:
            assert got_geom is None
        elif shapely.is_empty(expected_geom):
            assert got_geom is None or shapely.is_empty(got_geom)
        else:
            assert got_geom is not None
            assert got_geom.normalize().equals_exact(expected_geom.normalize(), tolerance=1e-9)

    summary = {entry["name"]: entry["calls"] for entry in summarize_hotpath_trace()}
    # Degenerate touch/disjoint rows now take the corrected single-row
    # difference fallback, so this mixed workload legitimately does more
    # work than the pure batched area-overlap case while staying bounded.
    assert summary.get("segment.classify.generate_candidates", 0) <= 6
    assert summary.get("overlay.split.classify_intersections", 0) <= 5


@requires_gpu
def test_polygon_intersection_single_pair_uses_same_row_candidate_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = from_shapely_geometries(
        [
            Point(0.0, 0.0).buffer(10.0, resolution=64),
        ]
    )
    right = from_shapely_geometries(
        [
            shapely.Polygon(
                [
                    (12.0, 0.0),
                    (4.0, 2.0),
                    (0.0, 12.0),
                    (-4.0, 2.0),
                    (-12.0, 0.0),
                    (-4.0, -2.0),
                    (0.0, -12.0),
                    (4.0, -2.0),
                    (12.0, 0.0),
                ]
            ),
        ]
    )

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()

    result = binary_constructive_owned(
        "intersection",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    got = result.to_shapely()[0]
    expected = shapely.intersection(
        np.asarray(left.to_shapely(), dtype=object),
        np.asarray(right.to_shapely(), dtype=object),
    ).tolist()[0]
    assert got is not None
    assert got.normalize().equals_exact(expected.normalize(), tolerance=1e-9)

    summary = {entry["name"]: entry["calls"] for entry in summarize_hotpath_trace()}
    assert summary.get("segment.candidates.same_row_fast_path") == 1
    assert "segment.candidates.binary_search" not in summary


@requires_gpu
def test_multipart_direct_pack_refines_bbox_overlap_with_device_predicate() -> None:
    """Overlapping envelopes alone must not force the grouped union carrier."""
    constructive_module = importlib.import_module("vibespatial.constructive.binary_constructive")
    cp = pytest.importorskip("cupy")

    parts = from_shapely_geometries(
        [
            Polygon([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.0, 0.0)]),
            Polygon([(1.1, 3.0), (3.0, 1.1), (3.0, 3.0), (1.1, 3.0)]),
        ],
        residency=Residency.DEVICE,
    )

    result = constructive_module._pack_disjoint_multipart_intersection_parts_gpu(
        parts,
        cp.zeros(parts.row_count, dtype=cp.int32),
        output_row_count=1,
    )

    assert result is not None
    assert result.row_count == 1
    assert shapely.normalize(result.to_shapely()[0]).equals(
        shapely.normalize(shapely.union_all(parts.to_shapely()))
    )


@requires_gpu
def test_multipart_capacity_pack_preserves_valid_empty_and_null_rows() -> None:
    constructive_module = importlib.import_module("vibespatial.constructive.binary_constructive")
    cp = pytest.importorskip("cupy")

    empty = shapely.intersection(box(0, 0, 1, 1), box(2, 2, 3, 3))
    parts = from_shapely_geometries(
        [empty, None, box(4, 4, 5, 5)],
        residency=Residency.DEVICE,
    )
    result = constructive_module._pack_disjoint_multipart_intersection_parts_gpu(
        parts,
        cp.arange(3, dtype=cp.int32),
        output_row_count=3,
        assume_disjoint=True,
    )

    assert result is not None
    got = result.to_shapely()
    assert got[0] is not None and got[0].is_empty
    assert got[1] is None
    assert got[2].equals(box(4, 4, 5, 5))


@requires_gpu
def test_boundary_segment_capacity_ignores_inactive_duplicate_tail() -> None:
    cp = pytest.importorskip("cupy")
    from vibespatial.overlay.boundary_graph import (
        undirected_boundary_segment_orders_gpu,
    )

    start_x = cp.asarray([0.0, 1.0, 1.0, 0.0, 0.0])
    start_y = cp.asarray([0.0, 0.0, 1.0, 1.0, 0.0])
    end_x = cp.asarray([1.0, 1.0, 0.0, 0.0, 1.0])
    end_y = cp.asarray([0.0, 1.0, 1.0, 0.0, 0.0])
    active = cp.asarray([True, True, True, True, False])

    order = undirected_boundary_segment_orders_gpu(
        start_x,
        start_y,
        end_x,
        end_y,
        cp.zeros(5, dtype=cp.int32),
        active,
    )

    assert sorted(cp.asnumpy(order).tolist()) == [0, 1, 2, 3]


@requires_gpu
def test_native_grouped_multipart_union_uses_structural_same_row_span_proof() -> None:
    constructive_module = importlib.import_module("vibespatial.constructive.binary_constructive")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    parts = from_shapely_geometries(
        [
            box(0.0, 0.0, 2.0, 2.0),
            box(1.0, 0.0, 3.0, 2.0),
            box(10.0, 0.0, 12.0, 2.0),
            box(11.0, 0.0, 13.0, 2.0),
        ],
        residency=Residency.DEVICE,
    )
    sorted_order = cp.arange(parts.row_count, dtype=cp.int64)
    group_offsets = cp.asarray([0, 2, 4], dtype=cp.int64)
    group_ids = cp.asarray([0, 1], dtype=cp.int64)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = constructive_module._regroup_native_grouped_parts_with_grouped_union_gpu(
        parts,
        sorted_order,
        group_offsets,
        group_ids,
        output_row_count=2,
        allow_direct_disjoint_pack=False,
        use_same_row_fast_path=True,
    )
    events = get_d2h_transfer_events(clear=True)

    assert result is not None
    assert result.row_count == 2
    got = result.to_shapely()
    assert shapely.normalize(got[0]).equals(
        shapely.normalize(shapely.union_all(parts.to_shapely()[:2]))
    )
    assert shapely.normalize(got[1]).equals(
        shapely.normalize(shapely.union_all(parts.to_shapely()[2:]))
    )
    assert "segment same-row span summary scalar fence" not in {event.reason for event in events}


@requires_gpu
def test_multipolygon_polygon_intersection_refines_bbox_overlap_before_union_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = from_shapely_geometries(
        [
            MultiPolygon(
                [
                    Polygon([(0, 0), (3, 0), (0, 3), (0, 0)]),
                    Polygon([(4, 4), (1, 4), (4, 1), (4, 4)]),
                ]
            ),
            MultiPolygon(
                [
                    Polygon([(10, 0), (13, 0), (10, 3), (10, 0)]),
                    Polygon([(14, 4), (11, 4), (14, 1), (14, 4)]),
                ]
            ),
        ]
    )
    right = from_shapely_geometries(
        [
            box(-1, -1, 5, 5),
            box(9, -1, 15, 5),
        ]
    )

    overlay_gpu_module = importlib.import_module("vibespatial.overlay.gpu")
    segmented_union_module = importlib.import_module(
        "vibespatial.kernels.constructive.segmented_union"
    )

    original_materialize = overlay_gpu_module._materialize_overlay_execution_plan
    original_segmented_union_all = segmented_union_module.segmented_union_all
    union_materialize_calls = 0
    segmented_union_calls = 0

    def _counted_materialize(*args, **kwargs):
        nonlocal union_materialize_calls
        if kwargs.get("operation") == "union":
            union_materialize_calls += 1
        return original_materialize(*args, **kwargs)

    def _counted_segmented_union_all(*args, **kwargs):
        nonlocal segmented_union_calls
        segmented_union_calls += 1
        return original_segmented_union_all(*args, **kwargs)

    monkeypatch.setattr(
        overlay_gpu_module,
        "_materialize_overlay_execution_plan",
        _counted_materialize,
    )
    monkeypatch.setattr(
        segmented_union_module,
        "segmented_union_all",
        _counted_segmented_union_all,
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
    for row, (got_geom, expected_geom) in enumerate(zip(got, expected, strict=True)):
        if expected_geom is None:
            assert got_geom is None, f"row {row}: expected null"
            continue
        if shapely.is_empty(expected_geom):
            assert got_geom is None or shapely.is_empty(got_geom), (
                f"row {row}: expected empty or null, got {got_geom}"
            )
            continue
        assert got_geom is not None
        assert shapely.normalize(got_geom).equals(shapely.normalize(expected_geom)), (
            f"row {row}: regrouped multipart intersection mismatch"
        )

    assert union_materialize_calls == 0
    assert segmented_union_calls == 0


@requires_gpu
def test_materialize_device_broadcast_stays_device_without_host_resolution() -> None:
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    source = from_shapely_geometries(
        [box(0.0, 0.0, 3.0, 2.0)],
        residency=Residency.DEVICE,
    )
    tiled = tile_single_row(source, 5)

    assert tiled.is_indexed_view

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    materialized = materialize_broadcast(tiled)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert materialized.residency is Residency.DEVICE
    assert not materialized.is_indexed_view
    assert materialized.row_count == 5
    assert materialized.families[GeometryFamily.POLYGON].row_count == 5
    assert "materialized broadcast-right geometry on device" not in runtime_reasons
    assert "owned geometry polygon coordinate-x materialization boundary" not in runtime_reasons
    assert "owned geometry host metadata validity boundary" not in runtime_reasons


@requires_gpu
def test_device_broadcast_polygon_intersection_uses_row_indirected_right() -> None:
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geoms = [
        box(0.0, 0.0, 4.0, 4.0),
        box(5.0, 0.0, 8.0, 3.0),
        Polygon([(10.0, 0.0), (14.0, 0.0), (13.0, 4.0), (10.0, 3.0)]),
    ]
    right_geom = box(1.0, 1.0, 12.0, 2.5)
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries([right_geom], residency=Residency.DEVICE)
    tiled = tile_single_row(right, len(left_geoms))

    assert tiled.is_indexed_view
    assert hasattr(tiled._index_map, "__cuda_array_interface__")
    assert bool(cp.all(tiled._index_map == 0))

    source = Path(binary_constructive_module.__file__).read_text()
    assert "materialize_broadcast" not in source

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = binary_constructive_owned(
        "intersection",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result.residency is Residency.DEVICE
    assert "owned geometry device-take nested slice-size allocation fence" not in runtime_reasons
    assert "owned geometry device-take slice-size allocation fence" not in runtime_reasons
    got = result.to_shapely()
    expected = shapely.intersection(
        np.asarray(left_geoms, dtype=object),
        right_geom,
    ).tolist()
    for actual, oracle in zip(got, expected, strict=True):
        assert actual is not None
        assert shapely.normalize(actual).equals(shapely.normalize(oracle))

    events = get_dispatch_events(clear=True)
    assert any(
        event.implementation == "polygon_intersection_partitioned_capacity_gpu" for event in events
    )


@requires_gpu
def test_exact_broadcast_polygon_topology_uses_bounded_complete_ring_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibespatial.constructive.binary_constructive import (
        _dispatch_polygon_intersection_overlay_broadcast_right_gpu,
    )
    from vibespatial.geometry.owned import OwnedGeometryArray
    from vibespatial.overlay import gpu as overlay_gpu
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events
    from vibespatial.spatial.prepared_polygon_mask import (
        PreparedPolygonMask,
        prepared_polygon_mask_fp64_plan,
    )

    left_geoms = [
        box(0.0, 0.0, 3.0, 3.0),
        box(3.8, 1.8, 4.2, 2.2),
        box(8.0, 0.0, 9.0, 1.0),
    ]
    mask = Polygon(
        [(1.0, 1.0), (6.0, 1.0), (6.0, 4.0), (4.0, 4.0), (4.0, 2.0), (1.0, 2.0)]
    )
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries([mask], residency=Residency.DEVICE)

    def _reject_physical_broadcast(self, rows, *args, **kwargs):
        raise AssertionError("broadcast-right topology physically repeated geometry rows")

    monkeypatch.setattr(
        OwnedGeometryArray,
        "_physical_device_take",
        _reject_physical_broadcast,
    )
    monkeypatch.setattr(
        overlay_gpu,
        "_compute_live_split_event_budget",
        lambda: 120,
    )
    clear_dispatch_events()
    prepared = PreparedPolygonMask.from_owned(
        right,
        precision_plan=prepared_polygon_mask_fp64_plan(),
    )
    assert prepared is not None
    try:
        result = _dispatch_polygon_intersection_overlay_broadcast_right_gpu(
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
            _prepared_mask=prepared,
        )
    finally:
        prepared.close()

    assert result is not None
    expected = shapely.intersection(np.asarray(left_geoms, dtype=object), mask)
    for actual, oracle in zip(result.to_shapely(), expected, strict=True):
        if oracle.is_empty:
            assert actual is None or actual.is_empty
        else:
            assert actual is not None
            assert shapely.normalize(actual).equals(shapely.normalize(oracle))

    events = get_dispatch_events(clear=True)
    event = next(
        event
        for event in events
        if event.implementation == "broadcast_right_ring_local_winding_topology_gpu"
    )
    assert "physical_mask_segments=6" in event.detail
    assert "candidate_rings=2" in event.detail
    assert "logical_right_segments=12" in event.detail
    assert "complete_ring_candidates_plus_ancestor_shell_baseline" in event.detail
    assert 12 < 3 * 6


@requires_gpu
def test_exact_broadcast_polygon_coverage_respects_logical_right_row() -> None:
    import cupy as cp

    from vibespatial.constructive.binary_constructive import (
        _dispatch_polygon_intersection_overlay_broadcast_right_gpu,
    )

    left_geoms = [
        box(1.0, 1.0, 2.0, 2.0),
        box(20.0, 20.0, 21.0, 21.0),
    ]
    right_source = from_shapely_geometries(
        [
            box(-100.0, -100.0, 100.0, 100.0),
            Polygon([(0.0, 0.0), (4.0, 0.0), (2.0, 4.0)]),
        ],
        residency=Residency.DEVICE,
    )
    right = right_source.device_take(cp.asarray([1], dtype=cp.int64))
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)

    result = _dispatch_polygon_intersection_overlay_broadcast_right_gpu(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert result is not None
    expected = shapely.intersection(
        np.asarray(left_geoms, dtype=object),
        Polygon([(0.0, 0.0), (4.0, 0.0), (2.0, 4.0)]),
    )
    for actual, oracle in zip(result.to_shapely(), expected, strict=True):
        if oracle.is_empty:
            assert actual is None or actual.is_empty
        else:
            assert actual is not None
            assert shapely.normalize(actual).equals(shapely.normalize(oracle))


@requires_gpu
def test_prepared_broadcast_mask_partitions_contained_exterior_and_boundary_rows() -> None:
    import cupy as cp

    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events
    from vibespatial.spatial.prepared_polygon_mask import (
        PreparedPolygonMask,
        prepared_polygon_mask_fp64_plan,
    )

    mask = Polygon(
        [(0, 0), (12, 0), (12, 12), (0, 12), (0, 0)],
        holes=[[(4, 4), (8, 4), (8, 8), (4, 8), (4, 4)]],
    )
    left_geoms = [
        box(1, 1, 2, 2),
        box(5, 5, 6, 6),
        box(3.5, 5, 4.5, 6),
        box(13, 1, 14, 2),
        box(-1, 9, 1, 11),
    ]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries([mask], residency=Residency.DEVICE)

    clear_dispatch_events()
    prepared = PreparedPolygonMask.from_owned(
        right,
        precision_plan=prepared_polygon_mask_fp64_plan(),
    )
    assert prepared is not None
    classification = prepared.classify_polygon_rows(left)
    assert classification is not None
    cp.cuda.get_current_stream().synchronize()

    assert cp.asnumpy(classification.valid).tolist() == [True] * 5
    assert cp.asnumpy(classification.covered_by).tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert cp.asnumpy(classification.exterior).tolist() == [
        False,
        True,
        False,
        True,
        False,
    ]
    assert cp.asnumpy(classification.boundary_unresolved).tolist() == [
        False,
        False,
        True,
        False,
        True,
    ]
    assert any(
        event.implementation == "prepared_single_mask_indexed_relation_exact_ray_gpu"
        and "candidate_work=" in event.detail
        and "scheduled_index_lane_bound=" in event.detail
        and "exact_lane_bound=" in event.detail
        and "total_scheduled_lane_bound=" in event.detail
        and "candidate_tile_capacity=" in event.detail
        and "precision=fp64" in event.detail
        and "precision_reason=" in event.detail
        for event in get_dispatch_events(clear=True)
    )
    prepared.close()


@requires_gpu
def test_prepared_broadcast_exact_topology_is_complete_ring_local_with_shell_baselines() -> None:
    from vibespatial.api._native_relation import NativeRelation
    from vibespatial.constructive.binary_constructive import (
        _dispatch_polygon_intersection_overlay_broadcast_right_gpu,
    )
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events
    from vibespatial.spatial.prepared_polygon_mask import (
        PreparedPolygonMask,
        prepared_polygon_mask_fp64_plan,
    )

    first = Polygon(
        [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)],
        holes=[
            [(4, 4), (8, 4), (8, 8), (4, 8), (4, 4)],
            [(12, 12), (16, 12), (16, 16), (12, 16), (12, 12)],
        ],
    )
    second = Polygon(
        [(100, 0), (120, 0), (120, 20), (100, 20), (100, 0)],
        holes=[[(104, 4), (108, 4), (108, 8), (104, 8), (104, 4)]],
    )
    third = Polygon(
        [(200, 0), (220, 0), (220, 20), (200, 20), (200, 0)],
    )
    mask = MultiPolygon([first, second, third])
    left_geoms = [
        box(3, 5, 5, 7),
        box(103, 5, 105, 7),
        box(199, 9, 201, 11),
    ]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries([mask], residency=Residency.DEVICE)
    prepared = PreparedPolygonMask.from_owned(
        right,
        precision_plan=prepared_polygon_mask_fp64_plan(),
    )
    assert prepared is not None
    ring_local = prepared.complete_ring_relation(left)
    assert isinstance(ring_local.ring_relation, NativeRelation)
    assert hasattr(ring_local.ring_relation.left_indices, "__cuda_array_interface__")
    assert hasattr(ring_local.ring_relation.right_indices, "__cuda_array_interface__")
    assert ring_local.candidate_ring_count == 5

    clear_dispatch_events()
    try:
        result = _dispatch_polygon_intersection_overlay_broadcast_right_gpu(
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
            _prepared_mask=prepared,
        )
    finally:
        prepared.close()

    assert result is not None
    expected = shapely.intersection(np.asarray(left_geoms, dtype=object), mask)
    for actual, oracle in zip(result.to_shapely(), expected, strict=True):
        assert actual is not None
        assert shapely.normalize(actual).equals(shapely.normalize(oracle))

    event = next(
        event
        for event in get_dispatch_events(clear=True)
        if event.implementation == "broadcast_right_ring_local_winding_topology_gpu"
    )
    assert "physical_mask_segments=24" in event.detail
    assert "candidate_rings=5" in event.detail
    assert "logical_right_segments=20" in event.detail
    assert "complete_ring_candidates_plus_ancestor_shell_baseline" in event.detail
    assert 20 < 3 * 24


def test_prepared_mask_shape_bounds_virtual_candidate_capacity_by_live_memory() -> None:
    from vibespatial.spatial.prepared_polygon_mask import (
        _MORTON_SPAN_BUCKET_UPPER_BOUNDS,
        _plan_mask_classification_shape,
    )

    span_bucket_counts = [0] * len(_MORTON_SPAN_BUCKET_UPPER_BOUNDS)
    span_bucket_counts[1] = 2_000_000
    constrained = _plan_mask_classification_shape(
        row_count=1_000_000,
        segment_count=1_000,
        free_device_bytes=64 << 20,
        span_bucket_counts=span_bucket_counts,
    )
    roomier = _plan_mask_classification_shape(
        row_count=1_000_000,
        segment_count=1_000,
        free_device_bytes=512 << 20,
        span_bucket_counts=span_bucket_counts,
    )

    assert constrained.dense_candidate_work == 1_000_000_000
    assert constrained.candidate_work == 2_000_000
    assert constrained.scheduled_index_lane_bound == 4_000_000
    assert constrained.exact_lane_bound == 2_000_000
    assert constrained.total_scheduled_lane_bound == 6_000_000
    assert constrained.output_bytes == 4_000_000
    assert constrained.candidate_work < constrained.dense_candidate_work
    assert 0 < constrained.candidate_tile_capacity < constrained.candidate_work
    assert (
        constrained.row_tile_size * constrained.segment_tile_size
        <= constrained.candidate_tile_capacity
    )
    assert constrained.tile_count > 1
    assert roomier.candidate_tile_capacity > constrained.candidate_tile_capacity


def test_prepared_mask_classification_does_not_materialize_candidate_relations() -> None:
    from vibespatial.spatial import prepared_polygon_mask as prepared_module

    source = Path(prepared_module.__file__).read_text()
    tree = ast.parse(source)
    classify = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PreparedPolygonMask"
        for node in node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "classify_polygon_rows"
    )

    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(classify))
    classify_source = ast.get_source_segment(source, classify)
    assert classify_source is not None
    assert "generate_bounds_pairs" not in classify_source
    assert "boundary_pairs" not in source
    assert "ray_pairs" not in source
    assert "cp.bincount" not in source
    assert "_prepare_morton_range_query" in source
    assert 'kernels["morton_range_tile_count"]' in source
    assert 'kernels["morton_range_tile_scatter"]' in source
    assert "exclusive_sum" in source
    assert "count_scatter_total" not in source
    assert "spatial_index_device_query" not in source
    assert "vs_mask_orient2d_exact" in source
    assert "precision_plan: PrecisionPlan" in source
    assert "_require_prepared_polygon_mask_precision_plan" in source

    constructive_source = Path(binary_constructive_module.__file__).read_text()
    dispatch_start = constructive_source.index(
        "def _dispatch_prepared_polygon_intersection_broadcast_right_gpu("
    )
    dispatch_end = constructive_source.index("\ndef ", dispatch_start + 1)
    dispatch_source = constructive_source[dispatch_start:dispatch_end]
    assert ".compact_rowset(" not in dispatch_source
    assert "device_physicalize_owned_row_selections_exact" in dispatch_source
    assert "compact_concrete_prefix=True" in dispatch_source
    assert "sparse_boundary_relation_plus_exact_physicalization" in dispatch_source

    empty_semantics_start = constructive_source.index(
        "def _apply_binary_empty_row_semantics_gpu("
    )
    empty_semantics_end = constructive_source.index(
        "\ndef ",
        empty_semantics_start + 1,
    )
    empty_semantics_source = constructive_source[
        empty_semantics_start:empty_semantics_end
    ]
    assert "device_mask_owned_capacity(" in empty_semantics_source
    assert "_compose_aligned_native_geometries(" in empty_semantics_source
    assert ".compact_rowset(" not in empty_semantics_source
    assert "_device_indexed_take(" not in empty_semantics_source


def test_prepared_mask_precision_contract_is_explicit_fp64_predicate() -> None:
    from dataclasses import replace

    from vibespatial.runtime.precision import KernelClass, PrecisionMode
    from vibespatial.spatial.prepared_polygon_mask import (
        _require_prepared_polygon_mask_precision_plan,
        prepared_polygon_mask_fp64_plan,
    )

    plan = prepared_polygon_mask_fp64_plan()
    assert plan.kernel_class is KernelClass.PREDICATE
    assert plan.storage_precision is PrecisionMode.FP64
    assert plan.compute_precision is PrecisionMode.FP64
    assert _require_prepared_polygon_mask_precision_plan(plan) is plan

    with pytest.raises(ValueError, match="PREDICATE PrecisionPlan"):
        _require_prepared_polygon_mask_precision_plan(
            replace(plan, kernel_class=KernelClass.CONSTRUCTIVE)
        )
    with pytest.raises(NotImplementedError, match="authoritative fp64"):
        _require_prepared_polygon_mask_precision_plan(
            replace(plan, compute_precision=PrecisionMode.FP32)
        )


@requires_gpu
def test_prepared_mask_ray_crossing_uses_exact_orientation_sign() -> None:
    import cupy as cp

    from vibespatial.spatial.prepared_polygon_mask import (
        _classify_mask_candidate_relation_tile_device,
        prepared_polygon_mask_fp64_plan,
    )

    point_x = float(1.0 / 10.0)
    direct_cross_x = 0.0 + (1.0 - 0.0) * (1.0 - 0.0) / (10.0 - 0.0)
    assert direct_cross_x == point_x
    assert not (direct_cross_x < point_x)

    flags, candidate_capacity = _classify_mask_candidate_relation_tile_device(
        cp.asarray([1], dtype=cp.int32),
        cp.asarray([0], dtype=cp.int32),
        cp.asarray([1], dtype=cp.int64),
        cp.asarray([point_x], dtype=cp.float64),
        cp.asarray([1.0], dtype=cp.float64),
        cp.asarray([0.0, 1.0], dtype=cp.float64),
        cp.asarray([0.0, 10.0], dtype=cp.float64),
        1,
        precision_plan=prepared_polygon_mask_fp64_plan(),
    )
    cp.cuda.get_current_stream().synchronize()
    actual = int(flags.get()[0])

    assert candidate_capacity == 1
    assert actual & 1 == 0
    assert actual & 2 == 2


@requires_gpu
def test_polygonal_multipart_difference_uses_one_batch_topology_plan() -> None:
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geoms = [box(0, 0, 10, 10), box(20, 0, 30, 10)]
    right_geoms = [
        MultiPolygon([box(-1, 2, 3, 4), box(7, 6, 11, 8)]),
        MultiPolygon([box(19, 1, 23, 3), box(27, 7, 31, 9)]),
    ]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = binary_constructive_owned(
        "difference",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    assert "polygonal difference polygon-part depth scalar fence" not in reasons
    expected = [lhs.difference(rhs) for lhs, rhs in zip(left_geoms, right_geoms, strict=True)]
    for actual, oracle in zip(result.to_shapely(), expected, strict=True):
        assert shapely.normalize(actual).equals(shapely.normalize(oracle))

    implementations = [event.implementation for event in get_dispatch_events(clear=True)]
    assert "polygonal_multipart_grouped_difference_gpu" not in implementations
    assert any(
        implementation.startswith("row_aligned_difference_") for implementation in implementations
    )


@requires_gpu
def test_lineal_polygonal_difference_subtracts_all_multipart_polygon_parts_on_device() -> None:
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    polygon_with_hole = Polygon(
        [(1, -1), (5, -1), (5, 1), (1, 1), (1, -1)],
        [[(2, -0.5), (4, -0.5), (4, 0.5), (2, 0.5), (2, -0.5)]],
    )
    left_geoms = [
        LineString([(0, 0), (6, 0)]),
        MultiLineString([[(0, 0), (4, 0)], [(0, 4), (6, 4)]]),
        LineString([(-1, 0), (2, 0)]),
        LineString(),
        LineString([(0, 0), (3, 0)]),
    ]
    right_geoms = [
        polygon_with_hole,
        MultiPolygon([box(1, -1, 2, 1), box(3, 3, 5, 5)]),
        box(0, 0, 1, 1),
        box(0, 0, 1, 1),
        None,
    ]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = binary_constructive_owned(
        "difference",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert result.residency is Residency.DEVICE
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    assert "lineal-polygonal difference polygon-part depth scalar fence" not in reasons
    got = result.to_shapely()
    expected = [lhs.difference(rhs) for lhs, rhs in zip(left_geoms, right_geoms, strict=True)]
    for actual, oracle in zip(got, expected, strict=True):
        if oracle is None:
            assert actual is None
        else:
            assert shapely.normalize(actual).equals(shapely.normalize(oracle))

    events = get_dispatch_events(clear=True)
    assert any(
        event.implementation == "lineal_polygonal_collective_split_topology_gpu" for event in events
    )


@requires_gpu
def test_rectangle_containment_difference_emits_hole_without_segment_extraction() -> None:
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geoms = [box(0, 0, 10, 10), box(20, 0, 30, 10)]
    right_geoms = [box(2, 2, 4, 4), box(23, 3, 27, 7)]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = binary_constructive_owned(
        "difference",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result.residency is Residency.DEVICE
    got = result.to_shapely()
    expected = [lhs.difference(rhs) for lhs, rhs in zip(left_geoms, right_geoms, strict=True)]
    for actual, oracle in zip(got, expected, strict=True):
        assert actual.equals(oracle)
    assert "segment extraction total-segments allocation fence" not in runtime_reasons
    events = get_dispatch_events(clear=True)
    assert any(
        event.implementation == "row_aligned_polygon_hole_difference_gpu"
        for event in events
    )


@requires_gpu
def test_nonrect_containment_difference_emits_native_hole_capacity() -> None:
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    left_geoms = [
        Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            holes=[[(7, 7), (8, 7), (8, 8), (7, 8), (7, 7)]],
        ),
        Polygon([(20, 0), (31, 0), (30, 10), (20, 9), (20, 0)]),
    ]
    right_geoms = [
        Polygon([(1, 1), (5, 1), (4, 5), (1, 4), (1, 1)]),
        Polygon([(22, 2), (27, 1), (28, 6), (23, 7), (22, 2)]),
    ]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = binary_constructive_owned(
        "difference",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    expected = [lhs.difference(rhs) for lhs, rhs in zip(left_geoms, right_geoms, strict=True)]
    for actual, oracle in zip(result.to_shapely(), expected, strict=True):
        assert shapely.equals(shapely.normalize(actual), shapely.normalize(oracle))
    assert "segment extraction total-segments allocation fence" not in runtime_reasons
    assert any(
        event.implementation == "row_aligned_polygon_hole_difference_gpu"
        for event in get_dispatch_events(clear=True)
    )


# ---------------------------------------------------------------------------
# 7. tile_single_row unit tests
# ---------------------------------------------------------------------------


def test_tile_single_row_metadata() -> None:
    """tile_single_row produces correct metadata arrays."""
    geom = box(0, 0, 1, 1)
    owned = from_shapely_geometries([geom])
    tiled = tile_single_row(owned, 5)

    assert tiled.row_count == 5
    assert len(tiled.validity) == 5
    assert all(tiled.validity)
    assert len(tiled.tags) == 5
    # All rows should have the same tag
    assert len(set(tiled.tags.tolist())) == 1
    # All family_row_offsets should be 0
    assert all(tiled.family_row_offsets == 0)


def test_tile_single_row_n1_returns_same() -> None:
    """tile_single_row with n=1 returns the same object."""
    owned = from_shapely_geometries([box(0, 0, 1, 1)])
    result = tile_single_row(owned, 1)
    assert result is owned


def test_tile_single_row_rejects_multi_row() -> None:
    """tile_single_row raises ValueError for multi-row input."""
    owned = from_shapely_geometries([box(0, 0, 1, 1), box(1, 1, 2, 2)])
    with pytest.raises(ValueError, match="1-row array"):
        tile_single_row(owned, 5)
