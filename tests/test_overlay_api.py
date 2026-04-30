from __future__ import annotations

import ast
import importlib
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
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon, box

import vibespatial
import vibespatial.api as geopandas
import vibespatial.api._native_results as native_results_module
from vibespatial import write_geoparquet
from vibespatial.api import GeoDataFrame, GeoSeries, read_file
from vibespatial.api._native_results import (
    LeftConstructiveResult,
    NativeAttributeTable,
    NativeTabularResult,
    PairwiseConstructiveResult,
    to_native_tabular_result,
)
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
                offenders.append(f"{path.relative_to(overlay_dir.parent.parent.parent)}:{node.lineno}")
    assert offenders == []


def test_dissolve_gpu_certification_has_no_raw_cupy_scalar_syncs() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "vibespatial"
        / "overlay"
        / "dissolve.py"
    )
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
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "vibespatial"
        / "overlay"
        / "bypass.py"
    )
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
        Path(__file__).resolve().parents[1]
        / "src"
        / "vibespatial"
        / "api"
        / "tools"
        / "overlay.py"
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
        Path(__file__).resolve().parents[1]
        / "src"
        / "vibespatial"
        / "api"
        / "tools"
        / "overlay.py"
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


def test_overlay_face_assembly_host_bridge_runs_only_after_device_decline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime required for selected-face host bridge")
    cp = pytest.importorskip("cupy")
    assemble_module = importlib.import_module("vibespatial.overlay.assemble")
    faces_module = importlib.import_module("vibespatial.overlay.faces")
    host_fallback_module = importlib.import_module("vibespatial.overlay.host_fallback")

    sentinel = object()
    observed: dict[str, object] = {}

    def _gpu_builder(*_args, **_kwargs):
        return None

    def _host_bridge(selected_face_indices):
        observed["bridge_input"] = selected_face_indices
        return np.asarray([2, 0], dtype=np.int64)

    def _host_builder(half_edge_graph, faces, selected_face_indices):
        observed["host_indices"] = selected_face_indices
        return sentinel

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
    assert hasattr(observed["bridge_input"], "__cuda_array_interface__")
    assert observed["host_indices"].tolist() == [2, 0]


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


def _assert_owned_row_mapping_valid(series: GeoSeries) -> None:
    owned = getattr(series.values, "_owned", None)
    assert owned is not None
    state = owned._ensure_device_state()

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
        event for event in owned.diagnostics
        if event.kind == DiagnosticKind.MATERIALIZATION
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
        lambda *args, **kwargs: pytest.fail("overlay should reuse cached subset-compatible sjoin pairs"),
    )

    result = overlay(left_poly, right, how="intersection")

    assert len(result) == 2


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
        geometry=DeviceGeometryArray._from_sequence(
            [box(1, 1, 4, 3)]
        ),
    )

    geopandas.sjoin(left, right, predicate="intersects")
    monkeypatch.setattr(
        overlay_module,
        "_intersecting_index_pairs",
        lambda *args, **kwargs: pytest.fail("overlay should reuse cached pairs for valid participating rows"),
    )

    result = overlay(left, right, how="intersection")

    assert len(result) == 2


def test_overlay_intersection_single_mask_does_not_rewrite_to_clip_on_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    clip_module = importlib.import_module("vibespatial.api.tools.clip")
    monkeypatch.setattr(
        clip_module,
        "clip",
        lambda *args, **kwargs: pytest.fail(
            "single-mask overlay intersection should stay on the native overlay path"
        ),
    )

    result = overlay(left, right, how="intersection")

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
        lambda *args, **kwargs: pytest.fail("public sindex.query should not run on the bbox fast path"),
    )

    idx1, idx2 = overlay_module._intersecting_index_pairs(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
    )

    assert idx1.tolist() == [0]
    assert idx2.tolist() == [0]


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

    owned = result.geometry.values._owned
    assert owned.residency is Residency.DEVICE
    assert owned.device_state is not None
    assert owned.row_count == len(result)
    assert all(
        not (
            buffer.host_materialized
            and buffer.x.size == 0
            and buffer.geometry_offsets.size > 1
        )
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


def test_overlay_few_right_keep_geom_type_uses_sh_before_rowwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    binary_constructive_module = importlib.import_module(
        "vibespatial.constructive.binary_constructive"
    )

    def fail_rowwise(*_args, **_kwargs):
        raise AssertionError("few-right keep_geom_type overlay should use SH before rowwise")

    monkeypatch.setattr(
        binary_constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        fail_rowwise,
    )

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
        event.surface == "vibespatial.kernels.constructive.polygon_intersection"
        and event.implementation == "polygon_intersection_gpu"
        and event.selected is ExecutionMode.GPU
        for event in dispatch_events
    )


def test_overlay_few_right_sh_rejects_invalid_nonempty_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_owned = from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)])
    right_owned = from_shapely_geometries([box(1.0, 1.0, 3.0, 3.0)])
    polygon_intersection_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_intersection"
    )
    validity_module = importlib.import_module("vibespatial.constructive.validity")
    monkeypatch.setattr(
        polygon_intersection_module,
        "polygon_intersection",
        lambda *_args, **_kwargs: left_owned,
    )
    monkeypatch.setattr(
        validity_module,
        "is_valid_owned",
        lambda *_args, **_kwargs: np.asarray([False]),
    )

    result = overlay_module._few_right_sh_intersection_owned(
        left_owned,
        right_owned,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert result is None


def test_overlay_few_right_sh_classifies_device_rectangles_without_host_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(
        binary_constructive_module,
        "_host_rectangle_polygon_mask",
        lambda *_args, **_kwargs: pytest.fail(
            "device rectangle batches should not use host rectangle classification"
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_host_convex_single_ring_polygon_mask",
        lambda *_args, **_kwargs: pytest.fail(
            "device rectangle batches should prove convexity without host classification"
        ),
    )

    reset_d2h_transfer_count()
    result = overlay_module._few_right_sh_intersection_owned(
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
        event.surface in {"geopandas.sindex.query", "geopandas.overlay.sindex"}
        for event in events
    )


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


def test_overlay_intersection_single_geometry_only_mask_rewrites_to_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    expected = geopandas.clip(
        left,
        right.geometry.iloc[0],
        keep_geom_type=True,
        sort=False,
    )

    def _fail_overlay_intersection(*_args, **_kwargs):
        raise AssertionError("single-mask geometry-only intersection should rewrite to clip")

    monkeypatch.setattr(
        overlay_module,
        "_overlay_intersection",
        _fail_overlay_intersection,
    )
    monkeypatch.setattr(overlay_module, "has_gpu_runtime", lambda: False)

    vibespatial.clear_dispatch_events()
    result = overlay(left, right, how="intersection")
    events = vibespatial.get_dispatch_events(clear=True)

    assert_geodataframe_equal(result, expected)
    assert any(
        event.surface == "geopandas.overlay"
        and event.implementation == "clip_rewrite"
        and "execution_family=clip_rewrite" in event.detail
        and "topology_class=mask_clip" in event.detail
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
        event.surface in {"geopandas.sindex.query", "geopandas.overlay.sindex"}
        for event in events
    )
    assert any(
        event.surface == "geopandas.array.intersection"
        and event.selected is ExecutionMode.GPU
        for event in events
    )
    assert any(
        event.surface == "geopandas.array.difference"
        and event.selected is ExecutionMode.GPU
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
        event.surface == "geopandas.array.difference"
        and event.selected is ExecutionMode.GPU
        for event in events
    )


def test_group_source_rows_from_offsets_expands_group_ids() -> None:
    got = overlay_module._group_source_rows_from_offsets(
        np.asarray([0, 2, 2, 5], dtype=np.int64)
    )
    assert got.tolist() == [0, 0, 2, 2, 2]


def test_group_source_rows_from_device_offsets_stay_device_resident() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
    got = overlay_module._group_source_rows_from_offsets(
        cp.asarray([0, 2, 2, 5], dtype=cp.int64),
        total_count=5,
    )

    assert hasattr(got, "__cuda_array_interface__")
    assert cp.asnumpy(got).tolist() == [0, 0, 2, 2, 2]


def test_grouped_overlay_difference_owned_builds_one_grouped_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = from_shapely_geometries(
        [
            box(0, 0, 10, 10),
            box(20, 0, 30, 10),
        ]
    )
    right = from_shapely_geometries(
        [
            box(1, 1, 2, 2),
            box(21, 1, 22, 2),
            box(23, 1, 24, 2),
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
    monkeypatch.setattr(overlay_gpu_module, "_materialize_overlay_execution_plan", _fake_materialize)

    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        np.asarray([0, 1, 3], dtype=np.int64),
        dispatch_mode=ExecutionMode.AUTO,
    )

    assert result is sentinel
    assert len(build_calls) == 1
    assert build_calls[0]["left_rows"] == 2
    assert build_calls[0]["right_rows"] == 3
    assert build_calls[0]["_row_isolated"] is True
    assert np.array_equal(
        np.asarray(build_calls[0]["_right_geometry_source_rows"]),
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
            box(23, 1, 24, 2),
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

    def _fail_group_offsets_host(*_args, **_kwargs):
        raise AssertionError("successful grouped difference should not export full offsets")

    def _fake_build(left_batch, right_batch, **kwargs):
        observed["source_rows"] = kwargs.get("_right_geometry_source_rows")
        return object()

    def _fake_materialize(plan, **kwargs):
        return sentinel, ExecutionMode.GPU

    monkeypatch.setattr(overlay_module, "_group_offsets_to_host", _fail_group_offsets_host)
    monkeypatch.setattr(overlay_gpu_module, "_build_overlay_execution_plan", _fake_build)
    monkeypatch.setattr(overlay_gpu_module, "_materialize_overlay_execution_plan", _fake_materialize)

    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        cp.asarray([0, 1, 3], dtype=cp.int64),
        dispatch_mode=ExecutionMode.GPU,
    )

    source_rows = observed["source_rows"]
    assert result is sentinel
    assert hasattr(source_rows, "__cuda_array_interface__")
    assert cp.asnumpy(source_rows).tolist() == [0, 1, 1]


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

    def _fail_sequential(*_args, **_kwargs):
        raise AssertionError("aligned single-pair groups should not use sequential fallback")

    def _fake_binary(op, left_arg, right_arg, **kwargs):
        calls.append(
            {
                "op": op,
                "left_is_original": left_arg is left,
                "right_is_original": right_arg is right,
                "dispatch_mode": kwargs.get("dispatch_mode"),
                "prefer_rowwise": kwargs.get("_prefer_rowwise_polygon_difference_overlay"),
            }
        )
        return sentinel

    monkeypatch.setattr(overlay_module, "_group_offsets_to_host", _fail_group_offsets_host)
    monkeypatch.setattr(
        overlay_module,
        "_sequential_grouped_difference_owned",
        _fail_sequential,
    )
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
    )

    assert result is sentinel
    assert calls == [
        {
            "op": "difference",
            "left_is_original": True,
            "right_is_original": True,
            "dispatch_mode": ExecutionMode.GPU,
            "prefer_rowwise": True,
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
    monkeypatch.setattr(
        overlay_module,
        "_sequential_grouped_difference_owned",
        lambda *_args, **_kwargs: pytest.fail(
            "aligned single-pair CPU string dispatch should still use pairwise path"
        ),
    )

    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        np.asarray([0, 1], dtype=np.int64),
        dispatch_mode="cpu",
    )

    assert result is sentinel
    assert calls == [ExecutionMode.CPU]


def test_batched_overlay_difference_single_batch_keeps_grouping_on_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")
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
        ],
        residency=Residency.DEVICE,
    )
    reasons: list[str] = []
    observed: dict[str, object] = {}
    original_bridge = overlay_module._overlay_device_to_host

    def _recording_bridge(value, *, reason: str, dtype=None):
        reasons.append(reason)
        return original_bridge(value, reason=reason, dtype=dtype)

    def _fake_grouped_difference(left_batch, right_batch, group_offsets, *, dispatch_mode):
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

    result, idx1_unique = overlay_module._batched_overlay_difference_owned(
        left,
        right,
        None,
        None,
        cp.asarray([0, 0, 1], dtype=cp.int32),
        cp.asarray([0, 1, 2], dtype=cp.int32),
        True,
        ExecutionMode.GPU,
    )

    assert result.row_count == 2
    assert hasattr(idx1_unique, "__cuda_array_interface__")
    assert cp.asnumpy(idx1_unique).tolist() == [0, 1]
    assert observed["left_rows"] == 2
    assert observed["right_rows"] == 3
    assert hasattr(observed["group_offsets"], "__cuda_array_interface__")
    assert cp.asnumpy(observed["group_offsets"]).tolist() == [0, 2, 3]
    assert "overlay difference unique-left index host export" not in reasons
    assert not any("batch-offset" in reason or "batch right-index" in reason for reason in reasons)


def test_overlay_difference_scatter_indices_stay_device_resident() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    cp = pytest.importorskip("cupy")

    no_neighbor, neighbor = overlay_module._difference_scatter_indices(
        5,
        cp.asarray([0, 3], dtype=cp.int64),
    )

    assert hasattr(no_neighbor, "__cuda_array_interface__")
    assert hasattr(neighbor, "__cuda_array_interface__")
    assert cp.asnumpy(no_neighbor).tolist() == [1, 2, 4]
    assert cp.asnumpy(neighbor).tolist() == [0, 3]


def test_grouped_overlay_difference_owned_falls_back_to_sequential_exact_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = from_shapely_geometries([box(0, 0, 10, 10)])
    right = from_shapely_geometries([box(1, 1, 2, 2), box(3, 3, 4, 4)])
    sentinel = from_shapely_geometries([box(0, 0, 8, 8)])
    fallback_calls: list[dict[str, object]] = []
    segmented_union_module = importlib.import_module(
        "vibespatial.kernels.constructive.segmented_union"
    )

    def _raising_build(*args, **kwargs):
        raise RuntimeError("boom")

    def _raising_grouped_union(*args, **kwargs):
        raise RuntimeError("grouped union unavailable")

    def _fake_sequential(left_batch, right_batch, group_offsets, *, dispatch_mode):
        fallback_calls.append(
            {
                "left_rows": left_batch.row_count,
                "right_rows": right_batch.row_count,
                "group_offsets": np.asarray(group_offsets).tolist(),
                "dispatch_mode": dispatch_mode,
            }
        )
        return sentinel

    monkeypatch.setattr(overlay_gpu_module, "_build_overlay_execution_plan", _raising_build)
    monkeypatch.setattr(segmented_union_module, "segmented_union_all", _raising_grouped_union)
    monkeypatch.setattr(overlay_module, "_sequential_grouped_difference_owned", _fake_sequential)

    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        np.asarray([0, 2], dtype=np.int64),
        dispatch_mode=ExecutionMode.GPU,
    )

    assert result is sentinel
    assert fallback_calls == [
        {
            "left_rows": 1,
            "right_rows": 2,
            "group_offsets": [0, 2],
            "dispatch_mode": ExecutionMode.GPU,
        }
    ]


def test_grouped_overlay_difference_plan_failure_uses_grouped_union_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

    segmented_union_module = importlib.import_module(
        "vibespatial.kernels.constructive.segmented_union"
    )
    left = from_shapely_geometries(
        [box(0, 0, 10, 10), box(20, 0, 30, 10)],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            box(1, 1, 2, 2),
            box(21, 1, 22, 2),
            box(23, 1, 24, 2),
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

    def _fake_grouped_union(right_arg, group_offsets, *, dispatch_mode):
        union_calls.append(
            {
                "right_is_original": right_arg is right,
                "group_offsets": np.asarray(group_offsets).tolist(),
                "dispatch_mode": dispatch_mode,
            }
        )
        return unioned

    def _fake_binary(op, left_arg, right_arg, **kwargs):
        difference_calls.append(
            {
                "op": op,
                "left_is_original": left_arg is left,
                "right_is_unioned": right_arg is unioned,
                "dispatch_mode": kwargs.get("dispatch_mode"),
                "prefer_rowwise": kwargs.get("_prefer_rowwise_polygon_difference_overlay"),
            }
        )
        return sentinel

    monkeypatch.setattr(overlay_gpu_module, "_build_overlay_execution_plan", _raising_build)
    monkeypatch.setattr(segmented_union_module, "segmented_union_all", _fake_grouped_union)
    monkeypatch.setattr(constructive_module, "binary_constructive_owned", _fake_binary)
    monkeypatch.setattr(
        overlay_module,
        "_sequential_grouped_difference_owned",
        lambda *_args, **_kwargs: pytest.fail(
            "grouped union fallback should run before sequential exact fallback"
        ),
    )

    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        np.asarray([0, 1, 3], dtype=np.int64),
        dispatch_mode=ExecutionMode.GPU,
    )

    assert result is sentinel
    assert union_calls == [
        {
            "right_is_original": True,
            "group_offsets": [0, 1, 3],
            "dispatch_mode": ExecutionMode.GPU,
        }
    ]
    assert difference_calls == [
        {
            "op": "difference",
            "left_is_original": True,
            "right_is_unioned": True,
            "dispatch_mode": ExecutionMode.GPU,
            "prefer_rowwise": True,
        }
    ]


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
        geometry=GeoSeries(
            [Polygon([(0, 0), (12, 0), (12, 8), (0, 8), (0, 0)])]
        ),
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
        result = overlay(left, right, how="difference")

    expected = shapely.difference(
        left.geometry.iloc[0],
        shapely.union_all(np.asarray(right.geometry, dtype=object)),
    )
    assert len(result) == 1
    assert grouped_union_calls <= 1
    assert grouped_plan_calls >= 1
    # The grouped plan materializes once; exact repair may route the compacted
    # invalid row through the row-preserving difference planner, which can
    # materialize its own difference plan without changing the public result.
    assert 1 <= grouped_materialize_calls <= 3
    assert difference_calls == 1
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
    assert all(got.normalize().equals_exact(want.normalize(), tolerance=1e-9) for got, want in zip(actual, expected, strict=True))

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
        ("10000", 43),
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
    nearby_buildings = buildings[
        buildings["building_id"].isin(nearby_building_ids)
    ].copy()

    transit_buffers = transit.copy()
    transit_buffers["geometry"] = transit_buffers.geometry.buffer(transit_buffer_m)

    developable = overlay(parcels, exclusions, how="difference")
    developable = developable[
        developable.geometry.geom_type.isin(polygonal_types)
    ].copy()

    served = geopandas.sjoin(
        developable,
        transit_buffers[["station_id", "geometry"]],
        predicate="intersects",
    )
    served_rows = served.index.unique()
    served_parcels = (
        developable.loc[served_rows].copy()
        if len(served_rows) > 0
        else developable.iloc[:0].copy()
    )

    left = served_parcels[["parcel_id", "geometry"]]
    right = nearby_buildings[["building_id", "geometry"]]

    vibespatial.clear_fallback_events()
    actual = overlay(left, right, how="intersection")
    fallback_events = vibespatial.get_fallback_events(clear=True)
    _assert_all_geometry_coordinates_finite(actual.geometry)
    assert not any(
        event.surface == "geopandas.array.make_valid"
        and event.selected is ExecutionMode.CPU
        for event in fallback_events
    )

    left_owned, right_owned = overlay_module._extract_owned_pair(left, right)
    index_result = overlay_module._intersecting_index_pairs(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
    )
    if isinstance(index_result, overlay_module.DeviceSpatialJoinResult):
        idx1, idx2 = index_result.to_host()
    elif isinstance(index_result, np.ndarray) and index_result.ndim == 2:
        idx1, idx2 = index_result
    else:
        idx1, idx2 = index_result
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
    source_scale = np.minimum(
        np.abs(np.asarray(shapely.area(pair_left.geometry.array), dtype=np.float64)),
        np.abs(np.asarray(shapely.area(pair_right.geometry.array), dtype=np.float64)),
    )
    area_tol = source_scale * overlay_module._POLYGON_KEEP_GEOM_TYPE_AREA_RTOL
    polygon_keep_mask = np.array(
        [
            geom is not None
            and not shapely.is_empty(geom)
            and geom.geom_type in polygonal_types
            and float(area) > float(tol)
            for geom, area, tol in zip(exact_values, exact_area, area_tol, strict=True)
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
            shapely.contains(exact_left, exact_points) & shapely.contains(exact_right, exact_points),
            dtype=bool,
        )

    expected = GeoDataFrame(
        {
            "parcel_id": pair_left.loc[keep_mask, "parcel_id"].to_numpy(),
            "building_id": pair_right.loc[keep_mask, "building_id"].to_numpy(),
            "_area_tol": area_tol[keep_mask],
            "geometry": exact_values[keep_mask],
        },
        geometry="geometry",
        crs=left.crs,
    ).sort_values(["parcel_id", "building_id"]).reset_index(drop=True)
    actual = actual.sort_values(["parcel_id", "building_id"]).reset_index(drop=True)

    assert len(actual) == expected_rows
    assert len(actual) == len(expected)
    assert actual[["parcel_id", "building_id"]].equals(expected[["parcel_id", "building_id"]])
    actual_area = np.asarray(shapely.area(actual.geometry.array), dtype=np.float64)
    assert np.all(actual_area > expected["_area_tol"].to_numpy(dtype=np.float64))


def test_binary_constructive_intersection_stays_strict_native_for_multipolygon_polygon_batch() -> None:
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
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
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
    assert result is left


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


def test_overlay_union_keeps_boundary_touching_difference_rows_separate_in_strict_native_mode() -> None:
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
    assert result.geometry.iloc[0].equals(
        Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)])
    )


def test_overlay_symmetric_difference_boundary_touches_preserve_polygon_in_strict_native_mode() -> None:
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
    assert result.geometry.iloc[0].equals(
        Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)])
    )


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


def test_overlay_intersection_many_vs_one_remainder_records_fallback_before_host_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_rem = from_shapely_geometries([box(0, 0, 2, 2)], residency=Residency.HOST)
    right_one = from_shapely_geometries([box(1, 1, 3, 3)], residency=Residency.HOST)

    monkeypatch.setattr(overlay_module, "has_gpu_runtime", lambda: False)
    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: (
            [],
            left_rem,
            np.asarray([0], dtype=np.intp),
            right_one,
            ExecutionMode.AUTO,
        ),
    )

    original_to_shapely = OwnedGeometryArray.to_shapely

    def _wrapped_to_shapely(self, *args, **kwargs):
        events = vibespatial.get_fallback_events()
        assert len(events) == 1
        assert events[0].surface == "geopandas.overlay.intersection"
        assert "many-vs-one remainder" in events[0].reason
        return original_to_shapely(self, *args, **kwargs)

    monkeypatch.setattr(OwnedGeometryArray, "to_shapely", _wrapped_to_shapely)

    vibespatial.clear_fallback_events()
    result = overlay_module._many_vs_one_intersection_owned(left_rem, right_one, 0)

    assert result.row_count == 1
    assert result.to_shapely()[0].equals(box(0, 0, 2, 2).intersection(box(1, 1, 3, 3)))
    assert vibespatial.get_fallback_events(clear=True)[0].reason.startswith(
        "many-vs-one remainder: vectorized Shapely intersection"
    )


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
        event.surface == "geopandas.array.intersection"
        and event.selected is ExecutionMode.GPU
        for event in events
    )
    assert any(
        event.surface == "geopandas.overlay"
        and event.selected is ExecutionMode.GPU
        for event in events
    )
    assert not any(
        event.surface == "geopandas.overlay"
        and event.implementation == "shapely_host"
        for event in events
    )
    assert not any(
        event.surface == "geopandas.array.make_valid"
        and event.selected is ExecutionMode.CPU
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

    clear_materialization_events()
    result = overlay(left, right, how="intersection")
    events = get_materialization_events(clear=True)

    assert len(result) == 3
    assert not any(
        event.operation in {"geodataframe_geom_type", "geodataframe_total_bounds"}
        for event in events
    )


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
            lambda self: pytest.fail(
                "owned validity-cache seeding should not use public geom_type"
            )
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
        event.surface == "geopandas.array.intersection"
        and event.selected is ExecutionMode.GPU
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
        event.surface == "geopandas.overlay"
        and event.selected is ExecutionMode.GPU
        for event in events
    )
    assert not any(
        event.surface == "geopandas.overlay"
        and event.implementation == "shapely_host"
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

    with strict_native_environment():
        with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay"):
            result = overlay(left, right, how="intersection", keep_geom_type=None)

    assert len(result) == 1
    assert result.geometry.geom_type.tolist() == ["Polygon"]
    assert float(result.geometry.iloc[0].area) > 0.0


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

    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        left_pairs,
        right_pairs,
        area_pairs,
        keep_geom_type_warning=False,
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

    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        None,
        None,
        area_pairs,
        keep_geom_type_warning=False,
    )

    assert keep_mask.tolist() == [True, False, False]
    assert dropped == 0
    assert len(filtered) == 1
    assert filtered.iloc[0].equals(box(0, 0, 1, 1))


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

    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        None,
        None,
        area_pairs,
        keep_geom_type_warning=False,
        left_source=left_source,
        right_source=right_source,
        left_rows=np.asarray([0, 1], dtype=np.intp),
        right_rows=np.asarray([0, 1], dtype=np.intp),
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

    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        left_pairs,
        right_pairs,
        area_pairs,
        keep_geom_type_warning=True,
    )

    assert keep_mask.tolist() == [True, False]
    assert dropped == 2
    assert len(filtered) == 1
    assert filtered.iloc[0].equals(box(0, 0, 1, 1))


def test_overlay_intersection_small_exact_boundary_reuses_existing_owned_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        lambda *args, **kwargs: pytest.fail("all-surviving keep-geom-type filter should not take every row"),
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
    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        None,
        None,
        area_pairs,
        keep_geom_type_warning=True,
        left_source=left_source,
        right_source=right_source,
        left_rows=np.arange(2, dtype=np.intp),
        right_rows=np.arange(2, dtype=np.intp),
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 0
    assert len(filtered) == 2
    assert vibespatial.get_fallback_events(clear=True) == []


def test_keep_geom_type_filter_rect_overlap_applies_native_area_tolerance() -> None:
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
    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        None,
        None,
        area_pairs,
        keep_geom_type_warning=True,
        left_source=left_source,
        right_source=right_source,
        left_rows=np.asarray([0], dtype=np.intp),
        right_rows=np.asarray([0], dtype=np.intp),
    )

    assert keep_mask.tolist() == [False]
    assert dropped == 1
    assert len(filtered) == 0
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
    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        None,
        None,
        area_pairs,
        keep_geom_type_warning=True,
        left_source=left_source,
        right_source=right_source,
        left_rows=np.arange(2, dtype=np.intp),
        right_rows=np.arange(2, dtype=np.intp),
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
    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        None,
        None,
        area_pairs,
        keep_geom_type_warning=True,
        left_source=left_source,
        right_source=right_source,
        left_rows=np.arange(2, dtype=np.intp),
        right_rows=np.arange(2, dtype=np.intp),
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
        GeometryArray.from_owned(
            from_shapely_geometries(left_geoms, residency=Residency.HOST)
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(right_geoms, residency=Residency.HOST)
        )
    )
    left_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
        )
    )
    right_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
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
    area_pairs = GeoSeries(GeometryArray.from_owned(area_owned))

    monkeypatch.setattr(
        overlay_module,
        "_take_geoseries_object_values",
        lambda *_args, **_kwargs: pytest.fail(
            "host-backed sources should not materialize when device-backed pair rows are available"
        ),
    )

    vibespatial.clear_fallback_events()
    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        left_pairs,
        right_pairs,
        area_pairs,
        keep_geom_type_warning=True,
        left_source=left_source,
        right_source=right_source,
        left_rows=np.arange(2, dtype=np.intp),
        right_rows=np.arange(2, dtype=np.intp),
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 0
    assert len(filtered) == 2
    assert vibespatial.get_fallback_events(clear=True) == []


def test_keep_geom_type_filter_kept_rows_with_shared_boundary_on_area_boundary_stays_native() -> None:
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


def test_device_count_dropped_polygon_intersection_warning_rows_handles_large_distinct_pair_batches() -> None:
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
        GeometryArray.from_owned(
            from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
        )
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


def test_device_count_dropped_polygon_intersection_warning_rows_skips_large_warning_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    constructive_module = importlib.import_module("vibespatial.constructive.binary_constructive")

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

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        lambda *_args, **_kwargs: pytest.fail(
            "large advisory warning batches should bypass exact boundary constructive refinement"
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

    assert dropped is None


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


def test_device_polygon_keep_geom_type_cover_mask_uses_broadcast_right(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    predicate_module = importlib.import_module("vibespatial.predicates.binary")
    observed_calls: list[tuple[str, int, int]] = []

    def _fake_evaluate_binary_predicate(predicate, left, right, **kwargs):
        observed_calls.append((predicate, left.row_count, right.row_count))
        assert kwargs["dispatch_mode"] is ExecutionMode.GPU
        return SimpleNamespace(values=np.ones(left.row_count, dtype=bool))

    monkeypatch.setattr(
        predicate_module,
        "evaluate_binary_predicate",
        _fake_evaluate_binary_predicate,
    )

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

    cover_mask = overlay_module._device_polygon_keep_geom_type_cover_mask(
        left_source,
        right_source,
        np.arange(3, dtype=np.intp),
        np.zeros(3, dtype=np.intp),
        np.arange(3, dtype=np.intp),
    )

    assert np.asarray(cover_mask, dtype=bool).tolist() == [True, True, True]
    assert observed_calls == [
        ("covered_by", 3, 1),
    ]


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

    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        None,
        None,
        area_pairs,
        keep_geom_type_warning=True,
        left_source=left_source,
        right_source=right_source,
        left_rows=np.asarray([0], dtype=np.intp),
        right_rows=np.asarray([0], dtype=np.intp),
    )

    assert keep_mask.tolist() == [True]
    assert dropped == 0
    assert len(filtered) == 1


def test_assemble_indexed_owned_chunks_preserves_exact_intersection_cache() -> None:
    first = from_shapely_geometries(
        [box(2, 0, 3, 1)],
        residency=Residency.HOST,
    )
    first._exact_intersection_values = np.asarray(
        [GeometryCollection([box(2, 0, 3, 1), LineString([(2, 0), (3, 0)])])],
        dtype=object,
    )
    first._exact_intersection_value_mask = np.asarray([True], dtype=bool)

    second = from_shapely_geometries(
        [box(0, 0, 1, 1)],
        residency=Residency.HOST,
    )

    assembled = overlay_module._assemble_indexed_owned_chunks(
        [
            (np.asarray([1], dtype=np.intp), first),
            (np.asarray([0], dtype=np.intp), second),
        ],
        2,
    )

    assert np.asarray(
        getattr(assembled, "_exact_intersection_value_mask", None),
        dtype=bool,
    ).tolist() == [False, True]
    exact_values = getattr(assembled, "_exact_intersection_values", None)
    assert exact_values is not None
    assert exact_values[1].geom_type == "GeometryCollection"


def test_assemble_indexed_owned_chunks_accepts_device_indices_without_host_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")

    first = from_shapely_geometries([box(2, 0, 3, 1)], residency=Residency.DEVICE)
    second = from_shapely_geometries([box(0, 0, 1, 1)], residency=Residency.DEVICE)
    first._polygon_rect_exact_polygon_only = cp.asarray([True], dtype=cp.bool_)
    second._polygon_rect_exact_polygon_only = cp.asarray([True], dtype=cp.bool_)

    original_bridge = overlay_module._overlay_device_to_host

    def _fail_index_exports(value, *, reason: str, dtype=None):
        if "indexed" in reason:
            pytest.fail(f"device indexed assembly should not export positions: {reason}")
        return original_bridge(value, reason=reason, dtype=dtype)

    monkeypatch.setattr(overlay_module, "_overlay_device_to_host", _fail_index_exports)

    assembled = overlay_module._assemble_indexed_owned_chunks(
        [
            (cp.asarray([1], dtype=cp.int64), first),
            (cp.asarray([0], dtype=cp.int64), second),
        ],
        2,
    )

    assert assembled.residency is Residency.DEVICE
    assert assembled.row_count == 2
    assert cp.asnumpy(assembled._polygon_rect_exact_polygon_only).tolist() == [True, True]
    got = assembled.to_shapely()
    assert got[0].equals(box(0, 0, 1, 1))
    assert got[1].equals(box(2, 0, 3, 1))


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


def test_repair_invalid_polygon_output_rows_small_batch_falls_back_when_overlap_mask_is_empty() -> None:
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


def test_overlay_make_valid_owned_rewrap_failure_records_fallback_before_host_materialization(
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
        events = vibespatial.get_fallback_events()
        assert len(events) == 1
        assert events[0].surface == "geopandas.array.make_valid"
        assert "host materialization required" in events[0].reason
        return original_to_shapely(*args, **kwargs)

    monkeypatch.setattr(fallback_owned, "to_shapely", _wrapped_to_shapely)

    vibespatial.clear_fallback_events()
    with pytest.raises(StrictNativeFallbackError):
        with strict_native_environment():
            overlay(left, right, how="intersection")

    assert from_owned_calls == 1
    assert materialized is False
    events = vibespatial.get_fallback_events(clear=True)
    assert len(events) == 1
    assert events[0].surface == "geopandas.array.make_valid"
    assert events[0].selected is ExecutionMode.CPU


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

    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        left_pairs,
        right_pairs,
        area_pairs,
        keep_geom_type_warning=True,
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
    assert np.asarray(observed["left_rows"]).dtype == np.intp
    assert np.asarray(observed["right_rows"]).dtype == np.intp


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
        event.surface == "geopandas.overlay"
        and event.implementation == "shapely_host"
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
        event.surface == "vibespatial.predicates.binary"
        and event.operation == "covered_by"
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
        event.surface == "geopandas.overlay"
        and event.implementation == "shapely_host"
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


def test_overlay_intersection_many_vs_one_small_remainder_prefers_broadcast_right_exact_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_owned = from_shapely_geometries(
        [
            Polygon([(-1, 1), (2, 1), (2, 4), (-1, 4), (-1, 1)]),
            Polygon([(1, -1), (4, -1), (4, 2), (1, 2), (1, -1)]),
            Polygon([(3, 3), (6, 3), (6, 6), (3, 6), (3, 3)]),
        ],
        residency=Residency.DEVICE,
    )
    right_owned = from_shapely_geometries(
        [Polygon([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6), (0, 0)])],
        residency=Residency.DEVICE,
    )

    binary_module = importlib.import_module("vibespatial.constructive.binary_constructive")
    overlay_gpu_module = importlib.import_module("vibespatial.overlay.gpu")
    broadcast_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(
        overlay_gpu_module,
        "_overlay_owned",
        lambda *args, **kwargs: pytest.fail(
            "small many-vs-one exact remainders should try the broadcast-right helper "
            "before the batched row-isolated overlay graph"
        ),
    )
    monkeypatch.setattr(
        binary_module,
        "_dispatch_polygon_intersection_overlay_broadcast_right_gpu",
        lambda left_arg, right_arg, **kwargs: (
            broadcast_calls.append((left_arg.row_count, right_arg.row_count)),
            left_arg,
        )[1],
    )
    monkeypatch.setattr(
        binary_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda *args, **kwargs: pytest.fail(
            "many-vs-one exact remainder should not materialize a tiled right operand "
            "and fall back to the legacy rowwise helper"
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "record_dispatch_event",
        lambda *args, **kwargs: None,
    )

    result = overlay_module._many_vs_one_intersection_owned(
        left_owned,
        right_owned,
        0,
    )

    assert broadcast_calls == [(3, 1)]
    assert result.row_count == 3


def test_overlay_intersection_many_vs_one_large_remainder_uses_cached_right_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    complex_left = from_shapely_geometries(
        [box(float(i) - 1.0, 1.0, float(i) + 2.0, 4.0) for i in range(24)],
        residency=Residency.DEVICE,
    )
    right_one = from_shapely_geometries(
        [Polygon([(0, 0), (32, 0), (32, 2), (8, 2), (8, 6), (0, 6), (0, 0)])],
        residency=Residency.DEVICE,
    )

    overlay_gpu_module = importlib.import_module("vibespatial.overlay.gpu")
    overlay_calls: list[tuple[int, int, bool]] = []
    real_overlay_owned = overlay_gpu_module._overlay_owned

    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: (
            [],
            complex_left,
            np.arange(complex_left.row_count, dtype=np.intp),
            right_one,
            ExecutionMode.GPU,
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_OVERLAY_ROWWISE_REMAINDER_MAX",
        16,
    )
    monkeypatch.setattr(
        overlay_module,
        "_OVERLAY_MEDIUM_REMAINDER_ROWWISE_MAX",
        16,
    )

    def _wrapped_overlay(left_arg, right_arg, **kwargs):
        overlay_calls.append(
            (
                left_arg.row_count,
                right_arg.row_count,
                kwargs.get("_cached_right_segments") is not None,
            )
        )
        return real_overlay_owned(left_arg, right_arg, **kwargs)

    monkeypatch.setattr(
        overlay_gpu_module,
        "_overlay_owned",
        _wrapped_overlay,
    )

    result = overlay_module._many_vs_one_intersection_owned(
        complex_left,
        right_one,
        0,
    )

    assert overlay_calls == [(24, 24, True)]
    assert result.row_count == complex_left.row_count


def test_overlay_intersection_many_vs_one_large_host_remainder_promotes_to_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    overlay_gpu_module = importlib.import_module("vibespatial.overlay.gpu")

    complex_left = from_shapely_geometries(
        [box(float(i) - 1.0, 1.0, float(i) + 2.0, 4.0) for i in range(24)],
        residency=Residency.HOST,
    )
    right_one = from_shapely_geometries(
        [Polygon([(0, 0), (32, 0), (32, 2), (8, 2), (8, 6), (0, 6), (0, 0)])],
        residency=Residency.HOST,
    )

    overlay_calls: list[tuple[str, str, int, int]] = []
    real_overlay_owned = overlay_gpu_module._overlay_owned

    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: (
            [],
            complex_left,
            np.arange(complex_left.row_count, dtype=np.intp),
            right_one,
            ExecutionMode.GPU,
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_OVERLAY_ROWWISE_REMAINDER_MAX",
        16,
    )
    monkeypatch.setattr(
        overlay_module,
        "_OVERLAY_MEDIUM_REMAINDER_ROWWISE_MAX",
        16,
    )
    monkeypatch.setattr(
        overlay_module,
        "plan_dispatch_selection",
        lambda *args, **kwargs: SimpleNamespace(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_host_exact_polygon_intersection_owned_batch",
        lambda *args, **kwargs: pytest.fail(
            "GPU runtime should not short-circuit host-backed many-vs-one batches to host exact intersection"
        ),
    )

    def _wrapped_overlay(*args, **kwargs):
        overlay_calls.append(
            (
                str(args[0].residency),
                str(args[1].residency),
                args[0].row_count,
                args[1].row_count,
            )
        )
        return real_overlay_owned(*args, **kwargs)

    monkeypatch.setattr(
        overlay_gpu_module,
        "_overlay_owned",
        _wrapped_overlay,
    )
    monkeypatch.setattr(
        overlay_module,
        "record_fallback_event",
        lambda *args, **kwargs: pytest.fail(
            "large many-vs-one host remainder should promote to device before CPU fallback"
        )
        if "many-vs-one remainder" in str(kwargs.get("reason", ""))
        else None,
    )

    result = overlay_module._many_vs_one_intersection_owned(
        complex_left,
        right_one,
        0,
    )

    assert overlay_calls == [("device", "device", 24, 24)]
    assert result.residency is Residency.DEVICE
    assert result.row_count == complex_left.row_count


def test_overlay_intersection_many_vs_one_large_remainder_chunks_exact_gpu_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    overlay_gpu_module = importlib.import_module("vibespatial.overlay.gpu")
    complex_left = from_shapely_geometries(
        [box(float(i) - 1.0, 1.0, float(i) + 2.0, 4.0) for i in range(10)],
        residency=Residency.DEVICE,
    )
    right_one = from_shapely_geometries(
        [box(0.0, 0.0, 16.0, 6.0)],
        residency=Residency.DEVICE,
    )
    chunk_sizes: list[int] = []

    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: (
            [],
            complex_left,
            np.arange(complex_left.row_count, dtype=np.intp),
            right_one,
            ExecutionMode.GPU,
        ),
    )
    monkeypatch.setattr(overlay_module, "_OVERLAY_ROWWISE_REMAINDER_MAX", 0)
    monkeypatch.setattr(overlay_module, "_OVERLAY_MEDIUM_REMAINDER_ROWWISE_MAX", 0)
    monkeypatch.setattr(
        overlay_module,
        "_OVERLAY_BROADCAST_RIGHT_EXACT_TARGET_SEGMENT_BYTES",
        200,
    )
    monkeypatch.setattr(
        overlay_module,
        "record_fallback_event",
        lambda *args, **kwargs: pytest.fail(
            "chunked many-vs-one GPU remainder should not fall back to host exact intersection"
        )
        if "many-vs-one remainder" in str(kwargs.get("reason", ""))
        else None,
    )

    real_overlay_owned = overlay_gpu_module._overlay_owned

    def _wrapped_overlay(left_arg, right_arg, **kwargs):
        chunk_sizes.append(left_arg.row_count)
        return real_overlay_owned(left_arg, right_arg, **kwargs)

    monkeypatch.setattr(
        overlay_gpu_module,
        "_overlay_owned",
        _wrapped_overlay,
    )

    result = overlay_module._many_vs_one_intersection_owned(
        complex_left,
        right_one,
        0,
    )

    assert len(chunk_sizes) > 1
    assert sum(chunk_sizes) == complex_left.row_count
    assert all(size < complex_left.row_count for size in chunk_sizes)
    assert result.row_count == complex_left.row_count


def test_overlay_intersection_many_vs_one_large_remainder_retries_smaller_gpu_batches_on_memory_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    binary_constructive_module = importlib.import_module(
        "vibespatial.constructive.binary_constructive"
    )
    overlay_gpu_module = importlib.import_module("vibespatial.overlay.gpu")
    complex_left = from_shapely_geometries(
        [box(float(i) - 1.0, 1.0, float(i) + 2.0, 4.0) for i in range(10)],
        residency=Residency.DEVICE,
    )
    right_one = from_shapely_geometries(
        [box(0.0, 0.0, 16.0, 6.0)],
        residency=Residency.DEVICE,
    )
    chunk_sizes: list[int] = []
    raised_large_chunk = {"value": False}

    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: (
            [],
            complex_left,
            np.arange(complex_left.row_count, dtype=np.intp),
            right_one,
            ExecutionMode.GPU,
        ),
    )
    monkeypatch.setattr(overlay_module, "_OVERLAY_ROWWISE_REMAINDER_MAX", 0)
    monkeypatch.setattr(overlay_module, "_OVERLAY_MEDIUM_REMAINDER_ROWWISE_MAX", 0)
    monkeypatch.setattr(
        overlay_module,
        "_OVERLAY_BROADCAST_RIGHT_EXACT_TARGET_SEGMENT_BYTES",
        1,
    )
    monkeypatch.setattr(
        overlay_module,
        "_OVERLAY_BROADCAST_RIGHT_EXACT_MIN_CHUNK_ROWS",
        8,
    )
    monkeypatch.setattr(
        binary_constructive_module,
        "_dispatch_polygon_intersection_overlay_broadcast_right_gpu",
        lambda *args, **kwargs: pytest.fail(
            "memory-pressure retry should stay on the tiled exact GPU path"
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "record_fallback_event",
        lambda *args, **kwargs: pytest.fail(
            "many-vs-one GPU retry should not fall back to host exact intersection"
        )
        if "many-vs-one remainder" in str(kwargs.get("reason", ""))
        else None,
    )

    real_overlay_owned = overlay_gpu_module._overlay_owned

    def _wrapped_overlay(left_arg, right_arg, **kwargs):
        chunk_sizes.append(left_arg.row_count)
        if left_arg.row_count == 8 and not raised_large_chunk["value"]:
            raised_large_chunk["value"] = True
            raise RuntimeError("max feasible pairs")
        return real_overlay_owned(left_arg, right_arg, **kwargs)

    monkeypatch.setattr(
        overlay_gpu_module,
        "_overlay_owned",
        _wrapped_overlay,
    )

    result = overlay_module._many_vs_one_intersection_owned(
        complex_left,
        right_one,
        0,
    )

    assert raised_large_chunk["value"] is True
    assert chunk_sizes[0] == 8
    assert chunk_sizes[1:] == [4, 6]
    assert result.row_count == complex_left.row_count


def test_prepare_many_vs_one_intersection_chunks_low_yield_uses_full_batch_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    bypass_module = importlib.import_module("vibespatial.overlay.bypass")

    left_owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
            box(4.0, 0.0, 5.0, 1.0),
        ],
        residency=Residency.DEVICE,
    )
    right_owned = from_shapely_geometries(
        [box(-1.0, -1.0, 10.0, 10.0)],
        residency=Residency.DEVICE,
    )
    contained = left_owned.take(np.asarray([0], dtype=np.int64))

    monkeypatch.setattr(
        overlay_module,
        "_MANY_VS_ONE_DIRECT_FULL_BATCH_MIN_ROWS",
        1,
    )
    monkeypatch.setattr(
        overlay_module,
        "_MANY_VS_ONE_DIRECT_FULL_BATCH_MAX_CONTAINED_FRACTION",
        0.5,
    )
    monkeypatch.setattr(
        bypass_module,
        "_containment_bypass_gpu",
        lambda *args, **kwargs: (
            contained,
            overlay_module.cp.asarray(
                [False, True, True],
                dtype=overlay_module.cp.bool_,
            ),
        ),
    )
    monkeypatch.setattr(
        bypass_module,
        "_is_clip_polygon_sh_eligible",
        lambda *args, **kwargs: (False, 0),
    )

    prepared = overlay_module._prepare_many_vs_one_intersection_chunks(
        left_owned,
        right_owned,
        0,
        global_positions=np.arange(left_owned.row_count, dtype=np.intp),
    )

    index_oga_pairs, complex_left, complex_positions, right_one, _pairwise_mode, use_full_batch = (
        prepared
    )
    assert use_full_batch is True
    assert index_oga_pairs == []
    assert complex_left is left_owned
    assert np.array_equal(
        complex_positions,
        np.arange(left_owned.row_count, dtype=np.intp),
    )
    assert right_one.row_count == 1


def test_prepare_many_vs_one_intersection_chunks_keeps_device_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")
    cp = pytest.importorskip("cupy")

    bypass_module = importlib.import_module("vibespatial.overlay.bypass")
    left_owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
            box(4.0, 0.0, 5.0, 1.0),
        ],
        residency=Residency.DEVICE,
    )
    right_owned = from_shapely_geometries(
        [box(-1.0, -1.0, 10.0, 10.0)],
        residency=Residency.DEVICE,
    )
    contained = left_owned.device_take(cp.asarray([0], dtype=cp.int64))

    monkeypatch.setattr(
        bypass_module,
        "_containment_bypass_gpu",
        lambda *args, **kwargs: (
            contained,
            cp.asarray([False, True, True], dtype=cp.bool_),
        ),
    )
    monkeypatch.setattr(
        bypass_module,
        "_is_clip_polygon_sh_eligible",
        lambda *args, **kwargs: (False, 0),
    )
    original_bridge = overlay_module._overlay_device_to_host

    def _fail_many_vs_one_position_exports(value, *, reason: str, dtype=None):
        if "many-vs-one" in reason and "index host export" in reason:
            pytest.fail(f"many-vs-one chunk prep exported device positions: {reason}")
        return original_bridge(value, reason=reason, dtype=dtype)

    monkeypatch.setattr(
        overlay_module,
        "_overlay_device_to_host",
        _fail_many_vs_one_position_exports,
    )

    prepared = overlay_module._prepare_many_vs_one_intersection_chunks(
        left_owned,
        right_owned,
        0,
        global_positions=cp.arange(left_owned.row_count, dtype=cp.int64),
    )

    index_oga_pairs, complex_left, complex_positions, right_one, _pairwise_mode, use_full_batch = (
        prepared
    )
    assert use_full_batch is False
    assert len(index_oga_pairs) == 1
    contained_positions, contained_owned = index_oga_pairs[0]
    assert hasattr(contained_positions, "__cuda_array_interface__")
    assert hasattr(complex_positions, "__cuda_array_interface__")
    assert cp.asnumpy(contained_positions).tolist() == [0]
    assert cp.asnumpy(complex_positions).tolist() == [1, 2]
    assert contained_owned.row_count == 1
    assert complex_left is not None
    assert complex_left.row_count == 2
    assert right_one.row_count == 1


def test_prepare_many_vs_one_intersection_chunks_large_low_yield_stays_chunked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    bypass_module = importlib.import_module("vibespatial.overlay.bypass")

    left_owned = from_shapely_geometries(
        [box(float(i), 0.0, float(i) + 1.0, 1.0) for i in range(64)],
        residency=Residency.DEVICE,
    )
    right_owned = from_shapely_geometries(
        [box(-1.0, -1.0, 128.0, 8.0)],
        residency=Residency.DEVICE,
    )
    contained = left_owned.take(np.asarray([0], dtype=np.int64))

    monkeypatch.setattr(
        overlay_module,
        "_MANY_VS_ONE_DIRECT_FULL_BATCH_MIN_ROWS",
        1,
    )
    monkeypatch.setattr(
        overlay_module,
        "_MANY_VS_ONE_DIRECT_FULL_BATCH_MAX_CONTAINED_FRACTION",
        0.5,
    )
    monkeypatch.setattr(
        overlay_module,
        "_OVERLAY_ROWWISE_REMAINDER_MAX",
        32,
    )
    monkeypatch.setattr(
        bypass_module,
        "_containment_bypass_gpu",
        lambda *args, **kwargs: (
            contained,
            overlay_module.cp.asarray(
                [False] + [True] * 63,
                dtype=overlay_module.cp.bool_,
            ),
        ),
    )
    monkeypatch.setattr(
        bypass_module,
        "_is_clip_polygon_sh_eligible",
        lambda *args, **kwargs: (False, 0),
    )

    prepared = overlay_module._prepare_many_vs_one_intersection_chunks(
        left_owned,
        right_owned,
        0,
        global_positions=np.arange(left_owned.row_count, dtype=np.intp),
    )

    index_oga_pairs, complex_left, complex_positions, right_one, _pairwise_mode, use_full_batch = (
        prepared
    )
    assert use_full_batch is False
    assert complex_left is not None
    assert complex_left.row_count == left_owned.row_count
    assert np.array_equal(complex_positions, np.arange(left_owned.row_count, dtype=np.intp))
    assert right_one.row_count == 1


def test_overlay_intersection_many_vs_one_low_yield_prefers_full_batch_exact_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    complex_left = from_shapely_geometries(
        [box(float(i), 0.0, float(i) + 1.0, 1.0) for i in range(24)],
        residency=Residency.DEVICE,
    )
    right_one = from_shapely_geometries(
        [box(-1.0, -1.0, 32.0, 4.0)],
        residency=Residency.DEVICE,
    )

    binary_module = importlib.import_module("vibespatial.constructive.binary_constructive")
    broadcast_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: (
            [],
            complex_left,
            np.arange(complex_left.row_count, dtype=np.intp),
            right_one,
            ExecutionMode.GPU,
            True,
        ),
    )
    monkeypatch.setattr(
        binary_module,
        "_dispatch_polygon_intersection_overlay_broadcast_right_gpu",
        lambda left_arg, right_arg, **kwargs: (
            broadcast_calls.append((left_arg.row_count, right_arg.row_count)),
            left_arg,
        )[1],
    )
    monkeypatch.setattr(
        binary_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda *args, **kwargs: pytest.fail(
            "low-yield full-batch exact overlay should stay on the scalar-right "
            "broadcast helper before the legacy rowwise path"
        ),
    )

    result = overlay_module._many_vs_one_intersection_owned(
        complex_left,
        right_one,
        0,
    )

    assert broadcast_calls == [(24, 1)]
    assert result.row_count == complex_left.row_count


def test_overlay_intersection_many_vs_one_marks_only_complex_rows_as_keep_geom_type_suspects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_sub = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
            box(4.0, 0.0, 5.0, 1.0),
        ],
        residency=Residency.HOST,
    )
    right_one = from_shapely_geometries(
        [box(0.0, 0.0, 10.0, 10.0)],
        residency=Residency.HOST,
    )
    contained = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(4.0, 0.0, 5.0, 1.0),
        ],
        residency=Residency.HOST,
    )
    complex_left = from_shapely_geometries(
        [box(2.0, 0.0, 3.0, 1.0)],
        residency=Residency.HOST,
    )

    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: (
            [(np.asarray([0, 2], dtype=np.intp), contained)],
            complex_left,
            np.asarray([1], dtype=np.intp),
            right_one,
            ExecutionMode.CPU,
        ),
    )
    monkeypatch.setattr(overlay_module, "has_gpu_runtime", lambda: False)

    result = overlay_module._many_vs_one_intersection_owned(
        left_sub,
        right_one,
        0,
    )

    assert np.asarray(
        getattr(result, "_polygon_rect_boundary_overlap", None),
        dtype=bool,
    ).tolist() == [False, True, False]


def test_overlay_intersection_many_vs_one_cpu_selected_exact_host_batch_skips_prepare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_sub = from_shapely_geometries(
        [box(float(i), 0.0, float(i) + 2.0, 2.0) for i in range(64)],
        residency=Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST,
    )
    right_one = from_shapely_geometries(
        [box(0.5, 0.5, 63.5, 2.5)],
        residency=Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST,
    )

    monkeypatch.setattr(
        overlay_module,
        "plan_dispatch_selection",
        lambda *args, **kwargs: SimpleNamespace(
            selected=ExecutionMode.CPU,
            requested=ExecutionMode.AUTO,
            reason="synthetic cpu selection",
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: pytest.fail(
            "CPU-selected many-vs-one exact host batch should bypass containment prep"
        ),
    )

    result = overlay_module._many_vs_one_intersection_owned(
        left_sub,
        right_one,
        0,
    )

    exact_mask = np.asarray(getattr(result, "_exact_intersection_value_mask", None), dtype=bool)
    overlap_mask = np.asarray(getattr(result, "_polygon_rect_boundary_overlap", None), dtype=bool)

    assert result.row_count == left_sub.row_count
    assert int(np.count_nonzero(exact_mask)) == left_sub.row_count
    assert not bool(overlap_mask.any())


def test_many_vs_one_exact_batch_plans_broadcast_right_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_prepare_many_vs_one_chunks_plans_broadcast_right_workload(
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
        return SimpleNamespace(selected=ExecutionMode.GPU)

    monkeypatch.setattr(
        overlay_module,
        "plan_dispatch_selection",
        _fake_plan_dispatch_selection,
    )

    prepared = overlay_module._prepare_many_vs_one_intersection_chunks(
        left_sub,
        right_one,
        0,
        global_positions=np.arange(left_sub.row_count, dtype=np.intp),
    )

    assert prepared is not None
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


def test_overlay_intersection_many_vs_one_fast_path_retries_after_first_failure(
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

    original = overlay_module._many_vs_one_intersection_owned
    calls = 0

    def _wrapped_many_vs_one(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise MemoryError("synthetic many-vs-one fast-path failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        overlay_module,
        "_many_vs_one_intersection_owned",
        _wrapped_many_vs_one,
    )

    with strict_native_environment():
        result = overlay(left, right, how="intersection", keep_geom_type=True)

    assert calls == 2
    assert list(result["col1"]) == [1, 2]


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

    exact_calls: list[int] = []

    def _fake_rowwise_exact(
        left_arg,
        right_arg,
        *,
        dispatch_mode=ExecutionMode.GPU,
        _cached_right_segments=None,
    ):
        exact_calls.append(left_arg.row_count)
        geoms = [
            box(float(5000 + i), 0.0, float(5000 + i + 0.5), 0.5)
            for i in range(left_arg.row_count)
        ]
        return from_shapely_geometries(geoms, residency=Residency.DEVICE)

    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: pytest.fail(
            "few-right intersection should not decompose into many-vs-one preparations"
        ),
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        _fake_rowwise_exact,
    )
    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        lambda *args, **kwargs: pytest.fail(
            "few-right public intersection should fuse exact leftovers "
            "through the rowwise helper"
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
    expected = [
        box(float(5000 + i), 0.0, float(5000 + i + 0.5), 0.5)
        for i in range(24)
    ]
    for got_geom, expected_geom in zip(result.geometry, expected, strict=True):
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

    assert isinstance(native_result, NativeTabularResult)
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

    assert isinstance(export_result, NativeTabularResult)
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

    assert isinstance(native_result, NativeTabularResult)
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

    assert isinstance(native_result, NativeTabularResult)
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
        raise AssertionError(
            "native union GeoParquet write should not require GeoDataFrame export"
        )

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

    assert isinstance(tabular, NativeTabularResult)
    assert_geodataframe_equal(
        tabular.to_geodataframe(),
        overlay(left, right, how="union"),
        normalize=True,
        check_column_type=False,
    )


def test_overlay_intersection_export_native_tabular_skips_pandas_attribute_assembler(
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
    assert isinstance(tabular, NativeTabularResult)
    assert isinstance(tabular.attributes, NativeAttributeTable)
    assert tabular.attributes.arrow_table is not None
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
    assert isinstance(tabular, NativeTabularResult)
    assert isinstance(tabular.attributes, NativeAttributeTable)
    assert tabular.attributes.arrow_table is not None
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

    assert isinstance(tabular, NativeTabularResult)

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

    assert isinstance(tabular, NativeTabularResult)

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


def test_overlay_difference_boundary_guard_uses_owned_family_metadata_without_geom_type_export() -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            GeometryArray.from_owned(
                from_shapely_geometries([box(0, 0, 2, 2), None])
            )
        ),
    )
    right = GeoDataFrame(
        {"col2": [10]},
        geometry=GeoSeries(
            GeometryArray.from_owned(from_shapely_geometries([box(1, 1, 4, 4)]))
        ),
    )

    clear_materialization_events()
    assert overlay_module._needs_host_overlay_difference_boundary_rebuild(left, right) is False
    events = get_materialization_events(clear=True)
    assert not any(
        event.operation in {"geoseries_geom_type", "geodataframe_geom_type"}
        for event in events
    )

    line_left = GeoDataFrame(
        {"col1": [1]},
        geometry=GeoSeries(
            GeometryArray.from_owned(
                from_shapely_geometries([LineString([(0, 0), (1, 1)])])
            )
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
    assert not any(
        event.operation == "device_geometryarray_to_numpy"
        for event in events
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


def test_overlay_extract_owned_pair_leaves_large_intersection_on_index_route() -> None:
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

    assert left_owned is None
    assert right_owned is None


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
        return left_arg

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

    rect_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda *args, **kwargs: pytest.fail(
            "rectangle-capable few-right intersection should bypass rowwise overlay"
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_few_right_sh_intersection_owned",
        lambda *args, **kwargs: None,
    )

    def _fake_polygon_rect_intersection(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU):
        rect_calls.append((left_arg.row_count, right_arg.row_count))
        geoms = [
            box(float(7000 + i), 0.0, float(7000 + i + 0.5), 0.5)
            for i in range(left_arg.row_count)
        ]
        return from_shapely_geometries(geoms, residency=Residency.DEVICE)

    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection",
        _fake_polygon_rect_intersection,
    )
    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        lambda *args, **kwargs: pytest.fail(
            "rectangle-capable few-right intersection should use direct "
            "polygon_rect_intersection before generic constructive dispatch"
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
    assert rect_calls == [(24, 24)]
    assert result["col1"].tolist() == idx1.tolist()
    assert result["zone_type"].tolist() == ["A"] * 8 + ["B"] * 8 + ["C"] * 8


def test_few_right_rect_clip_accepts_right_rectangle_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

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
    can_handle_calls: list[tuple[bool, bool]] = []
    rect_calls: list[tuple[bool, bool]] = []

    def _fake_can_handle(left_arg, right_arg):
        can_handle_calls.append((left_arg is left_pairs, right_arg is right_pairs))
        return left_arg is left_pairs and right_arg is right_pairs

    def _fake_polygon_rect_intersection(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU):
        rect_calls.append((left_arg is left_pairs, right_arg is right_pairs))
        return sentinel

    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection_can_handle",
        _fake_can_handle,
    )
    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection",
        _fake_polygon_rect_intersection,
    )

    result = overlay_module._few_right_polygon_rect_intersection_owned(
        left_pairs,
        right_pairs,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert result is not None
    assert can_handle_calls == [(True, True)]
    assert rect_calls == [(True, True)]
    for got, exp in zip(result.to_shapely(), sentinel.to_shapely(), strict=True):
        assert got.equals(exp)


def test_overlay_intersection_few_right_large_batches_reuse_cached_right_segments(
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
                    holes=[[(10, -1), (20, -1), (20, 2), (10, 2)]],
                ),
                Polygon(
                    [(60, -2), (180, -2), (180, 3), (60, 3)],
                    holes=[[(90, -1), (100, -1), (100, 2), (90, 2)]],
                ),
                Polygon(
                    [(140, -2), (260, -2), (260, 3), (140, 3)],
                    holes=[[(170, -1), (180, -1), (180, 2), (170, 2)]],
                ),
                Polygon(
                    [(220, -2), (340, -2), (340, 3), (220, 3)],
                    holes=[[(250, -1), (260, -1), (260, 2), (250, 2)]],
                ),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    idx1 = np.arange(320, dtype=np.int32)
    idx2 = np.repeat(np.arange(4, dtype=np.int32), 80)

    rowwise_calls: list[tuple[int, bool]] = []

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        lambda *args, **kwargs: pytest.fail(
            "large few-right batches should stay on one exact rowwise pass"
        ),
    )

    def _fake_rowwise(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU, _cached_right_segments=None):
        rowwise_calls.append((left_arg.row_count, _cached_right_segments is not None))
        geoms = [
            box(float(9000 + i), 0.0, float(9000 + i + 0.5), 0.5)
            for i in range(left_arg.row_count)
        ]
        return from_shapely_geometries(geoms, residency=Residency.DEVICE)

    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
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
    assert rowwise_calls == [(320, True)]
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

    fallback_calls: list[tuple[int, bool]] = []

    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: pytest.fail(
            "few-right intersection fallback should preserve one-batch semantics"
        ),
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda *args, **kwargs: None,
    )

    def _fake_binary(op, left_arg, right_arg, **kwargs):
        assert op == "intersection"
        fallback_calls.append(
            (
                left_arg.row_count,
                bool(kwargs.get("_prefer_exact_polygon_intersection")),
            )
        )
        geoms = [
            box(float(5000 + i), 0.0, float(5000 + i + 0.5), 0.5)
            for i in range(left_arg.row_count)
        ]
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
    assert fallback_calls == [(24, True)]
    assert result["col1"].tolist() == idx1.tolist()
    assert result["zone_type"].tolist() == ["A"] * 8 + ["B"] * 8 + ["C"] * 8


def test_overlay_intersection_exact_mode_prefers_rectangle_kernel_for_rectangles(
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

    kernel_calls: list[int] = []

    def _fake_polygon_rect_intersection(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU):
        kernel_calls.append(left_arg.row_count)
        return from_shapely_geometries(
            [
                box(100.0, 0.0, 101.0, 1.0),
                box(200.0, 0.0, 201.0, 1.0),
            ],
            residency=Residency.DEVICE,
        )

    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection",
        _fake_polygon_rect_intersection,
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
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
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
    assert kernel_calls == [2]
    assert result["col1"].tolist() == [1, 2]


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

    kernel_calls: list[int] = []

    def _fake_polygon_rect_intersection(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU):
        kernel_calls.append(left_arg.row_count)
        return from_shapely_geometries(
            [
                box(100.0, 0.0, 101.0, 1.0),
                box(110.0, 0.0, 111.0, 1.0),
                box(200.0, 0.0, 201.0, 1.0),
                box(210.0, 0.0, 211.0, 1.0),
            ],
            residency=Residency.DEVICE,
        )

    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection",
        _fake_polygon_rect_intersection,
    )
    monkeypatch.setattr(
        polygon_intersection_module,
        "polygon_intersection",
        lambda *args, **kwargs: pytest.fail(
            "indexed rectangle-capable batches should use polygon_rect_intersection before polygon_intersection"
        ),
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda *args, **kwargs: pytest.fail(
            "indexed rectangle-capable batches should use polygon_rect_intersection before rowwise overlay"
        ),
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
    assert kernel_calls == [4]
    assert result["col1"].tolist() == [1, 1, 2, 2]
    assert result["col2"].tolist() == [10, 10, 20, 20]
    assert [geom.bounds for geom in result.geometry] == [
        (100.0, 0.0, 101.0, 1.0),
        (110.0, 0.0, 111.0, 1.0),
        (200.0, 0.0, 201.0, 1.0),
        (210.0, 0.0, 211.0, 1.0),
    ]


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
                box(100.0, 0.0, 101.0, 1.0),
                box(200.0, 0.0, 201.0, 1.0),
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
                box(100.0, 0.0, 101.0, 1.0),
                box(200.0, 0.0, 201.0, 1.0),
            ],
            residency=Residency.DEVICE,
        )
        result._polygon_rect_boundary_overlap = np.zeros(2, dtype=bool)
        return result

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        _fake_binary,
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
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
                box(100.0, 0.0, 101.0, 1.0),
                box(200.0, 0.0, 201.0, 1.0),
            ],
            residency=Residency.DEVICE,
        )
        result._polygon_rect_boundary_overlap = np.zeros(2, dtype=bool)
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
    rowwise_calls: list[int] = []

    def _fake_polygon_rect_intersection(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU):
        rect_calls.append(left_arg.row_count)
        return from_shapely_geometries(
            [box(100.0, 0.0, 101.0, 1.0)],
            residency=Residency.DEVICE,
        )

    def _fake_rowwise(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU, _cached_right_segments=None):
        rowwise_calls.append(left_arg.row_count)
        return from_shapely_geometries(
            [box(200.0, 0.0, 201.0, 1.0)],
            residency=Residency.DEVICE,
        )

    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection",
        _fake_polygon_rect_intersection,
    )
    monkeypatch.setattr(
        polygon_intersection_module,
        "polygon_intersection",
        lambda *args, **kwargs: pytest.fail(
            "mixed rectangle batches should bypass polygon_intersection and split into rectangle + remainder paths"
        ),
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
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
    assert rect_calls == [1]
    assert rowwise_calls == [1]
    assert result["col1"].tolist() == [1, 2]
    assert result["col2"].tolist() == [10, 20]
    assert [geom.bounds for geom in result.geometry] == [
        (100.0, 0.0, 101.0, 1.0),
        (200.0, 0.0, 201.0, 1.0),
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

    def _fake_polygon_rect_intersection(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU):
        rect_calls.append(left_arg.row_count)
        return from_shapely_geometries(
            [
                box(100.0, 0.0, 101.0, 1.0),
                box(200.0, 0.0, 201.0, 1.0),
            ],
            residency=Residency.DEVICE,
        )

    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection_can_handle",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection",
        _fake_polygon_rect_intersection,
    )
    monkeypatch.setattr(
        polygon_intersection_module,
        "polygon_intersection",
        lambda *args, **kwargs: pytest.fail(
            "all-rectangle-capable batches should still recover through the rectangle subset path"
        ),
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda *args, **kwargs: pytest.fail(
            "all-rectangle-capable batches should not fall through to rowwise overlay when the batch probe misses"
        ),
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
    assert rect_calls == [2]
    assert result["col1"].tolist() == [1, 2]
    assert result["col2"].tolist() == [10, 20]
    assert [geom.bounds for geom in result.geometry] == [
        (100.0, 0.0, 101.0, 1.0),
        (200.0, 0.0, 201.0, 1.0),
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
        geometry=GeoSeries([
            box(1.0, 0.0, 3.0, 2.0),
            box(2.0, 0.0, 4.0, 2.0),
        ]),
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


def test_overlay_difference_preserves_left_geometry_name() -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(2, 2, 4, 4)]),
    ).rename(columns={"geometry": "polygons"}).set_geometry("polygons")
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
