from __future__ import annotations

import ast
import importlib
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pandas.testing import assert_series_equal
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

import vibespatial.api as geopandas
import vibespatial.api._native_result_core as native_result_core_module
from vibespatial.api import GeoDataFrame, GeoSeries
from vibespatial.api._native_expression import NativeExpression
from vibespatial.api._native_grouped import NativeGrouped, NativeGroupedSelection
from vibespatial.api._native_metadata import NativeGeometryMetadata, NativeSpatialIndex
from vibespatial.api._native_relation import (
    NativeRelation,
    NativeRelationFamilyPartition,
    NativeRelationSelection,
)
from vibespatial.api._native_result_core import (
    GeometryNativeResult,
    NativeAttributeTable,
    NativeGeometryColumn,
    NativeGeometryComposition,
    NativeGeometryCompositionPart,
    NativeGeometryProvenance,
    NativeTabularResult,
    NativeTabularSelection,
    _host_array,
    _host_row_positions,
)
from vibespatial.api._native_results import (
    GroupedConstructiveResult,
    RelationIndexResult,
    RelationJoinExportResult,
    RelationJoinResult,
    _concat_native_tabular_results,
    _geometry_composition_from_owned_parts_at_capacity,
    _native_attribute_table_from_projected_frames,
    _native_pairwise_attribute_table,
    _ordered_geometry_collection_from_owned_parts_at_capacity,
    _pairwise_constructive_to_native_tabular_result,
    _relation_constructive_to_native_tabular_result,
    _relation_selection_constructive_to_native_tabular_result,
    _rename_native_tabular_result,
)
from vibespatial.api._native_rowset import (
    NativeDeviceSelection,
    NativeIndexPlan,
    NativeRowSet,
)
from vibespatial.api._native_state import (
    NativeFrameState,
    attach_native_state,
    drop_native_state,
    get_native_state,
)
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.api.tools.sjoin import (
    _device_columns_from_native_attributes,
    _sjoin_export_result,
    _sjoin_nearest_relation_result,
    sjoin,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    from_shapely_geometries,
    seed_all_validity_cache,
    seed_homogeneous_host_metadata,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection, has_gpu_runtime
from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events
from vibespatial.runtime.fallbacks import (
    STRICT_NATIVE_ENV_VAR,
    StrictNativeFallbackError,
    clear_fallback_events,
    get_fallback_events,
)
from vibespatial.runtime.materialization import (
    MaterializationBoundary,
    NativeExportBoundary,
    StrictNativeMaterializationError,
    clear_materialization_events,
    get_materialization_events,
    record_materialization_event,
    record_native_export_boundary,
)
from vibespatial.runtime.residency import Residency, TransferTrigger
from vibespatial.spatial.indexing import build_flat_spatial_index


class _Owner:
    pass


class _ExplodingGetArray:
    dtype = np.dtype("int32")

    def __init__(self, values=(0, 1, 2)) -> None:
        self._values = np.asarray(values, dtype=np.int32)
        self.get_called = False

    @property
    def shape(self) -> tuple[int, ...]:
        return self._values.shape

    def __len__(self) -> int:
        return int(self._values.size)

    def get(self):
        self.get_called = True
        raise AssertionError("host materialization should not happen")


class _HostableArray:
    def __init__(self, values=(0, 1, 2)) -> None:
        self._values = np.asarray(values, dtype=np.int32)

    def get(self):
        return self._values


class _FakeDeviceTable:
    shape = (2, 2)


def test_composed_nonempty_mask_uses_capacity_scatter() -> None:
    source = Path(native_result_core_module.__file__).read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "valid_nonempty_mask_device"
    )
    method_source = ast.unparse(method)

    assert "output_rows[d_active]" not in method_source
    assert "maximum.at" in method_source


def test_source_device_to_host_boundaries_are_named() -> None:
    root = Path("src/vibespatial")
    raw_asnumpy: dict[str, list[int]] = {}
    unnamed_runtime_copies: dict[str, list[int]] = {}
    raw_device_gets: dict[str, list[int]] = {}
    allowed_get_owners = {
        "_ACTIVE_SESSION",
        "CCCLPrecompiler",
        "NVRTCPrecompiler",
        "_MATERIALIZATION_CONTEXT",
        "capture",
    }
    classified_get_boundaries = {
        ("src/vibespatial/api/_native_result_core.py", "_host_array"),
        ("src/vibespatial/overlay/_host_boundary.py", "overlay_device_to_host"),
    }

    def _owner_name(node: ast.Call) -> str:
        if not isinstance(node.func, ast.Attribute):
            return ""
        value = node.func.value
        if isinstance(value, ast.Name):
            return value.id
        if isinstance(value, ast.Attribute):
            return value.attr
        return ""

    class _BoundaryVisitor(ast.NodeVisitor):
        def __init__(self, path_key: str) -> None:
            self.path_key = path_key
            self.function_stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if not isinstance(node.func, ast.Attribute):
                self.generic_visit(node)
                return
            if node.func.attr == "asnumpy":
                raw_asnumpy.setdefault(self.path_key, []).append(node.lineno)
            if node.func.attr in {"copy_device_to_host", "copy_device_to_host_async"} and not any(
                keyword.arg == "reason" for keyword in node.keywords
            ):
                unnamed_runtime_copies.setdefault(self.path_key, []).append(node.lineno)
            if (
                node.func.attr == "get"
                and not node.args
                and _owner_name(node) not in allowed_get_owners
            ):
                current_function = self.function_stack[-1] if self.function_stack else ""
                if (self.path_key, current_function) not in classified_get_boundaries:
                    raw_device_gets.setdefault(self.path_key, []).append(node.lineno)
            self.generic_visit(node)

    for path in sorted(root.rglob("*.py")):
        if path == Path("src/vibespatial/cuda/_runtime.py"):
            continue
        path_key = str(path)
        tree = ast.parse(path.read_text(), filename=path_key)
        _BoundaryVisitor(path_key).visit(tree)

    assert raw_asnumpy == {}
    assert unnamed_runtime_copies == {}
    assert raw_device_gets == {}


def test_materialization_events_are_distinct_from_fallbacks() -> None:
    clear_materialization_events()

    event = record_materialization_event(
        surface="test.surface",
        boundary=MaterializationBoundary.USER_EXPORT,
        operation="export",
        reason="test export",
    )

    assert event.boundary is MaterializationBoundary.USER_EXPORT
    assert get_materialization_events() == [event]
    assert get_materialization_events(clear=True) == [event]
    assert get_materialization_events() == []


def test_native_export_boundary_records_target_and_size_contract() -> None:
    clear_materialization_events()

    event = record_native_export_boundary(
        NativeExportBoundary(
            surface="test.native_export",
            operation="native_to_arrow",
            target="arrow",
            reason="explicit test export",
            row_count=3,
            byte_count=96,
            detail="attribute_columns=2",
        )
    )

    assert event.boundary is MaterializationBoundary.USER_EXPORT
    assert event.operation == "native_to_arrow"
    assert "native_export_target=arrow" in event.detail
    assert "rows=3" in event.detail
    assert "bytes=96" in event.detail
    assert "attribute_columns=2" in event.detail
    assert get_materialization_events(clear=True) == [event]


def test_strict_native_disallows_internal_materialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    with pytest.raises(StrictNativeMaterializationError):
        record_materialization_event(
            surface="test.internal",
            boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
            operation="to_host",
            reason="hidden hot-path host conversion",
            strict_disallowed=True,
        )


def test_strict_native_allows_user_export_materialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    event = record_materialization_event(
        surface="test.user_export",
        boundary=MaterializationBoundary.USER_EXPORT,
        operation="to_geodataframe",
        reason="explicit compatibility export",
        strict_disallowed=False,
    )

    assert event.boundary is MaterializationBoundary.USER_EXPORT


def test_host_array_records_materialization_event() -> None:
    clear_materialization_events()

    got = _host_array(_HostableArray(), dtype=np.int64)

    assert got.tolist() == [0, 1, 2]
    events = get_materialization_events(clear=True)
    assert len(events) == 1
    assert events[0].boundary is MaterializationBoundary.INTERNAL_HOST_CONVERSION
    assert events[0].d2h_transfer is True


def test_strict_native_disallows_hidden_host_array_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_materialization_events()

    with pytest.raises(StrictNativeMaterializationError):
        _host_array(_HostableArray(), dtype=np.int64)

    events = get_materialization_events(clear=True)
    assert len(events) == 1
    assert events[0].operation == "array_to_host"
    assert events[0].strict_disallowed is True


def test_strict_native_allows_explicit_host_array_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_materialization_events()

    got = _host_array(
        _HostableArray(),
        dtype=np.int64,
        strict_disallowed=False,
    )

    assert got.tolist() == [0, 1, 2]
    events = get_materialization_events(clear=True)
    assert len(events) == 1
    assert events[0].operation == "array_to_host"
    assert events[0].strict_disallowed is False


def test_host_array_records_caller_materialization_context() -> None:
    clear_materialization_events()

    got = _host_array(
        _HostableArray(),
        dtype=np.int64,
        strict_disallowed=False,
        surface="test.relation_bridge",
        operation="relation_bridge_to_host",
        reason="unit test explicit bridge context",
        detail="side=left, rows=3",
    )

    assert got.tolist() == [0, 1, 2]
    events = get_materialization_events(clear=True)
    assert len(events) == 1
    assert events[0].surface == "test.relation_bridge"
    assert events[0].operation == "relation_bridge_to_host"
    assert events[0].reason == "unit test explicit bridge context"
    assert events[0].detail == "side=left, rows=3"
    assert events[0].strict_disallowed is False


def test_native_geometry_provenance_takes_and_concats_repair_masks() -> None:
    left = NativeGeometryProvenance(
        operation="left",
        row_count=3,
        source_rows=np.asarray([0, 1, 2], dtype=np.int64),
        part_family_tags=np.asarray(
            [
                FAMILY_TAGS[GeometryFamily.POINT],
                FAMILY_TAGS[GeometryFamily.LINESTRING],
                FAMILY_TAGS[GeometryFamily.POLYGON],
            ],
            dtype=np.int8,
        ),
        repaired_mask=np.asarray([False, True, False], dtype=bool),
    )
    right = NativeGeometryProvenance(
        operation="right",
        row_count=2,
        source_rows=np.asarray([10, 11], dtype=np.int64),
        part_family_tags=np.asarray(
            [
                FAMILY_TAGS[GeometryFamily.MULTIPOINT],
                FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
            ],
            dtype=np.int8,
        ),
    )

    taken = left.take(np.asarray([1, 2], dtype=np.int64))
    concatenated = NativeGeometryProvenance.concat(
        [left, right],
        operation="combined",
    )

    assert taken.repaired_mask.tolist() == [True, False]
    assert taken.part_family_tags.tolist() == [
        FAMILY_TAGS[GeometryFamily.LINESTRING],
        FAMILY_TAGS[GeometryFamily.POLYGON],
    ]
    assert concatenated.part_family_tags.tolist() == [
        FAMILY_TAGS[GeometryFamily.POINT],
        FAMILY_TAGS[GeometryFamily.LINESTRING],
        FAMILY_TAGS[GeometryFamily.POLYGON],
        FAMILY_TAGS[GeometryFamily.MULTIPOINT],
        FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
    ]
    assert concatenated.repaired_mask.tolist() == [
        False,
        True,
        False,
        False,
        False,
    ]


def test_strict_native_allows_admitted_relation_index_host_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_materialization_events()
    relation = RelationIndexResult(
        _HostableArray([1, 0]),
        _HostableArray([0, 1]),
    )

    left, right = relation.to_host(
        surface="test.relation_index",
        operation="relation_index_to_host",
        reason="unit test admitted relation bridge",
        dtype=np.intp,
    )

    assert left.tolist() == [1, 0]
    assert right.tolist() == [0, 1]
    events = get_materialization_events(clear=True)
    assert len(events) == 2
    assert {event.detail for event in events} == {"side=left", "side=right"}
    assert all(event.surface == "test.relation_index" for event in events)
    assert all(event.operation == "relation_index_to_host" for event in events)
    assert all(event.strict_disallowed is False for event in events)


def test_pairwise_attribute_bridge_records_admitted_index_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    left = GeoDataFrame(
        {
            "value": [10, 20],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "score": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    clear_materialization_events()

    attributes = _native_pairwise_attribute_table(
        left,
        right,
        _HostableArray([1, 0]),
        _HostableArray([0, 1]),
    )

    frame = attributes.to_pandas(copy=False)
    events = get_materialization_events(clear=True)
    assert frame["value"].tolist() == [20, 10]
    assert frame["score"].tolist() == [1, 2]
    assert len(events) == 2
    assert {(event.surface, event.operation, event.strict_disallowed) for event in events} == {
        (
            "vibespatial.api._native_results._native_pairwise_attribute_table",
            "pairwise_attribute_indices_to_host",
            False,
        )
    }


def test_native_expression_lowers_scalar_comparison_to_rowset() -> None:
    expression = NativeExpression(
        operation="geometry.area",
        values=np.asarray([1.0, np.nan, 4.0, 9.0], dtype=np.float64),
        source_token="frame",
        source_row_count=4,
        dtype="float64",
        precision="fp64",
    )

    rowset = expression.greater_than(3.0)

    assert rowset.positions.tolist() == [2, 3]
    assert rowset.source_token == "frame"
    assert rowset.source_row_count == 4
    assert rowset.ordered is True
    assert rowset.unique is True
    assert expression.between(3.0, 9.0).positions.tolist() == [2, 3]
    assert expression.between(3.0, 9.0, inclusive="neither").positions.tolist() == [2]
    assert expression.equal_to(9.0).positions.tolist() == [3]
    assert expression.equal_to_selection(9.0).positions.tolist() == [3]
    assert expression.not_equal(1.0).positions.tolist() == [2, 3]

    with pytest.raises(ValueError, match="finite threshold"):
        expression.greater_than(np.inf)
    with pytest.raises(ValueError, match="finite thresholds"):
        expression.between(-np.inf, 9.0)
    with pytest.raises(ValueError, match="lower threshold"):
        expression.between(9.0, 3.0)
    with pytest.raises(ValueError, match="source_row_count"):
        NativeExpression(
            operation="geometry.area",
            values=np.asarray([1.0, 2.0], dtype=np.float64),
            source_row_count=3,
        )


def test_native_device_selection_contract_has_no_implicit_count_fence() -> None:
    source = Path(importlib.import_module("vibespatial.api._native_rowset").__file__).read_text()
    method_source = source.split(
        "    def from_mask(\n",
        1,
    )[1].split("\n    @property", 1)[0]

    assert "cp.cumsum" in method_source
    assert "scatter_selected_positions_i64" in method_source
    assert "copy_device_to_host" not in method_source
    assert ".get(" not in method_source
    assert "int(d_logical_count" not in method_source
    assert "selected_count + row - inclusive_offsets[row]" in source
    assert "def partition_capacity_positions(" in source
    prefix_source = source.split(
        "    def as_capacity_prefix(\n",
        1,
    )[1].split("\n    def _host_logical_count", 1)[0]
    assert "positions=cp.arange(self.capacity" in prefix_source
    assert "logical_count=self.logical_count" in prefix_source
    assert "copy_device_to_host" not in prefix_source
    source_mask = source.split(
        "    def source_mask(",
        1,
    )[1].split("\n    def as_capacity_prefix", 1)[0]
    assert "self.active_capacity_mask()" in source_mask
    assert "self.safe_capacity_positions()" in source_mask
    assert "copy_device_to_host" not in source_mask
    assert "cp.flatnonzero" not in source_mask


def test_composed_family_selection_masks_capacity_without_part_compaction() -> None:
    source = Path(native_result_core_module.__file__).read_text()
    method_source = source.split(
        "    def select_family_domain_device(self, families):",
        1,
    )[1].split("\n    def mask_capacity", 1)[0]

    assert "cp.maximum.at(" in method_source
    assert "part.geometry.mask_capacity(d_part_keep)" in method_source
    assert "cp.flatnonzero" not in method_source
    assert "_device_indexed_take" not in method_source


def test_native_tabular_selection_contract_preserves_exact_result_invariant() -> None:
    rowset_source = Path(
        importlib.import_module("vibespatial.api._native_rowset").__file__
    ).read_text()
    concatenate_source = rowset_source.split(
        "    def concatenate(\n",
        1,
    )[1].split("\n    @property", 1)[0]
    result_source = Path(native_result_core_module.__file__).read_text()
    result_adapter_source = Path(
        importlib.import_module("vibespatial.api._native_results").__file__
    ).read_text()
    selection_source = result_source.split(
        "class NativeTabularSelection:",
        1,
    )[1].split("\ndef _materialize_attribute_geometry_frame", 1)[0]
    rename_source = result_adapter_source.split(
        "def _rename_native_tabular_result(",
        1,
    )[1].split("\ndef _pairwise_constructive_native_state_attribute_parts", 1)[0]
    concat_source = result_adapter_source.split(
        "def _concat_native_tabular_results(",
        1,
    )[1].split("\ndef _concat_native_tabular_selections", 1)[0]

    assert "copy_device_to_host" not in concatenate_source
    assert ".get(" not in concatenate_source
    assert "NativeDeviceSelection" in selection_source
    assert "capacity_result: NativeTabularResult" in selection_source
    assert "compact_rowset(" in selection_source
    assert "class NativeTabularResult" not in selection_source
    assert "isinstance(result, NativeTabularSelection)" in rename_source
    assert "capacity_result=_rename_native_tabular_result(" in rename_source
    assert "_concat_native_tabular_selections(" in concat_source


def test_polygonal_part_native_result_preserves_capacity_selection() -> None:
    source = Path(importlib.import_module("vibespatial.api._native_results").__file__).read_text()
    producer_source = source.split(
        "def _polygonal_parts_constructive_to_native_tabular_result(",
        1,
    )[1].split("\ndef ", 1)[0]
    tag_source = source.split(
        "def _owned_row_family_tags(",
        1,
    )[1].split("\ndef ", 1)[0]

    assert "_explode_polygonal_rows_to_polygon_capacity_gpu" in producer_source
    assert "NativeTabularSelection(" in producer_source
    assert "selection=selection" in producer_source
    assert "_explode_polygonal_rows_to_polygons_gpu" not in producer_source
    assert "preserve_indexed_view=True" in tag_source


def test_native_relation_selection_consumes_capacity_without_compaction() -> None:
    source = Path(importlib.import_module("vibespatial.api._native_relation").__file__).read_text()
    relation_selection_source = source.split(
        "class NativeRelationSelection:",
        1,
    )[1].split("\n\n__all__", 1)[0]
    producer_source = source.split(
        "class NativeRelation:",
        1,
    )[1].split("\n\n@dataclass(frozen=True)\nclass NativeRelationSelection:", 1)[0]

    assert "def filter_by_distance_selection(" in producer_source
    assert "def filter_by_equal_columns_selection(" in producer_source
    assert "NativeDeviceSelection.from_mask(" in producer_source
    grouped_source = Path(
        importlib.import_module("vibespatial.api._native_grouped").__file__
    ).read_text()
    grouped_selection_source = grouped_source.split(
        "class NativeGroupedSelection:",
        1,
    )[1].split("\n\n@dataclass(frozen=True)\nclass NativeGrouped:", 1)[0]

    assert "NativeGroupedSelection(" in relation_selection_source
    assert "def physicalize_geometries(" in relation_selection_source
    assert ".device_take_capacity(" in relation_selection_source
    assert "def constructive_native(" in relation_selection_source
    constructive_source = relation_selection_source.split(
        "    def constructive_native(",
        1,
    )[1].split("\n\n@dataclass(frozen=True)\nclass NativeRelationConstructiveSelection", 1)[0]
    assert ".physicalize_device_rows(" not in constructive_source
    assert "selection.gather_capacity(" in grouped_selection_source
    assert "selection.active_capacity_mask()" in grouped_selection_source
    assert "compact_rowset(" not in grouped_selection_source
    assert "copy_device_to_host" not in grouped_selection_source
    assert "int(self.logical_count" not in grouped_selection_source
    assert "def left_reduce_right_numeric(" in relation_selection_source
    assert "def right_reduce_left_numeric(" in relation_selection_source
    result_adapter_source = Path(
        importlib.import_module("vibespatial.api._native_results").__file__
    ).read_text()
    selection_constructive_source = result_adapter_source.split(
        "def _relation_selection_constructive_to_native_tabular_result(",
        1,
    )[1].split("\ndef _pairwise_constructive_result_to_native_tabular_result", 1)[0]
    assert ".physicalize_geometries(" in selection_constructive_source
    assert ".constructive_native(" in selection_constructive_source
    assert ".gather_capacity(" in selection_constructive_source
    assert ".as_capacity_prefix()" in selection_constructive_source
    assert "compact_rowset(" not in selection_constructive_source
    assert "to_host" not in selection_constructive_source
    owned_source = Path(importlib.import_module("vibespatial.geometry.owned").__file__).read_text()
    capacity_take_source = owned_source.split(
        "    def device_take_capacity(",
        1,
    )[1].split("\n    def _physical_device_take(", 1)[0]
    assert "copy_device_to_host" not in capacity_take_source
    assert "cp.asnumpy" not in capacity_take_source


@pytest.mark.gpu
def test_native_relation_selection_reduces_source_values_without_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for selected relation reduction")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    relation = NativeRelation(
        left_indices=cp.asarray([0, -99, 0, 99], dtype=cp.int32),
        right_indices=cp.asarray([0, 999_999, 2, -9], dtype=cp.int32),
        left_token="query",
        right_token="tree",
        left_row_count=3,
        right_row_count=3,
    )
    selected = NativeRelationSelection(
        relation=relation,
        selection=NativeDeviceSelection.from_mask(
            cp.asarray([True, False, True, False]),
            source_token="query",
            source_row_count=4,
        ),
    )
    values = cp.asarray([1.5, 2.5, 4.0], dtype=cp.float64)
    large_values = cp.asarray([2**53 + 1, 0, 2], dtype=cp.int64)
    bool_values = cp.asarray([True, False, True], dtype=cp.bool_)
    reset_d2h_transfer_count()

    with assert_zero_d2h_transfers():
        reduced = selected.left_reduce_right_numeric(
            values,
            "sum",
            left_row_count=3,
        )
        large_reduced = selected.left_reduce_right_numeric(
            large_values,
            "sum",
            left_row_count=3,
        )
        bool_reduced = selected.left_reduce_right_numeric(
            bool_values,
            "sum",
            left_row_count=3,
        )
        counts = selected.left_match_count_expression(source_row_count=3)

    assert cp.asnumpy(reduced.values).tolist() == [5.5, 0.0, 0.0]
    assert cp.asnumpy(large_reduced.values).tolist() == [2**53 + 3, 0, 0]
    assert large_reduced.values.dtype == cp.dtype(cp.int64)
    assert cp.asnumpy(bool_reduced.values).tolist() == [2, 0, 0]
    assert bool_reduced.values.dtype == cp.dtype(cp.int64)
    assert cp.asnumpy(counts.values).tolist() == [2, 0, 0]
    reset_d2h_transfer_count()


@pytest.mark.gpu
def test_native_device_selection_keeps_logical_count_on_device() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for capacity-backed selection")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    expression = NativeExpression(
        operation="predicate.intersects",
        values=cp.asarray([False, True, False, True, True]),
        source_token="frame",
        source_row_count=5,
        dtype="bool",
        precision="predicate",
    )
    clear_materialization_events()
    reset_d2h_transfer_count()

    with assert_zero_d2h_transfers():
        selection = expression.equal_to_selection(True)
        gathered = selection.gather_capacity(
            cp.asarray([10, 20, 30, 40, 50], dtype=cp.int64),
            fill_value=-1,
        )
        active = selection.active_capacity_mask()
        source_mask = selection.source_mask(
            active_mask=cp.asarray([True, False, True, True, True]),
        )

    assert isinstance(selection, NativeDeviceSelection)
    assert selection.capacity == 5
    assert selection.source_token == "frame"
    with pytest.raises(TypeError, match="logical length is device-resident"):
        len(selection)
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(selection.logical_count).tolist() == [3]
    assert cp.asnumpy(active).tolist() == [True, True, True, False, False]
    assert cp.asnumpy(selection.partition_capacity_positions()).tolist() == [1, 3, 4, 0, 2]
    assert cp.asnumpy(gathered).tolist() == [20, 40, 50, -1, -1]
    assert cp.asnumpy(source_mask).tolist() == [False, True, False, False, True]
    reset_d2h_transfer_count()


@pytest.mark.gpu
def test_native_device_selection_identity_requires_stable_full_partition() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for capacity-backed selection")
    cp = pytest.importorskip("cupy")

    stable = NativeDeviceSelection.from_mask(
        cp.asarray([True, True, True]),
        source_row_count=3,
        geometry_family_domain=(GeometryFamily.POLYGON,),
        trusted_all_valid_rows=True,
    )
    reordered = NativeDeviceSelection(
        positions=cp.asarray([2, 1, 0], dtype=cp.int64),
        logical_count=cp.asarray([3], dtype=cp.int64),
        source_row_count=3,
        ordered=True,
        unique=True,
    )

    stable_rows = stable.compact_rowset(strict_disallowed=False)
    reordered_rows = reordered.compact_rowset(strict_disallowed=False)

    assert stable_rows.identity is True
    assert stable_rows.geometry_family_domain == (GeometryFamily.POLYGON,)
    assert stable_rows.trusted_all_valid_rows is True
    assert reordered_rows.identity is False


@pytest.mark.gpu
def test_native_device_selection_remaps_full_capacity_partition_without_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for capacity-backed selection")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import assert_zero_d2h_transfers

    selection = NativeDeviceSelection.from_mask(
        cp.asarray([False, True, True, False]),
        source_token="candidate",
        source_row_count=4,
    )

    with assert_zero_d2h_transfers():
        remapped = selection.remap_source_positions(
            cp.asarray([8, 3, 6, 1], dtype=cp.int64),
            source_token="source",
            source_row_count=10,
        )

    assert remapped.source_token == "source"
    assert remapped.source_row_count == 10
    assert cp.asnumpy(remapped.logical_count).tolist() == [2]
    assert cp.asnumpy(remapped.partition_capacity_positions()).tolist() == [3, 6, 8, 1]


@pytest.mark.gpu
def test_native_tabular_selection_concatenates_dynamic_prefixes_without_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for dynamic tabular selection")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    def _capacity_result(values, x_offset: float) -> NativeTabularResult:
        owned = from_shapely_geometries(
            [Point(x_offset + index, 0.0) for index in range(len(values))],
            residency=Residency.DEVICE,
        )
        return NativeTabularResult(
            attributes=pd.DataFrame({"value": values}),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )

    first = NativeTabularSelection(
        capacity_result=_capacity_result([10, 11], 0.0),
        selection=NativeDeviceSelection.from_mask(
            cp.asarray([False, True]),
            source_row_count=2,
        ),
    )
    second = NativeTabularSelection(
        capacity_result=_capacity_result([20, 21, 22], 10.0),
        selection=NativeDeviceSelection.from_mask(
            cp.asarray([True, False, True]),
            source_row_count=3,
        ),
    )
    clear_materialization_events()
    reset_d2h_transfer_count()

    with assert_zero_d2h_transfers():
        renamed = _rename_native_tabular_result(
            first,
            {"value": "renamed"},
        )
        combined = _concat_native_tabular_results(
            [first, second],
            geometry_name="geometry",
            crs=None,
        )

    assert isinstance(renamed, NativeTabularSelection)
    assert tuple(renamed.capacity_result.attributes.columns) == ("renamed",)
    assert isinstance(combined, NativeTabularSelection)
    assert combined.capacity == 5
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(combined.logical_count).tolist() == [3]
    assert cp.asnumpy(combined.selection.positions).tolist() == [1, 2, 4, 0, 3]
    reset_d2h_transfer_count()


def test_native_expression_binary_arithmetic_lowers_to_rowset() -> None:
    wet_area = NativeExpression(
        operation="attribute.wet_area",
        values=np.asarray([0.2, np.nan, 4.0, 9.0], dtype=np.float64),
        source_token="frame",
        source_row_count=4,
        dtype="float64",
        precision="fp64",
    )
    parcel_area = NativeExpression(
        operation="attribute.parcel_area",
        values=np.asarray([1.0, 2.0, 2.0, 3.0], dtype=np.float64),
        source_token="frame",
        source_row_count=4,
        dtype="float64",
        precision="fp64",
    )

    ratio = wet_area / parcel_area

    assert isinstance(ratio, NativeExpression)
    assert np.allclose(
        np.asarray(ratio.values),
        np.asarray([0.2, np.nan, 2.0, 3.0]),
        equal_nan=True,
    )
    assert ratio.source_token == "frame"
    assert ratio.source_row_count == 4
    assert ratio.greater_equal(2.0).positions.tolist() == [2, 3]
    assert ratio.compare("<", 1.0).positions.tolist() == [0]
    assert (wet_area - 1.0).greater_than(2.0).positions.tolist() == [2, 3]

    mismatched = NativeExpression(
        operation="attribute.other",
        values=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        source_token="other-frame",
        source_row_count=4,
    )
    assert wet_area.binary_arithmetic("/", mismatched) is None
    assert wet_area.compare(">=", mismatched) is None


def test_native_expression_integer_scalar_arithmetic_preserves_integer_dtype() -> None:
    values = np.asarray([2024, 2025, 2026], dtype=np.int32)
    expression = NativeExpression(
        operation="datetime.year",
        values=values,
        source_token="frame",
        source_row_count=3,
        dtype="int32",
        precision="source",
    )

    multiplied = expression * 12
    packed = multiplied + 8
    divided = expression / 2

    assert np.issubdtype(np.asarray(multiplied.values).dtype, np.integer)
    assert np.issubdtype(np.asarray(packed.values).dtype, np.integer)
    assert np.asarray(packed.values).tolist() == [24296, 24308, 24320]
    assert np.issubdtype(np.asarray(divided.values).dtype, np.floating)


def test_native_expression_declines_unrepresentable_integer_scalar() -> None:
    expression = NativeExpression(
        operation="attribute.value",
        values=np.asarray([1, 2, 3], dtype=np.int64),
        source_token="frame",
        source_row_count=3,
    )

    assert expression.binary_arithmetic("+", 1 << 80) is None


def test_native_topk_memory_estimate_scales_with_float_sort_keys() -> None:
    import pyarrow as pa

    from vibespatial.api.geodataframe import _native_topk_required_bytes

    one_key = _native_topk_required_bytes(1_000, (pa.float64(),))
    three_keys = _native_topk_required_bytes(
        1_000,
        (pa.float64(), pa.float64(), pa.int64()),
    )

    assert one_key is not None and one_key > 1_000 * 64
    assert three_keys is not None and three_keys > one_key
    assert _native_topk_required_bytes(1_000, (pa.string(),)) is None


def _device_topk_frame(
    table: pa.Table,
    *,
    composed: bool = False,
) -> GeoDataFrame:
    plc = pytest.importorskip("pylibcudf")
    row_count = table.num_rows
    index = pd.Index(
        [f"row-{position // 2}" for position in range(row_count)],
        name="source",
    )
    public = table.to_pandas()
    for column in table.column_names:
        if pa.types.is_boolean(table.schema.field(column).type):
            public[column] = pd.array(table[column].to_pylist(), dtype="boolean")
    owned = from_shapely_geometries(
        [Point(position, position) for position in range(row_count)],
        residency=Residency.DEVICE,
    )
    gdf = GeoDataFrame(
        {
            **{column: public[column].array for column in table.column_names},
            "geometry": GeoSeries(
                DeviceGeometryArray._from_owned(owned),
                index=index,
                name="geometry",
            ),
        },
        index=index,
    )
    if composed:
        parts = tuple(
            NativeAttributeTable(
                device_table=plc.Table.from_arrow(table.select([column])),
                index_override=index,
                column_override=(column,),
                schema_override=table.select([column]).schema,
            )
            for column in table.column_names
        )
        attributes = NativeAttributeTable(
            parts=parts,
            index_override=index,
            column_override=tuple(table.column_names),
        )
    else:
        attributes = NativeAttributeTable(
            device_table=plc.Table.from_arrow(table),
            index_override=index,
            column_override=tuple(table.column_names),
            schema_override=table.schema,
        )
    attach_native_state(
        gdf,
        NativeFrameState.from_native_tabular_result(
            NativeTabularResult(
                attributes=attributes,
                geometry=GeometryNativeResult.from_owned(owned, crs=None),
                geometry_name="geometry",
                column_order=(*table.column_names, "geometry"),
            )
        ),
    )
    return gdf


@pytest.mark.parametrize(
    ("keep", "expected_positions"),
    [
        pytest.param("first", [1], id="first"),
        pytest.param("last", [4], id="last"),
        pytest.param("all", [1, 2, 4], id="all"),
    ],
)
def test_geodataframe_topk_refines_primary_skew_and_exact_ties_device_native(
    keep: str,
    expected_positions: list[int],
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device top-k")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    table = pa.table(
        {
            "primary": pa.array([10, 10, 10, 10, 10], type=pa.int64()),
            "secondary": pa.array([2, 3, 3, 1, 3], type=pa.int64()),
            "source_position": pa.array(range(5), type=pa.int64()),
        }
    )
    gdf = _device_topk_frame(table, composed=True)
    reset_d2h_transfer_count()
    clear_materialization_events()
    clear_dispatch_events()
    clear_fallback_events()

    with assert_zero_d2h_transfers():
        selected = gdf.nlargest(1, ["primary", "secondary"], keep=keep)
        selected_state = get_native_state(selected)

    assert selected_state is not None
    assert selected_state.attributes.is_device_backed
    assert selected["source_position"].tolist() == expected_positions
    assert selected.index.tolist() == [gdf.index[position] for position in expected_positions]
    assert get_fallback_events(clear=True) == []
    assert [event.implementation for event in get_dispatch_events(clear=True)] == [
        "native_device_topk_rowset"
    ]
    reset_d2h_transfer_count()


def test_geodataframe_topk_prepares_later_keys_only_for_primary_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device top-k shape telemetry")
    from vibespatial.runtime.hotpath_trace import (
        reset_hotpath_trace,
        summarize_hotpath_trace,
    )

    row_count = 250_000
    table = pa.table(
        {
            "primary": pa.array(np.arange(row_count, dtype=np.float64)),
            "secondary": pa.array(np.arange(row_count, dtype=np.float64) % 997),
            "tertiary": pa.array(np.arange(row_count, dtype=np.int64)),
            "source_position": pa.array(np.arange(row_count, dtype=np.int64)),
        }
    )
    gdf = _device_topk_frame(table, composed=True)
    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "counter")
    reset_hotpath_trace()

    selected = gdf.nlargest(100, ["primary", "secondary", "tertiary"])
    stages = summarize_hotpath_trace()

    assert selected["source_position"].tolist() == list(
        range(row_count - 1, row_count - 101, -1)
    )
    prepare = next(
        stage for stage in stages if stage["name"] == "tabular.topk.active_key_prepare"
    )
    assert prepare["calls"] == 1
    assert prepare["metadata"]["work_amplification"]["sum"][
        "prepared_key_rows"
    ] == row_count
    assert prepare["metadata"]["work_amplification"]["max"][
        "active_rows"
    ] == row_count
    selected_gather = next(
        stage for stage in stages if stage["name"] == "tabular.topk.selected_key_gather"
    )
    assert selected_gather["metadata"]["work_amplification"]["max"][
        "selected_rows"
    ] == 100
    rowset_take = next(
        stage for stage in stages if stage["name"] == "tabular.topk.rowset_take"
    )
    assert rowset_take["metadata"]["work_amplification"]["max"][
        "output_rows"
    ] == 100


@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
@pytest.mark.parametrize("keep", ["first", "last", "all"])
def test_geodataframe_topk_float_semantic_matrix_matches_pandas(
    method: str,
    keep: str,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device top-k semantic matrix")
    table = pa.table(
        {
            "primary": pa.array(
                [np.nan, -0.0, 0.0, 2.0, 2.0, 1.0, 1.0, None],
                type=pa.float64(),
            ),
            "secondary": pa.array([9, 2, 1, 0, 0, 3, 3, 8], type=pa.int64()),
            "source_position": pa.array(range(8), type=pa.int64()),
        }
    )
    gdf = _device_topk_frame(table, composed=True)
    reference = table.to_pandas()

    selected = getattr(gdf, method)(
        4,
        ["primary", "secondary"],
        keep=keep,
    )
    expected = getattr(reference, method)(
        4,
        ["primary", "secondary"],
        keep=keep,
    )

    assert selected["source_position"].tolist() == expected[
        "source_position"
    ].tolist()


@pytest.mark.parametrize(
    ("columns", "n", "expected_positions"),
    [
        pytest.param(["a", "b"], 3, [0, 1, 3], id="multikey-truncated-boundary"),
        pytest.param(["a", "b"], 4, [0, 1, 2, 3], id="multikey-full-boundary"),
        pytest.param(["single"], 3, [1, 0, 2], id="single-key-reverses-selected-ties"),
    ],
)
def test_geodataframe_topk_keep_last_matches_pandas_recursive_tie_order(
    columns: list[str],
    n: int,
    expected_positions: list[int],
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device top-k tie ordering")
    table = pa.table(
        {
            "a": pa.array([3, 3, 2, 2, 1], type=pa.int64()),
            "b": pa.array([7, 7, 5, 5, 0], type=pa.int64()),
            "single": pa.array([4, 4, 3, 2, 1], type=pa.int64()),
            "source_position": pa.array(range(5), type=pa.int64()),
        }
    )
    gdf = _device_topk_frame(table)

    selected = gdf.nlargest(n, columns, keep="last")

    assert selected["source_position"].tolist() == expected_positions


@pytest.mark.parametrize(
    "key_array",
    [
        pytest.param(
            pa.array([3, None, 2, None, 1], type=pa.int64()),
            id="nullable-int64",
        ),
        pytest.param(
            pa.array(
                [
                    pd.Timestamp("2026-01-03"),
                    None,
                    pd.Timestamp("2026-01-02"),
                    None,
                    pd.Timestamp("2026-01-01"),
                ],
                type=pa.timestamp("ns"),
            ),
            id="nullable-timestamp",
        ),
        pytest.param(
            pa.array([True, None, False, None, True], type=pa.bool_()),
            id="nullable-bool",
        ),
    ],
)
@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
def test_geodataframe_topk_nullable_fixed_width_matches_pandas(
    key_array: pa.Array,
    method: str,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for nullable device top-k")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    table = pa.table(
        {
            "key": key_array,
            "source_position": pa.array(range(len(key_array)), type=pa.int64()),
        }
    )
    gdf = _device_topk_frame(table)
    reference = table.to_pandas()
    if pa.types.is_boolean(key_array.type):
        reference["key"] = pd.array(key_array.to_pylist(), dtype="boolean")
    expected = getattr(reference, method)(4, "key", keep="first")
    reset_d2h_transfer_count()
    clear_materialization_events()
    clear_fallback_events()

    with assert_zero_d2h_transfers():
        selected = getattr(gdf, method)(4, "key", keep="first")
        selected_state = get_native_state(selected)

    assert selected_state is not None
    assert selected["source_position"].tolist() == expected["source_position"].tolist()
    assert get_fallback_events(clear=True) == []
    reset_d2h_transfer_count()


@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
@pytest.mark.parametrize("keep", ["first", "last", "all"])
def test_geodataframe_topk_preserves_complete_missing_primary_bucket(
    method: str,
    keep: str,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for nullable device top-k")
    table = pa.table(
        {
            "primary": pa.array([5.0, 4.0, 3.0, None, None]),
            "secondary": pa.array([0.0, 0.0, 0.0, 2.0, 1.0]),
            "source_position": pa.array(range(5), type=pa.int64()),
        }
    )
    frame = _device_topk_frame(table)
    expected = getattr(table.to_pandas(), method)(
        4,
        ["primary", "secondary"],
        keep=keep,
    )

    selected = getattr(frame, method)(
        4,
        ["primary", "secondary"],
        keep=keep,
    )

    assert get_native_state(selected) is not None
    assert selected["source_position"].tolist() == expected[
        "source_position"
    ].tolist()


@pytest.mark.parametrize("n", [0, 20])
def test_geodataframe_topk_empty_and_full_selection_stay_device_native(n: int) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for boundary device top-k")
    table = pa.table(
        {
            "key": pa.array([2, None, 1], type=pa.int64()),
            "source_position": pa.array(range(3), type=pa.int64()),
        }
    )
    gdf = _device_topk_frame(table)

    selected = gdf.nsmallest(n, "key", keep="last")
    expected = table.to_pandas().nsmallest(n, "key", keep="last")

    assert get_native_state(selected) is not None
    assert selected["source_position"].tolist() == expected["source_position"].tolist()


def test_geodataframe_topk_admission_precedes_row_view_gather_and_strict_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device top-k admission")
    from vibespatial.api import geodataframe as geodataframe_module
    from vibespatial.cuda._runtime import DeviceMemoryAdmission, get_cuda_runtime
    from vibespatial.testing import strict_native_environment

    table = pa.table(
        {
            "key": pa.array([3, 2, 1], type=pa.int64()),
            "source_position": pa.array(range(3), type=pa.int64()),
        }
    )
    gdf = _device_topk_frame(table)
    state = get_native_state(gdf)
    assert state is not None
    runtime = get_cuda_runtime()
    admission_calls = []

    def _decline(*, stage, required_bytes, requested_units=0):
        admission_calls.append((stage, required_bytes, requested_units))
        return DeviceMemoryAdmission(
            stage=stage,
            required_bytes=required_bytes,
            remaining_bytes=0,
            budget_bytes=0,
            admitted=False,
            requested_units=requested_units,
            admitted_units=0,
            bytes_per_unit=required_bytes,
        )

    monkeypatch.setattr(runtime, "admit_device_memory", _decline)
    monkeypatch.setattr(
        NativeAttributeTable,
        "to_pylibcudf_columns",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("declined top-k must not gather logical keys")
        ),
    )

    assert geodataframe_module._native_topk_rowset(
        gdf,
        1,
        "key",
        largest=True,
        keep="first",
    ) is None
    assert admission_calls and admission_calls[0][0] == "tabular-topk"
    clear_fallback_events()
    with strict_native_environment(), pytest.raises(StrictNativeFallbackError):
        gdf.nlargest(1, "key")
    assert [event.surface for event in get_fallback_events(clear=True)] == [
        "geopandas.geodataframe.nlargest"
    ]


def test_datetime_component_host_fallback_is_observable_and_strict_rejects() -> None:
    from vibespatial.testing import strict_native_environment

    frame = geopandas.GeoDataFrame(
        {
            "when": pd.to_datetime(["2024-01-02", "2025-03-04"]),
            "later": pd.to_datetime(["2024-01-02 00:01", "2025-03-04 00:02"]),
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        crs="EPSG:4326",
    )
    geopandas.clear_fallback_events()

    result = frame.datetime_component("when", "month")
    elapsed = frame.datetime_difference_seconds("when", "later")
    events = geopandas.get_fallback_events(clear=True)

    assert result.tolist() == [1, 3]
    assert elapsed.tolist() == [60.0, 120.0]
    assert any(
        event.surface == "geopandas.geodataframe.datetime_component"
        for event in events
    )
    assert any(
        event.surface == "geopandas.geodataframe.datetime_difference_seconds"
        for event in events
    )
    with strict_native_environment(), pytest.raises(
        RuntimeError,
        match="disallows geopandas fallback",
    ):
        frame.datetime_component("when", "year")


def test_native_expression_guarded_comparison_exposes_ambiguous_rowset() -> None:
    expression = NativeExpression(
        operation="geometry.area",
        values=np.asarray([1.0, np.nan, 4.0, 4.0 + 5e-10, 9.0], dtype=np.float64),
        source_token="frame",
        source_row_count=5,
        dtype="float64",
        precision="fp64",
    )

    guarded = expression.greater_than_guarded(4.0, epsilon=1e-9)

    assert guarded.rowset.positions.tolist() == [4]
    assert guarded.ambiguous.positions.tolist() == [2, 3]
    assert guarded.ambiguous_rowset.source_token == "frame"
    assert guarded.ambiguous_rowset.source_row_count == 5
    assert guarded.operation == "geometry.area>"
    assert guarded.scalar == 4.0
    assert guarded.epsilon == 1e-9
    assert guarded.inclusive is None
    assert guarded.is_device is False
    assert guarded.is_unambiguous is False

    exact_boundary = expression.greater_equal_guarded(4.0, epsilon=0.0)
    assert exact_boundary.rowset.positions.tolist() == [3, 4]
    assert exact_boundary.ambiguous.positions.tolist() == [2]

    ranged = expression.between_guarded(1.0, 9.0, inclusive="both", epsilon=0.0)
    assert ranged.rowset.positions.tolist() == [2, 3]
    assert ranged.ambiguous.positions.tolist() == [0, 4]
    assert ranged.scalar == (1.0, 9.0)
    assert ranged.inclusive == "both"

    with pytest.raises(ValueError, match="epsilon"):
        expression.less_equal_guarded(4.0, epsilon=np.nan)
    with pytest.raises(ValueError, match="epsilon"):
        expression.between_guarded(1.0, 9.0, epsilon=-1.0)


def test_native_expression_guarded_comparison_stays_device_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native expression guard")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    expression = NativeExpression(
        operation="geometry.area",
        values=cp.asarray([1.0, 4.0, 4.0 + 5e-10, 9.0, np.nan], dtype=cp.float64),
        source_token="frame",
        source_row_count=5,
        dtype="float64",
        precision="fp64",
    )
    reset_d2h_transfer_count()

    with assert_zero_d2h_transfers():
        guarded = expression.greater_than_guarded(4.0, epsilon=1e-9)
        ranged = expression.between_guarded(1.0, 9.0, inclusive="both", epsilon=0.0)
        assert guarded.rowset.is_device
        assert guarded.ambiguous_rowset.is_device
        assert ranged.rowset.is_device
        assert ranged.ambiguous_rowset.is_device
        assert len(guarded.rowset) == 1
        assert len(guarded.ambiguous_rowset) == 2

    assert cp.asnumpy(guarded.rowset.positions).tolist() == [3]
    assert cp.asnumpy(guarded.ambiguous.positions).tolist() == [1, 2]
    assert cp.asnumpy(ranged.rowset.positions).tolist() == [1, 2]
    assert cp.asnumpy(ranged.ambiguous.positions).tolist() == [0, 3]
    reset_d2h_transfer_count()


def test_native_rowset_set_algebra_preserves_source_contract() -> None:
    left = NativeRowSet.from_positions(
        np.asarray([3, 1, 1], dtype=np.int64),
        source_token="frame",
        source_row_count=5,
    )
    right = NativeRowSet.from_positions(
        np.asarray([1, 4], dtype=np.int64),
        source_token="frame",
        source_row_count=5,
    )

    intersection = left.intersection(right)
    union = left.union(right)
    difference = left.difference(right)
    complement = left.complement()

    assert intersection.positions.tolist() == [1]
    assert union.positions.tolist() == [1, 3, 4]
    assert difference.positions.tolist() == [3]
    assert complement.positions.tolist() == [0, 2, 4]
    assert intersection.source_token == "frame"
    assert intersection.source_row_count == 5
    assert intersection.ordered is True
    assert intersection.unique is True

    stale = NativeRowSet.from_positions(
        np.asarray([1], dtype=np.int64),
        source_token="stale",
        source_row_count=5,
    )
    wrong_count = NativeRowSet.from_positions(
        np.asarray([1], dtype=np.int64),
        source_token="frame",
        source_row_count=4,
    )
    out_of_bounds = NativeRowSet.from_positions(
        np.asarray([5], dtype=np.int64),
        source_token="frame",
        source_row_count=5,
    )

    with pytest.raises(ValueError, match="source token"):
        left.intersection(stale)
    with pytest.raises(ValueError, match="source row count"):
        left.intersection(wrong_count)
    with pytest.raises(ValueError, match="within source_row_count"):
        left.union(out_of_bounds)
    with pytest.raises(ValueError, match="within source_row_count"):
        NativeRowSet.from_positions(
            np.asarray([], dtype=np.int64),
            source_token="frame",
            source_row_count=5,
            ordered=True,
            unique=True,
        ).intersection(out_of_bounds)
    with pytest.raises(ValueError, match="requires source_row_count"):
        NativeRowSet.from_positions(np.asarray([1], dtype=np.int64)).complement()


def test_native_rowset_identity_to_host_positions_skips_device_copy() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device identity rowset")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    rowset = NativeRowSet.from_positions(
        cp.arange(4, dtype=cp.int64),
        source_token="frame",
        source_row_count=4,
        ordered=True,
        unique=True,
        identity=True,
    )

    clear_materialization_events()
    reset_d2h_transfer_count()
    with assert_zero_d2h_transfers():
        positions = rowset.to_host_positions(surface="unit.identity_rowset")

    assert positions.tolist() == [0, 1, 2, 3]
    assert get_materialization_events(clear=True) == []


def test_native_rowset_identity_set_algebra_shortcuts_preserve_device_carriers() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device identity rowset algebra")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    identity_positions = cp.arange(5, dtype=cp.int64)
    member_positions = cp.asarray([1, 3], dtype=cp.int64)
    empty_positions = cp.asarray([], dtype=cp.int64)
    identity = NativeRowSet.from_positions(
        identity_positions,
        source_token="frame",
        source_row_count=5,
        ordered=True,
        unique=True,
        identity=True,
    )
    members = NativeRowSet.from_positions(
        member_positions,
        source_token="frame",
        source_row_count=5,
        ordered=True,
        unique=True,
    )
    empty = NativeRowSet.from_positions(
        empty_positions,
        source_token="frame",
        source_row_count=5,
        ordered=True,
        unique=True,
    )

    clear_materialization_events()
    reset_d2h_transfer_count()
    with assert_zero_d2h_transfers():
        full_union = identity.union(members)
        identity_intersection = identity.intersection(members)
        empty_union = empty.union(members)
        no_remaining = members.difference(identity)
        still_members = members.difference(empty)
        still_identity = identity.difference(empty)
        empty_intersection = empty.intersection(identity)
        complement = members.complement()
        identity_complement = identity.complement()
        empty_complement = empty.complement()

    assert full_union.positions is identity_positions
    assert full_union.identity is True
    assert identity_intersection.positions is member_positions
    assert identity_intersection.identity is False
    assert empty_union.positions is member_positions
    assert still_identity.positions is identity_positions
    assert still_identity.identity is True
    assert still_members.positions is member_positions
    assert no_remaining.is_device
    assert len(no_remaining) == 0
    assert empty_intersection.positions is empty_positions
    assert cp.asnumpy(complement.positions).tolist() == [0, 2, 4]
    assert len(identity_complement) == 0
    assert empty_complement.identity is True
    assert cp.asnumpy(empty_complement.positions).tolist() == [0, 1, 2, 3, 4]
    assert get_materialization_events(clear=True) == []


def test_native_relation_wraps_existing_pairs_without_host_materialization() -> None:
    left = _ExplodingGetArray([2, 0, 2])
    right = _ExplodingGetArray([1, 1, 0])
    relation_result = RelationIndexResult(left, right)

    assert relation_result.size == 3
    assert len(relation_result) == 3
    assert len(RelationJoinResult(relation_result)) == 3

    relation = NativeRelation.from_relation_index_result(
        relation_result,
        left_token="left",
        right_token="right",
        predicate="intersects",
    )
    rowset = relation.left_rowset()

    assert len(relation) == 3
    assert rowset.positions is left
    assert rowset.source_token == "left"
    assert left.get_called is False
    assert right.get_called is False


def test_relation_index_size_rejects_mismatched_pairs_without_host_materialization() -> None:
    left = _ExplodingGetArray([0, 1, 2])
    right = _ExplodingGetArray([0])
    relation_result = RelationIndexResult(left, right)

    with pytest.raises(ValueError, match="pair length mismatch"):
        _ = relation_result.size

    assert left.get_called is False
    assert right.get_called is False


def test_relation_index_broadcast_right_synthesizes_host_rows_without_right_export() -> None:
    left = np.asarray([2, 0, 1], dtype=np.int32)
    right = _ExplodingGetArray([9])
    relation_result = RelationIndexResult(left, right, broadcast_right_value=7)

    h_left, h_right = relation_result.to_host(dtype=np.intp)

    assert h_left.tolist() == [2, 0, 1]
    assert h_right.tolist() == [7, 7, 7]
    assert relation_result.size == 3
    assert right.get_called is False


def test_point_box_rowset_probe_stays_device_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device rowset probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )
    from vibespatial.spatial.query_box import _query_point_tree_box_row_positions_device

    owned = from_shapely_geometries(
        [
            Point(0, 0),
            Point(1, 1),
            Point(2, 2),
            Point(4, 4),
        ]
    )
    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test device rowset point-box probe",
    )
    reset_d2h_transfer_count()

    with assert_zero_d2h_transfers():
        positions = _query_point_tree_box_row_positions_device(
            owned,
            predicate="intersects",
            box_bounds=np.asarray([0.0, 0.0, 1.5, 1.5], dtype=np.float64),
            force_gpu=True,
        )
        assert positions is not None
        rowset = NativeRowSet.from_positions(
            positions,
            source_row_count=owned.row_count,
            ordered=True,
            unique=True,
        )
        assert rowset.is_device
        assert len(rowset) == 2

    assert cp.asnumpy(positions).tolist() == [0, 1]
    reset_d2h_transfer_count()


def test_device_row_position_materialization_reports_size() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device row-position materialization")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    clear_materialization_events()
    reset_d2h_transfer_count()

    got = _host_row_positions(cp.asarray([0, 2, 4], dtype=cp.int64))

    assert got.tolist() == [0, 2, 4]
    events = get_materialization_events(clear=True)
    assert len(events) == 1
    assert events[0].surface == "vibespatial.api._native_result_core._host_row_positions"
    assert events[0].detail == "rows=3, bytes=24"
    d2h_events = get_d2h_transfer_events(clear=True)
    assert d2h_events[-1].reason.endswith("::row_positions_to_host")
    reset_d2h_transfer_count()


def test_strict_native_disallows_hidden_device_row_position_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device row-position materialization")
    cp = pytest.importorskip("cupy")
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    with pytest.raises(StrictNativeMaterializationError):
        _host_row_positions(cp.asarray([0, 2, 4], dtype=cp.int64))


def test_native_attribute_take_can_gather_device_positions_without_index_preservation() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device attribute gather")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    clear_materialization_events()
    attributes = NativeAttributeTable(
        arrow_table=pa.table(
            {
                "value": pa.array([10, 20, 30, 40], type=pa.int64()),
                "weight": pa.array([1.5, 2.5, 3.5, 4.5], type=pa.float64()),
            }
        )
    )

    taken = attributes.take(
        cp.asarray([0, 2, 3], dtype=cp.int32),
        preserve_index=False,
    )

    assert taken.device_table is not None
    assert len(taken) == 3
    assert tuple(taken.columns) == ("value", "weight")
    assert get_materialization_events(clear=True) == []
    assert taken.device_table.to_arrow().column(0).to_pylist() == [10, 30, 40]


def test_native_frame_device_range_index_take_preserves_labels_without_hot_materialization() -> (
    None
):
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native frame take")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    attributes = NativeAttributeTable(
        arrow_table=pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    )
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(2, 2)]).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test device native frame range-index label preservation",
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=attributes,
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    rowset = NativeRowSet.from_positions(
        cp.asarray([2, 0], dtype=cp.int32),
        source_token=state.lineage_token,
        source_row_count=state.row_count,
        ordered=True,
        unique=True,
    )
    clear_materialization_events()
    reset_d2h_transfer_count()

    with assert_zero_d2h_transfers():
        taken = state.take(rowset, preserve_index=True)

    assert taken.index_plan.kind == "device-labels"
    assert taken.attributes.device_table is not None
    assert get_materialization_events() == []

    chained_rowset = NativeRowSet.from_positions(
        cp.asarray([1, 0], dtype=cp.int32),
        source_token=taken.lineage_token,
        source_row_count=taken.row_count,
        ordered=True,
        unique=True,
    )
    clear_materialization_events()
    reset_d2h_transfer_count()

    with assert_zero_d2h_transfers():
        chained = taken.take(chained_rowset, preserve_index=True)

    assert chained.index_plan.kind == "device-labels"
    assert chained.attributes.device_table is not None
    assert get_materialization_events() == []

    payload = chained.to_native_tabular_result()
    events = get_materialization_events(clear=True)
    d2h_events = get_d2h_transfer_events(clear=True)
    assert payload.attributes.index.tolist() == [0, 1]
    assert payload.index_plan is not None
    assert payload.index_plan.kind == "device-labels"
    assert all(event.operation != "index_plan_to_host" for event in events)
    assert d2h_events == []

    exported = payload.to_geodataframe()
    events = get_materialization_events(clear=True)
    d2h_events = get_d2h_transfer_events(clear=True)
    assert all(event.operation != "index_plan_to_host" for event in events)
    assert d2h_events == []
    assert exported.index.tolist() == [0, 2]
    assert exported["value"].tolist() == [10, 30]
    events = get_materialization_events(clear=True)
    d2h_events = get_d2h_transfer_events(clear=True)
    assert any(event.operation == "index_plan_to_host" for event in events)
    assert any(event.reason.endswith("::index_plan_to_host") for event in d2h_events)
    reset_d2h_transfer_count()


def test_geodataframe_identity_native_boolean_filter_reuses_public_frame(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native geom_type mask")
    from vibespatial.api import geodataframe as geodataframe_module

    owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), box(1.0, 1.0, 2.0, 2.0)],
        residency=Residency.DEVICE,
    )
    payload = NativeTabularResult(
        attributes=NativeAttributeTable(dataframe=pd.DataFrame({"value": [10, 20]})),
        geometry=GeometryNativeResult.from_owned(owned, crs=None),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    frame = payload.to_geodataframe()
    mask = frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    assert bool(getattr(mask.array.rowset, "identity", False))

    def _explode(*_args, **_kwargs):
        raise AssertionError("identity native filter should not rebuild public frame")

    monkeypatch.setattr(
        geodataframe_module,
        "_public_frame_from_native_state",
        _explode,
    )

    filtered = frame[mask]

    assert filtered is not frame
    assert filtered.index.equals(frame.index)
    assert filtered["value"].tolist() == [10, 20]
    assert get_native_state(filtered) is not None


def test_native_tabular_take_host_labels_keeps_numeric_attributes_device() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for host-label device-position take")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    index = pd.Index(["c", "a", "b"], name="parcel")
    attributes = NativeAttributeTable(dataframe=pd.DataFrame({"value": [10, 20, 30]}, index=index))
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(2, 2)]).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native tabular host-label rowset take",
    )
    payload = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=attributes,
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    ).to_native_tabular_result()
    row_positions = cp.asarray([2, 0], dtype=cp.int32)
    clear_materialization_events()
    reset_d2h_transfer_count()

    with assert_zero_d2h_transfers():
        taken = payload.take(row_positions, preserve_index=True)

    events = get_materialization_events(clear=True)
    d2h_events = get_d2h_transfer_events(clear=True)
    assert taken.index_plan is not None
    assert taken.index_plan.kind == "host-labels-take"
    assert taken.attributes.device_table is not None
    assert events == []
    assert d2h_events == []

    exported = taken.to_geodataframe()
    events = get_materialization_events(clear=True)

    assert all(event.operation != "index_plan_take_positions_to_host" for event in events)
    assert exported.index.tolist() == ["b", "c"]
    events = get_materialization_events(clear=True)
    assert exported["value"].tolist() == [30, 10]
    assert any(event.operation == "index_plan_take_positions_to_host" for event in events)
    assert all(event.operation != "row_positions_to_host" for event in events)
    reset_d2h_transfer_count()


def test_native_attribute_table_extracts_admitted_numeric_arrays_without_export() -> None:
    clear_materialization_events()
    pandas_attributes = NativeAttributeTable(
        dataframe=pd.DataFrame(
            {
                "score": [1.0, 2.0, 3.0],
                "flag": [True, False, True],
                "label": ["a", "b", "c"],
            }
        )
    )
    arrow_attributes = NativeAttributeTable(
        arrow_table=pa.table(
            {
                "score": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
                "rows": pa.array([1, 2, 3], type=pa.int64()),
            }
        )
    )
    duplicate_columns = NativeAttributeTable(
        dataframe=pd.DataFrame([[1, 2]], columns=["score", "score"])
    )
    loader_attributes = NativeAttributeTable.from_loader(
        lambda: pd.DataFrame({"score": [1, 2]}),
        index_override=pd.RangeIndex(2),
        columns=("score",),
    )

    pandas_numeric = pandas_attributes.numeric_column_arrays(["score", "flag"])
    arrow_numeric = arrow_attributes.numeric_column_arrays(["score", "rows"])

    assert pandas_numeric is not None
    assert arrow_numeric is not None
    assert pandas_numeric["score"].tolist() == [1.0, 2.0, 3.0]
    assert pandas_numeric["flag"].tolist() == [True, False, True]
    assert arrow_numeric["score"].tolist() == [1.0, 2.0, 3.0]
    assert arrow_numeric["rows"].tolist() == [1, 2, 3]
    assert pandas_attributes.numeric_column_arrays(["label"]) is None
    assert pandas_attributes.numeric_column_arrays(["missing"]) is None
    assert duplicate_columns.numeric_column_arrays(["score"]) is None
    assert loader_attributes.numeric_column_arrays(["score"]) is None
    assert get_materialization_events(clear=True) == []


def test_native_attribute_reset_index_defers_loader_materialization() -> None:
    calls = 0
    index = pd.Index(["a", "b"], name="group")

    def _load() -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return pd.DataFrame({"value": [10, 20]}, index=index)

    attributes = NativeAttributeTable.from_loader(
        _load,
        index_override=index,
        columns=("value",),
    )

    reset, leading_columns, trailing_columns = attributes.reset_index_deferred()

    assert calls == 0
    assert leading_columns == ("group",)
    assert trailing_columns == ("value",)
    assert reset.loader is not None
    assert reset.index.equals(pd.RangeIndex(2))
    assert tuple(reset.columns) == ("group", "value")

    materialized = reset.to_pandas(copy=False)

    assert calls == 1
    assert materialized.to_dict("list") == {"group": ["a", "b"], "value": [10, 20]}


def test_native_attribute_loader_take_device_positions_defers_host_normalization() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for loader-backed device-position take")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    calls = 0

    def _load() -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return pd.DataFrame({"value": [10, 20, 30]}, index=pd.RangeIndex(3))

    attributes = NativeAttributeTable.from_loader(
        _load,
        index_override=pd.RangeIndex(3),
        columns=("value",),
    )
    row_positions = cp.asarray([2, 0], dtype=cp.int64)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        taken = attributes.take(row_positions, preserve_index=False)

    assert calls == 0
    assert taken.loader is not None
    assert taken.index.equals(pd.RangeIndex(2))
    assert tuple(taken.columns) == ("value",)
    assert get_materialization_events(clear=True) == []
    assert get_d2h_transfer_events(clear=True) == []

    materialized = taken.to_pandas(copy=False)
    events = get_materialization_events(clear=True)

    assert calls == 1
    assert materialized.index.equals(pd.RangeIndex(2))
    assert materialized["value"].tolist() == [30, 10]
    assert any(
        event.operation == "row_positions_to_host" and not event.strict_disallowed
        for event in events
    )
    reset_d2h_transfer_count()


def test_native_attribute_promotion_requires_visible_gpu_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibespatial.runtime._runtime as runtime_module

    monkeypatch.setattr(runtime_module, "has_gpu_runtime", lambda: False)
    attributes = NativeAttributeTable(dataframe=pd.DataFrame({"value": [10, 20, 30]}))

    assert attributes.promote_numeric_to_device() is None


def test_native_attribute_dataframe_take_device_positions_promotes_numeric_to_device() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for dataframe-backed device-position take")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    attributes = NativeAttributeTable(
        dataframe=pd.DataFrame({"value": [10, 20, 30]}, index=pd.RangeIndex(3))
    )
    row_positions = cp.asarray([1, 2], dtype=cp.int64)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        taken = attributes.take(row_positions, preserve_index=False)

    assert taken.device_table is not None
    assert taken.index.equals(pd.RangeIndex(2))
    assert tuple(taken.columns) == ("value",)
    assert get_materialization_events(clear=True) == []
    assert get_d2h_transfer_events(clear=True) == []

    materialized = taken.to_pandas(copy=False)
    events = get_materialization_events(clear=True)

    assert materialized.index.equals(pd.RangeIndex(2))
    assert materialized["value"].tolist() == [20, 30]
    assert all(event.operation != "row_positions_to_host" for event in events)
    reset_d2h_transfer_count()


def test_native_attribute_dataframe_take_device_positions_defers_object_columns() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for dataframe-backed device-position take")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    attributes = NativeAttributeTable(
        dataframe=pd.DataFrame({"label": ["a", "b", "c"]}, index=pd.RangeIndex(3))
    )
    row_positions = cp.asarray([1, 2], dtype=cp.int64)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        taken = attributes.take(row_positions, preserve_index=False)

    assert taken.loader is not None
    assert taken.index.equals(pd.RangeIndex(2))
    assert tuple(taken.columns) == ("label",)
    assert get_materialization_events(clear=True) == []
    assert get_d2h_transfer_events(clear=True) == []

    materialized = taken.to_pandas(copy=False)
    events = get_materialization_events(clear=True)

    assert materialized.index.equals(pd.RangeIndex(2))
    assert materialized["label"].tolist() == ["b", "c"]
    assert any(
        event.operation == "row_positions_to_host" and not event.strict_disallowed
        for event in events
    )
    reset_d2h_transfer_count()


def test_native_attribute_reset_index_preserves_numeric_device_payload_without_runtime_d2h() -> (
    None
):
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device reset-index attribute contract")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    source = pa.table({"value": pa.array([10, 20], type=pa.int64())})
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(source),
        index_override=pd.Index([100, 200], name="group"),
        column_override=("value",),
        schema_override=source.schema,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()
    clear_dispatch_events()

    with assert_zero_d2h_transfers():
        reset, leading_columns, trailing_columns = attributes.reset_index_deferred()
        arrays = reset.numeric_column_arrays(("group", "value"))

    assert leading_columns == ("group",)
    assert trailing_columns == ("value",)
    assert reset.device_table is not None
    assert reset.index.equals(pd.RangeIndex(2))
    assert tuple(reset.columns) == ("group", "value")
    assert arrays is not None
    assert cp.asnumpy(arrays["group"]).tolist() == [100, 200]
    assert cp.asnumpy(arrays["value"]).tolist() == [10, 20]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_native_attribute_reset_index_string_device_index_stays_deferred() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device reset-index attribute fallback")
    plc = pytest.importorskip("pylibcudf")

    source = pa.table({"value": pa.array([10, 20], type=pa.int64())})
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(source),
        index_override=pd.Index(["a", "b"], name="group"),
        column_override=("value",),
        schema_override=source.schema,
    )
    clear_materialization_events()

    reset, leading_columns, trailing_columns = attributes.reset_index_deferred()

    assert leading_columns == ("group",)
    assert trailing_columns == ("value",)
    assert reset.loader is not None
    assert tuple(reset.columns) == ("group", "value")
    assert get_materialization_events(clear=True) == []


def test_native_attribute_table_extracts_device_numeric_arrays_without_export() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device attribute dtype contract")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    import pylibcudf as plc

    admitted = pa.table(
        {
            "score": pa.array([1.5, 2.5, 3.5], type=pa.float32()),
            "rows": pa.array([1, 2, 3], type=pa.int32()),
            "flag": pa.array([True, False, True], type=pa.bool_()),
        }
    )
    unsupported = pa.table({"label": pa.array(["a", "b", "c"])})
    nullable = pa.table({"score": pa.array([1.0, None, 3.0], type=pa.float64())})
    admitted_attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(admitted),
        column_override=tuple(admitted.column_names),
        schema_override=admitted.schema,
    )
    unsupported_attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(unsupported),
        column_override=tuple(unsupported.column_names),
        schema_override=unsupported.schema,
    )
    nullable_attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(nullable),
        column_override=tuple(nullable.column_names),
        schema_override=nullable.schema,
    )
    clear_materialization_events()

    columns = admitted_attributes.numeric_column_arrays(["score", "rows", "flag"])

    assert columns is not None
    assert columns["score"].dtype == cp.float32
    assert columns["rows"].dtype == cp.int32
    assert columns["flag"].dtype == cp.bool_
    assert get_materialization_events() == []
    assert unsupported_attributes.numeric_column_arrays(["label"]) is None
    assert nullable_attributes.numeric_column_arrays(["score"]) is None
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(columns["score"]).tolist() == [1.5, 2.5, 3.5]
    assert cp.asnumpy(columns["rows"]).tolist() == [1, 2, 3]
    assert cp.asnumpy(columns["flag"]).tolist() == [True, False, True]


def test_native_attribute_table_numeric_to_pandas_uses_direct_device_columns() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device attribute export contract")
    pytest.importorskip("pylibcudf")
    import pylibcudf as plc

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    source = pa.table(
        {
            "score": pa.array([1.5, 2.5, 3.5], type=pa.float32()),
            "rows": pa.array([1, 2, 3], type=pa.int32()),
            "flag": pa.array([True, False, True], type=pa.bool_()),
        }
    )
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(source),
        index_override=pd.Index([10, 11, 12], name="source"),
        column_override=tuple(source.column_names),
        schema_override=source.schema,
    )
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    clear_materialization_events()

    frame = attributes.to_pandas()
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    materializations = get_materialization_events(clear=True)

    assert frame.index.tolist() == [10, 11, 12]
    assert frame["score"].tolist() == pytest.approx([1.5, 2.5, 3.5])
    assert frame["rows"].tolist() == [1, 2, 3]
    assert frame["flag"].tolist() == [True, False, True]
    assert frame["score"].dtype == np.dtype("float32")
    assert frame["rows"].dtype == np.dtype("int32")
    assert frame["flag"].dtype == np.dtype("bool")
    assert reasons == [
        "device attribute numeric column host export",
        "device attribute numeric column host export",
        "device attribute numeric column host export",
    ]
    assert [event.surface for event in materializations] == [
        "vibespatial.api.NativeAttributeTable.to_pandas",
    ]
    assert materializations[0].operation == "device_numeric_attributes_to_pandas"


def test_native_attribute_table_project_columns_preserves_device_payload_without_runtime_d2h() -> (
    None
):
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device attribute projection")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    import pylibcudf as plc

    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    arrow_table = pa.table(
        {
            "score": pa.array([1.5, 2.5, 3.5], type=pa.float64()),
            "rows": pa.array([1, 2, 3], type=pa.int64()),
            "flag": pa.array([True, False, True], type=pa.bool_()),
        }
    )
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(arrow_table),
        index_override=pd.RangeIndex(3),
        column_override=tuple(arrow_table.column_names),
        schema_override=arrow_table.schema,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        projected = attributes.project_columns(("flag", "score"))
        arrays = projected.numeric_column_arrays(("flag", "score"))

    assert projected.device_table is not None
    assert tuple(projected.columns) == ("flag", "score")
    assert arrays is not None
    assert cp.asnumpy(arrays["flag"]).tolist() == [True, False, True]
    assert cp.asnumpy(arrays["score"]).tolist() == [1.5, 2.5, 3.5]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_native_attribute_table_moves_string_datetime_device_payload_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for non-numeric device attribute movement")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")

    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    source = pa.table(
        {
            "name": pa.array(["a", "b", "c"], type=pa.string()),
            "when": pa.array(
                pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                type=pa.timestamp("us"),
            ),
            "money": pa.array(
                [Decimal("1.25"), Decimal("3.50"), Decimal("2.00")],
                type=pa.decimal128(8, 2),
            ),
            "value": pa.array([1, 2, 3], type=pa.int64()),
        }
    )
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(source),
        index_override=pd.RangeIndex(3),
        column_override=tuple(source.column_names),
        schema_override=source.schema,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        projected = attributes.project_columns(("name", "when"))
        taken = projected.take(cp.asarray([2, 0], dtype=cp.int32), preserve_index=False)

    assert projected.device_table is not None
    assert taken.device_table is not None
    assert taken.row_positions is not None
    assert tuple(taken.columns) == ("name", "when")
    assert taken.numeric_column_arrays(("name", "when")) is None
    assert get_materialization_events(clear=True) == []
    exported = taken.to_arrow(index=False)
    assert exported.column(0).to_pylist() == ["c", "a"]
    assert exported.column(1).to_pylist() == [
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-01"),
    ]
    reset_d2h_transfer_count()


def test_native_attribute_string_assignment_stays_device_backed() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device attribute assignment")
    pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")

    from vibespatial.cuda._runtime import reset_d2h_transfer_count

    owned = from_shapely_geometries(
        [Point(0, 0), Point(1, 1), Point(2, 2)],
        residency=Residency.DEVICE,
    )
    frame = GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": DeviceGeometryArray._from_owned(owned),
        },
        geometry="geometry",
    )
    source = pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    native_payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(source),
            index_override=frame.index,
            column_override=tuple(source.column_names),
            schema_override=source.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=None),
        geometry_name="geometry",
        column_order=("value", "geometry"),
        index_plan=NativeIndexPlan.from_index(frame.index),
    )
    from vibespatial.api._native_state import attach_native_state_from_native_tabular_result

    attach_native_state_from_native_tabular_result(frame, native_payload)

    clear_materialization_events()
    reset_d2h_transfer_count()
    frame["region"] = pd.Series(
        pd.array(["north", "south", "east"], dtype="string"),
        index=frame.index,
        name="region",
    )
    events = get_materialization_events(clear=True)
    state = get_native_state(frame)

    assert state is not None
    assert state.attributes.device_table is not None
    assert tuple(state.attributes.columns) == ("value", "region")
    assert state.attributes.device_column_policies(("region",))["region"].category == (
        "string-movement-only"
    )
    assert not any(event.d2h_transfer for event in events)


def test_native_tabular_public_export_defers_device_attribute_columns() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for lazy device attribute export")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")

    from vibespatial.api._native_public_arrays import (
        NativeAttributeColumnArray,
        NativeIndexLabelsArray,
        NativeNumericExpressionArray,
    )
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [Point(0, 0), Point(1, 1)],
        residency=Residency.DEVICE,
    )
    source = pa.table(
        {
            "value": pa.array([10, 20], type=pa.int64()),
            "region": pa.array(["north", "south"], type=pa.string()),
        }
    )
    index_plan = NativeIndexPlan.from_index(pd.RangeIndex(10, 12, name="site")).take(
        cp.asarray([1, 0], dtype=cp.int64),
        preserve_index=True,
        unique=True,
    )
    native_result = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(source),
            index_override=pd.RangeIndex(2),
            column_override=tuple(source.column_names),
            schema_override=source.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=None),
        geometry_name="geometry",
        column_order=("value", "region", "geometry"),
        index_plan=index_plan,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        exported = native_result.to_geodataframe()
    events = get_materialization_events(clear=True)
    exported_state = get_native_state(exported)

    assert exported_state is not None
    assert exported_state.attributes.device_table is not None
    assert isinstance(getattr(exported.index, "_values", None), NativeIndexLabelsArray)
    assert isinstance(exported["value"].array, NativeNumericExpressionArray)
    assert isinstance(exported["region"].array, NativeAttributeColumnArray)
    assert not any(event.operation == "device_attributes_to_arrow" for event in events)
    assert not any(event.operation == "device_numeric_attributes_to_pandas" for event in events)
    assert not any(event.operation == "index_plan_to_host" for event in events)
    assert not any(event.d2h_transfer for event in events)


def test_native_numeric_public_array_equals_numpy_backed_frame() -> None:
    from vibespatial.api._native_expression import NativeExpression
    from vibespatial.api._native_public_arrays import NativeNumericExpressionArray

    native = pd.DataFrame(
        {
            "value": NativeNumericExpressionArray(
                NativeExpression(
                    operation="test.equals",
                    values=np.asarray([1.0, np.nan, 3.0], dtype=np.float64),
                    source_token=None,
                    source_row_count=3,
                    dtype="float64",
                    precision="source",
                )
            )
        }
    )
    equal = pd.DataFrame({"value": np.asarray([1.0, np.nan, 3.0])})
    different = pd.DataFrame({"value": np.asarray([1.0, np.nan, 4.0])})

    assert native.equals(equal)
    assert not native.equals(different)


def test_native_attribute_arrow_export_uses_physical_extension_dtypes() -> None:
    from vibespatial.api._native_expression import NativeExpression
    from vibespatial.api._native_public_arrays import (
        NativeAttributeColumnArray,
        NativeNumericExpressionArray,
    )

    source_attributes = NativeAttributeTable(
        arrow_table=pa.table({"label": ["north", "south"]}),
        index_override=pd.RangeIndex(2),
    )
    frame = pd.DataFrame(
        {
            "parcel_id": NativeNumericExpressionArray(
                NativeExpression(
                    operation="test.arrow_export",
                    values=np.asarray([10, 20], dtype=np.int64),
                    source_token=None,
                    source_row_count=2,
                    dtype="int64",
                    precision="source",
                )
            ),
            "label": NativeAttributeColumnArray(source_attributes, "label"),
        }
    )

    exported = NativeAttributeTable(dataframe=frame).to_arrow(index=False)

    assert exported.schema.field("parcel_id").type == pa.int64()
    label_type = exported.schema.field("label").type
    assert pa.types.is_string(label_type) or pa.types.is_large_string(label_type)
    assert b"vibespatial_" not in (exported.schema.metadata or {}).get(b"pandas", b"")
    assert exported.to_pandas().to_dict("list") == {
        "parcel_id": [10, 20],
        "label": ["north", "south"],
    }


def test_projected_frame_arrow_export_lowers_mixed_native_index_label_column() -> None:
    from vibespatial.api._native_public_arrays import (
        NativeAttributeColumnArray,
        NativeIndexLabelsArray,
    )

    source_attributes = NativeAttributeTable(
        arrow_table=pa.table({"region": ["north", "south"]}),
        index_override=pd.RangeIndex(2),
    )
    label_plan = NativeIndexPlan.from_index(pd.Index([200, 100])).with_selection(
        np.asarray([1, 0], dtype=np.int64),
        source_token="projected-frame-source",
        source_row_count=2,
        unique=True,
    )
    frames = [
        pd.DataFrame(
            {
                "region": NativeAttributeColumnArray(
                    source_attributes,
                    "region",
                )
            }
        ),
        pd.DataFrame(
            {
                "parcel_id": NativeIndexLabelsArray(label_plan),
                "score": pd.array([1.5, 2.5], dtype="Float64"),
            }
        ),
    ]

    attributes = _native_attribute_table_from_projected_frames(
        frames,
        index_override=pd.RangeIndex(2),
    )
    exported = attributes.to_arrow(index=False)

    assert exported.column_names == ["region", "parcel_id", "score"]
    assert exported.to_pydict() == {
        "region": ["north", "south"],
        "parcel_id": [200, 100],
        "score": [1.5, 2.5],
    }
    assert b"vibespatial_" not in (exported.schema.metadata or {}).get(b"pandas", b"")


def test_projected_frame_arrow_export_lowers_generic_native_index_label_column() -> None:
    from vibespatial.api._native_public_arrays import NativeIndexLabelsArray

    label_plan = NativeIndexPlan.from_index(pd.Index([300, 400])).with_selection(
        np.asarray([0, 1], dtype=np.int64),
        source_token="generic-projected-frame-source",
        source_row_count=2,
        unique=True,
    )
    frame = pd.DataFrame(
        {
            "parcel_id": NativeIndexLabelsArray(label_plan),
            "zone": pd.array(["west", "east"], dtype="string"),
        }
    )

    attributes = _native_attribute_table_from_projected_frames(
        [frame],
        index_override=pd.RangeIndex(2),
    )
    exported = attributes.to_arrow(index=False)

    assert exported.column_names == ["parcel_id", "zone"]
    assert exported.to_pydict() == {
        "parcel_id": [300, 400],
        "zone": ["west", "east"],
    }
    assert b"vibespatial_" not in (exported.schema.metadata or {}).get(b"pandas", b"")


def test_native_attribute_array_vector_indexing_preserves_extension_blocks() -> None:
    from vibespatial.api._native_public_arrays import NativeAttributeColumnArray

    table = NativeAttributeTable(
        arrow_table=pa.table(
            {
                "value": [30, 10, 20],
                "region": ["east", "north", "south"],
            }
        ),
        index_override=pd.RangeIndex(3),
    )
    frame = pd.DataFrame(
        {column: NativeAttributeColumnArray(table, column) for column in table.columns}
    )

    filtered = frame.loc[np.asarray([True, False, True])]
    sorted_frame = filtered.sort_values("value")

    assert isinstance(filtered["value"].array, NativeAttributeColumnArray)
    assert isinstance(filtered["region"].array, NativeAttributeColumnArray)
    assert sorted_frame["value"].tolist() == [20, 30]
    assert sorted_frame["region"].tolist() == ["south", "east"]


def test_native_attribute_array_fillna_detaches_from_source_table() -> None:
    from vibespatial.api._native_public_arrays import NativeAttributeColumnArray

    table = NativeAttributeTable(
        arrow_table=pa.table({"region": ["east", None, "south"]}),
        index_override=pd.RangeIndex(3),
    )
    source = pd.Series(NativeAttributeColumnArray(table, "region"))

    filled = source.fillna("unknown")

    assert filled.tolist() == ["east", "unknown", "south"]
    assert source.iloc[[0, 2]].tolist() == ["east", "south"]
    assert pd.isna(source.iloc[1])


def test_native_attribute_table_reports_device_dtype_policies_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device attribute dtype policies")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")

    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    source = pa.table(
        {
            "score": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            "maybe": pa.array([1.0, None, 3.0], type=pa.float64()),
            "flag": pa.array([True, False, True], type=pa.bool_()),
            "category": pa.array(
                ["a", "b", "a"],
                type=pa.dictionary(pa.int8(), pa.string()),
            ),
            "name": pa.array(["x", "y", "z"], type=pa.string()),
            "when": pa.array(
                pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                type=pa.timestamp("us"),
            ),
            "nothing": pa.nulls(3),
        }
    )
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(source),
        index_override=pd.RangeIndex(3),
        column_override=tuple(source.column_names),
        schema_override=source.schema,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        policies = attributes.device_column_policies()
        numeric = attributes.numeric_column_arrays(("score", "flag"))
        nullable_numeric = attributes.numeric_column_arrays(("maybe",))
        categorical_numeric = attributes.numeric_column_arrays(("category",))
        moved = attributes.project_columns(("maybe", "category", "name", "when")).take(
            cp.asarray([2, 0], dtype=cp.int32), preserve_index=False
        )

    assert policies["score"].category == "all-valid-numeric-bool"
    assert policies["score"].can_compute_numeric is True
    assert policies["score"].can_sort is True
    assert policies["score"].can_distinct is True
    assert policies["score"].can_grouped_nth is True
    assert policies["flag"].category == "all-valid-numeric-bool"
    assert policies["maybe"].category == "nullable-numeric-bool-movement-only"
    assert policies["maybe"].can_compute_numeric is False
    assert policies["maybe"].can_distinct is True
    assert policies["maybe"].can_grouped_nth is True
    assert policies["category"].category == "categorical-movement-only"
    assert policies["category"].can_sort is True
    assert policies["category"].can_distinct is True
    assert policies["category"].can_grouped_nth is True
    assert policies["name"].category == "string-movement-only"
    assert policies["name"].can_sort is True
    assert policies["name"].can_distinct is True
    assert policies["name"].can_grouped_nth is True
    assert policies["when"].category == "datetime-movement-only"
    assert policies["when"].can_sort is True
    assert policies["when"].can_distinct is True
    assert policies["when"].can_grouped_nth is True
    assert policies["nothing"].category == "null-movement-only"
    assert policies["nothing"].can_sort is False
    assert policies["nothing"].can_distinct is False
    assert policies["nothing"].can_grouped_nth is False
    assert numeric is not None
    assert nullable_numeric is None
    assert categorical_numeric is None
    assert moved.device_table is not None
    assert moved.row_positions is not None
    assert tuple(moved.columns) == ("maybe", "category", "name", "when")
    assert get_materialization_events(clear=True) == []

    exported = moved.to_arrow(index=False)
    assert exported.column(0).to_pylist() == [3.0, 1.0]
    assert exported.column(1).to_pylist() == ["a", "a"]
    assert exported.column(2).to_pylist() == ["z", "x"]
    assert exported.column(3).to_pylist() == [
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-01"),
    ]
    reset_d2h_transfer_count()


def test_native_attribute_table_grouped_device_take_columns_preserve_non_numeric_without_runtime_d2h() -> (
    None
):
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device grouped attribute takes")

    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    source = pa.table(
        {
            "name": pa.array(["alpha", "bravo", "charlie", "delta"], type=pa.string()),
            "when": pa.array(
                pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
                type=pa.timestamp("us"),
            ),
            "category": pa.array(
                ["c0", "c1", "c0", "c1"],
                type=pa.dictionary(pa.int8(), pa.string()),
            ),
        }
    )
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(source),
        index_override=pd.RangeIndex(4),
        column_override=tuple(source.column_names),
        schema_override=source.schema,
    )
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([1, 0, 1, 0], dtype=cp.int32),
        group_count=2,
        output_index=pd.Index(["g0", "g1"], name="group"),
    )

    reset_d2h_transfer_count()
    clear_materialization_events()
    with assert_zero_d2h_transfers():
        reduced = attributes.grouped_device_take_columns(
            grouped,
            {"name": "first", "when": "last", "category": "first"},
        )

    assert reduced is not None
    assert reduced.device_table is not None
    assert tuple(reduced.columns) == ("name", "when", "category")
    assert list(reduced.index) == ["g0", "g1"]
    assert get_materialization_events(clear=True) == []

    exported = reduced.to_pandas()
    assert exported["name"].tolist() == ["bravo", "alpha"]
    assert exported["when"].tolist() == [
        pd.Timestamp("2020-01-04"),
        pd.Timestamp("2020-01-03"),
    ]
    assert exported["category"].astype(str).tolist() == ["c1", "c0"]
    reset_d2h_transfer_count()


@pytest.mark.parametrize(
    ("reducer", "expected_labels", "expected_categories", "expected_when"),
    [
        (
            "first",
            ["alpha", "charlie", None, None],
            ["c0", "c2", None, None],
            [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03"), pd.NaT, pd.NaT],
        ),
        (
            "last",
            ["bravo", "charlie", None, None],
            ["c1", "c2", None, None],
            [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"), pd.NaT, pd.NaT],
        ),
    ],
)
def test_native_attribute_table_grouped_device_take_columns_skip_nulls_and_dense_groups(
    reducer: str,
    expected_labels: list[str | None],
    expected_categories: list[str | None],
    expected_when: list[pd.Timestamp],
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for nullable device grouped attribute takes")

    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    source = pa.table(
        {
            "label": pa.array(
                [None, "alpha", "bravo", None, "charlie", None],
                type=pa.string(),
            ),
            "category": pa.array(
                [None, "c0", "c1", None, "c2", None],
                type=pa.dictionary(pa.int8(), pa.string()),
            ),
            "when": pa.array(
                [
                    None,
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2020-01-02"),
                    None,
                    pd.Timestamp("2020-01-03"),
                    None,
                ],
                type=pa.timestamp("us"),
            ),
        }
    )
    physical = NativeAttributeTable(
        device_table=plc.Table.from_arrow(source),
        index_override=pd.RangeIndex(6),
        column_override=tuple(source.column_names),
        schema_override=source.schema,
    )
    attributes = physical.take(
        cp.asarray([5, 1, 2, 0, 4, 3], dtype=cp.int32),
        preserve_index=False,
    )
    output_index = pd.CategoricalIndex(
        pd.Categorical(
            ["g0", "g1", "g2", "g3"],
            categories=["g0", "g1", "g2", "g3"],
        ),
        name="group",
    )
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 0, 0, 1, 1, 2], dtype=cp.int32),
        group_count=4,
        output_index=output_index,
    )

    assert attributes.row_positions is not None
    reset_d2h_transfer_count()
    clear_materialization_events()
    with assert_zero_d2h_transfers():
        reduced = attributes.grouped_device_take_columns(
            grouped,
            {"label": reducer, "category": reducer, "when": reducer},
        )

    assert reduced is not None
    assert reduced.device_table is not None
    assert reduced.row_positions is None
    pd.testing.assert_index_equal(reduced.index, output_index)
    assert reduced.schema_override is not None
    assert reduced.schema_override.field("label").type == pa.string()
    assert reduced.schema_override.field("category").type == source.schema.field(
        "category"
    ).type
    assert reduced.schema_override.field("when").type == pa.timestamp("us")
    assert get_materialization_events(clear=True) == []

    exported = reduced.to_pandas()
    assert exported["label"].tolist()[:2] == expected_labels[:2]
    assert exported["label"].isna().tolist() == [False, False, True, True]
    assert exported["category"].astype(object).tolist()[:2] == expected_categories[:2]
    assert exported["category"].isna().tolist() == [False, False, True, True]
    assert isinstance(exported["category"].dtype, pd.CategoricalDtype)
    assert exported["when"].tolist()[:2] == expected_when[:2]
    assert exported["when"].isna().tolist() == [False, False, True, True]
    assert pd.api.types.is_datetime64_any_dtype(exported["when"].dtype)
    reset_d2h_transfer_count()


def test_native_attribute_table_grouped_device_take_columns_preserve_empty_schema() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for empty device grouped attribute takes")

    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    source = pa.table(
        {
            "label": pa.array([], type=pa.string()),
            "when": pa.array([], type=pa.timestamp("us")),
        }
    )
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(source),
        index_override=pd.RangeIndex(0),
        column_override=tuple(source.column_names),
        schema_override=source.schema,
    )
    output_index = pd.Index([], dtype="object", name="group")
    grouped = NativeGrouped.from_dense_codes(
        cp.empty(0, dtype=cp.int32),
        group_count=0,
        output_index=output_index,
    )

    reset_d2h_transfer_count()
    clear_materialization_events()
    with assert_zero_d2h_transfers():
        reduced = attributes.grouped_device_take_columns(
            grouped,
            {"label": "first", "when": "last"},
        )

    assert reduced is not None
    assert reduced.device_table is not None
    assert len(reduced) == 0
    pd.testing.assert_index_equal(reduced.index, output_index)
    assert reduced.schema_override == source.schema
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_native_attribute_table_grouped_device_take_columns_honors_admission_decline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for grouped attribute admission")

    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import DeviceMemoryAdmission, get_cuda_runtime

    source = pa.table({"label": pa.array([None, "alpha"], type=pa.string())})
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(source),
        index_override=pd.RangeIndex(2),
        column_override=("label",),
        schema_override=source.schema,
    )
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 0], dtype=cp.int32),
        group_count=1,
    )
    runtime = get_cuda_runtime()
    admission_calls = []

    def _decline_grouped_take(*, stage, required_bytes, requested_units=0):
        admission_calls.append((stage, required_bytes, requested_units))
        return DeviceMemoryAdmission(
            stage=stage,
            required_bytes=required_bytes,
            remaining_bytes=0,
            budget_bytes=0,
            admitted=False,
            requested_units=requested_units,
            admitted_units=0,
            bytes_per_unit=required_bytes,
        )

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("declined grouped take must not construct a groupby")

    monkeypatch.setattr(runtime, "admit_device_memory", _decline_grouped_take)
    monkeypatch.setattr(plc.groupby, "GroupBy", _fail_groupby)

    reduced = attributes.grouped_device_take_columns(
        grouped,
        {"label": "first"},
    )

    assert reduced is None
    assert len(admission_calls) == 1
    stage, required_bytes, requested_units = admission_calls[0]
    assert stage == "tabular-grouped-first-last"
    assert required_bytes >= 1 << 20
    assert requested_units == 2


def test_native_attribute_table_concat_preserves_device_payload_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device attribute concat")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    import pylibcudf as plc

    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    left_arrow = pa.table(
        {
            "score": pa.array([1.5, 2.5], type=pa.float64()),
            "flag": pa.array([True, False], type=pa.bool_()),
        }
    )
    right_arrow = pa.table(
        {
            "score": pa.array([3.5], type=pa.float64()),
            "flag": pa.array([True], type=pa.bool_()),
        }
    )
    left = NativeAttributeTable(
        device_table=plc.Table.from_arrow(left_arrow),
        index_override=pd.Index(["a", "b"], name="site"),
        column_override=tuple(left_arrow.column_names),
        schema_override=left_arrow.schema,
    )
    right = NativeAttributeTable(
        device_table=plc.Table.from_arrow(right_arrow),
        index_override=pd.Index(["c"], name="site"),
        column_override=tuple(right_arrow.column_names),
        schema_override=right_arrow.schema,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        concatenated = NativeAttributeTable.concat(
            [left, right],
            ignore_index=True,
            sort=False,
        )
        arrays = concatenated.numeric_column_arrays(("score", "flag"))

    assert concatenated.device_table is not None
    assert concatenated.index.equals(pd.RangeIndex(3))
    assert tuple(concatenated.columns) == ("score", "flag")
    assert arrays is not None
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(arrays["score"]).tolist() == [1.5, 2.5, 3.5]
    assert cp.asnumpy(arrays["flag"]).tolist() == [True, False, True]
    reset_d2h_transfer_count()


def test_all_valid_single_family_owned_device_take_avoids_scalar_d2h_probes() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device geometry take probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)])
    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test all-valid single-family device take",
    )
    reset_d2h_transfer_count()

    with assert_zero_d2h_transfers():
        taken = owned.take(cp.asarray([0, 2, 3], dtype=cp.int32))

    assert taken.row_count == 3
    reset_d2h_transfer_count()


def test_native_relation_derives_semijoin_antijoin_and_counts() -> None:
    relation = NativeRelation(
        left_indices=np.asarray([2, 0, 2], dtype=np.int32),
        right_indices=np.asarray([1, 1, 0], dtype=np.int32),
        left_token="left",
        right_token="right",
        left_row_count=4,
        right_row_count=3,
    )

    assert relation.left_rowset(unique=True).positions.tolist() == [0, 2]
    assert relation.left_semijoin_rowset(order="first").positions.tolist() == [2, 0]
    assert relation.left_antijoin_rowset().positions.tolist() == [1, 3]
    assert relation.left_match_counts().tolist() == [1, 0, 2, 0]
    left_count_expression = relation.left_match_count_expression()
    assert left_count_expression.operation == "relation.left_match_count"
    assert left_count_expression.source_token == "left"
    assert left_count_expression.source_row_count == 4
    assert left_count_expression.values.tolist() == [1, 0, 2, 0]
    assert left_count_expression.greater_than(0).positions.tolist() == [0, 2]
    assert relation.right_semijoin_rowset().positions.tolist() == [0, 1]
    assert relation.right_semijoin_rowset(order="first").positions.tolist() == [1, 0]
    assert relation.right_antijoin_rowset().positions.tolist() == [2]
    assert relation.right_match_counts().tolist() == [1, 2, 0]
    right_count_expression = relation.right_match_count_expression()
    assert right_count_expression.source_token == "right"
    assert right_count_expression.source_row_count == 3
    assert right_count_expression.values.tolist() == [1, 2, 0]

    with pytest.raises(ValueError, match="semijoin order"):
        relation.left_semijoin_rowset(order="labels")


def test_native_relation_full_sorted_semijoin_marks_identity_rowset() -> None:
    relation = NativeRelation(
        left_indices=np.asarray([0, 1, 2, 3], dtype=np.int32),
        right_indices=np.asarray([3, 1, 2, 0], dtype=np.int32),
        left_row_count=4,
        right_row_count=4,
    )

    right_rowset = relation.right_semijoin_rowset()
    assert right_rowset.positions.tolist() == [0, 1, 2, 3]
    assert right_rowset.identity is True

    first_seen = relation.right_semijoin_rowset(order="first")
    assert first_seen.positions.tolist() == [3, 1, 2, 0]
    assert first_seen.identity is False


def test_native_relation_reduces_opposite_side_values_by_grouped_rows() -> None:
    relation = NativeRelation(
        left_indices=np.asarray([0, 0, 2, 2, 2], dtype=np.int32),
        right_indices=np.asarray([1, 2, 0, 1, 2], dtype=np.int32),
        left_token="left",
        right_token="right",
        left_row_count=4,
        right_row_count=3,
    )
    right_values = np.asarray([10, 20, 30], dtype=np.int64)
    left_values = np.asarray([1, 2, 3, 4], dtype=np.int64)

    grouped_left = relation.grouped_by_left()
    left_sum = relation.left_reduce_right_numeric(right_values, "sum")
    right_sum = relation.right_reduce_left_numeric(left_values, "sum")
    left_attributes = relation.left_reduce_right_numeric_columns(
        {
            "score_sum": right_values,
            "score_mean": right_values,
        },
        {
            "score_sum": "sum",
            "score_mean": "mean",
        },
    ).to_native_attribute_table()

    assert grouped_left.group_offsets.tolist() == [0, 2, 5]
    assert grouped_left.group_ids.tolist() == [0, 2]
    assert left_sum.to_pandas(name="score").tolist() == [50, 0, 60, 0]
    assert right_sum.to_pandas(name="score").tolist() == [3, 4, 4]
    assert left_attributes.to_pandas()["score_sum"].tolist() == [50, 0, 60, 0]
    score_mean = left_attributes.to_pandas()["score_mean"].to_numpy()
    assert score_mean[[0, 2]].tolist() == [25.0, 20.0]
    assert np.isnan(score_mean[[1, 3]]).all()

    with pytest.raises(ValueError, match="right_values length"):
        relation.left_reduce_right_numeric([1, 2], "sum")
    with pytest.raises(ValueError, match="right_columns lengths"):
        relation.left_reduce_right_numeric_columns({"score": [1, 2]}, "sum")


def test_native_relation_distance_expression_filters_and_reduces_pairs() -> None:
    relation = NativeRelation(
        left_indices=np.asarray([0, 0, 1, 2], dtype=np.int32),
        right_indices=np.asarray([2, 3, 1, 0], dtype=np.int32),
        left_token="left",
        right_token="right",
        predicate="nearest",
        distances=np.asarray([5.0, 1.0, 3.0, 7.0], dtype=np.float64),
        left_row_count=3,
        right_row_count=4,
        sorted_by_left=True,
    )

    expression = relation.distance_expression()
    rowset = expression.less_equal(3.0)
    filtered = relation.filter_pairs(rowset)
    direct_filtered = relation.filter_by_distance("<=", 3.0)
    left_min = relation.left_reduce_distances("min").to_pandas(name="distance")
    right_min = relation.right_reduce_distances("min").to_pandas(name="distance")

    assert expression.operation == "relation.distance"
    assert expression.source_row_count == len(relation)
    assert rowset.positions.tolist() == [1, 2]
    assert filtered.left_indices.tolist() == [0, 1]
    assert filtered.right_indices.tolist() == [3, 1]
    assert filtered.distances.tolist() == [1.0, 3.0]
    assert filtered.sorted_by_left is True
    assert direct_filtered.left_indices.tolist() == filtered.left_indices.tolist()
    assert left_min.tolist() == [1.0, 3.0, 7.0]
    np.testing.assert_allclose(
        right_min.to_numpy(),
        np.asarray([7.0, 3.0, 5.0, 1.0], dtype=np.float64),
        equal_nan=True,
    )

    with pytest.raises(ValueError, match="requires distances"):
        NativeRelation(
            left_indices=np.asarray([0], dtype=np.int32),
            right_indices=np.asarray([0], dtype=np.int32),
        ).distance_expression()
    with pytest.raises(ValueError, match="distances length"):
        NativeRelation(
            left_indices=np.asarray([0, 1], dtype=np.int32),
            right_indices=np.asarray([0, 1], dtype=np.int32),
            distances=np.asarray([1.0], dtype=np.float64),
        ).distance_expression()
    with pytest.raises(ValueError, match="source_row_count"):
        relation.filter_pairs(
            NativeRowSet.from_positions(
                np.asarray([0], dtype=np.int64),
                source_row_count=len(relation) + 1,
            )
        )


def test_native_grouped_dense_codes_reduce_numeric_with_index_semantics() -> None:
    codes = np.asarray([1, 0, 1, -1, 2, 0], dtype=np.int32)
    values = np.asarray([10, 2, 5, 99, 7, 3], dtype=np.int64)
    output_index = pd.CategoricalIndex(
        pd.Categorical(["a", "b", "c"], categories=["a", "b", "c"]),
        name="group",
    )

    grouped = NativeGrouped.from_dense_codes(
        codes,
        group_count=3,
        output_index=output_index,
        source_token="frame",
    )
    summed = grouped.reduce_numeric(values, "sum").to_pandas(name="value")
    counted = grouped.reduce_numeric(values, "count").to_pandas(name="value")
    meaned = grouped.reduce_numeric(values, "mean").to_pandas(name="value")
    minimized = grouped.reduce_numeric(values, "min").to_pandas(name="value")
    maximized = grouped.reduce_numeric(values, "max").to_pandas(name="value")
    firsted = grouped.reduce_numeric(values, "first").to_pandas(name="value")
    lasted = grouped.reduce_numeric(values, "last").to_pandas(name="value")
    bool_summed = grouped.reduce_numeric(
        np.asarray([True, True, False, True, True, False], dtype=bool),
        "sum",
    ).to_pandas(name="flag")
    bool_any = grouped.reduce_numeric(
        np.asarray([False, False, True, True, False, False], dtype=bool),
        "any",
    ).to_pandas(name="flag")
    bool_all = grouped.reduce_numeric(
        np.asarray([True, False, True, True, True, False], dtype=bool),
        "all",
    ).to_pandas(name="flag")
    reduced_table = grouped.reduce_numeric_columns(
        {
            "value": values,
            "rows": values,
            "has_flag": np.asarray([False, False, True, True, False, False], dtype=bool),
            "all_flag": np.asarray([True, False, True, True, True, False], dtype=bool),
        },
        {
            "value": "sum",
            "rows": "count",
            "has_flag": "any",
            "all_flag": "all",
        },
    )
    reduced_attributes = reduced_table.to_native_attribute_table()

    assert grouped.source_token == "frame"
    assert grouped.sorted_order.tolist() == [1, 5, 0, 2, 4]
    assert grouped.group_offsets.tolist() == [0, 2, 4, 5]
    assert grouped.group_ids.tolist() == [0, 1, 2]
    assert summed.index.equals(output_index)
    assert summed.tolist() == [5, 15, 7]
    assert counted.tolist() == [2, 2, 1]
    assert meaned.tolist() == [2.5, 7.5, 7.0]
    assert minimized.tolist() == [2, 5, 7]
    assert maximized.tolist() == [3, 10, 7]
    assert firsted.tolist() == [2, 10, 7]
    assert lasted.tolist() == [3, 5, 7]
    assert bool_summed.tolist() == [1, 1, 1]
    assert bool_summed.dtype == np.dtype("int64")
    assert bool_any.tolist() == [False, True, False]
    assert bool_any.dtype == np.dtype("bool")
    assert bool_all.tolist() == [False, True, True]
    assert bool_all.dtype == np.dtype("bool")
    assert reduced_table.is_device is False
    assert reduced_attributes.index.equals(output_index)
    assert reduced_attributes.to_pandas()["value"].tolist() == [5, 15, 7]
    assert reduced_attributes.to_pandas()["rows"].tolist() == [2, 2, 1]
    assert reduced_attributes.to_pandas()["has_flag"].tolist() == [False, True, False]
    assert reduced_attributes.to_pandas()["all_flag"].tolist() == [False, True, True]


def test_native_grouped_reduces_expression_with_lineage_checks() -> None:
    grouped = NativeGrouped.from_dense_codes(
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        group_count=2,
        source_token="frame",
    )
    expression = NativeExpression(
        operation="geometry.area",
        values=np.asarray([1.0, 4.0, 9.0, 16.0], dtype=np.float64),
        source_token="frame",
        source_row_count=4,
        dtype="float64",
        precision="fp64",
    )

    reduced = grouped.reduce_expression(expression, "sum").to_pandas(name="area")

    assert reduced.tolist() == [5.0, 25.0]
    with pytest.raises(ValueError, match="source token"):
        grouped.reduce_expression(
            NativeExpression(
                operation="geometry.area",
                values=np.asarray([1.0, 4.0, 9.0, 16.0], dtype=np.float64),
                source_token="other",
                source_row_count=4,
            ),
            "sum",
        )


def test_native_grouped_extrema_reducers_handle_empty_groups() -> None:
    grouped = NativeGrouped.from_dense_codes(
        np.asarray([1, 0, 1, -1], dtype=np.int32),
        group_count=3,
    )
    values = np.asarray([10, 2, 5, 99], dtype=np.int64)
    flags = np.asarray([True, False, True, True], dtype=bool)

    minimum = grouped.reduce_numeric(values, "min").to_pandas(name="value")
    maximum = grouped.reduce_numeric(values, "max").to_pandas(name="value")
    firsted = grouped.reduce_numeric(values, "first").to_pandas(name="value")
    lasted = grouped.reduce_numeric(values, "last").to_pandas(name="value")
    flag_max = grouped.reduce_numeric(flags, "max").to_pandas(name="flag")
    flag_last = grouped.reduce_numeric(flags, "last").to_pandas(name="flag")
    flag_any = grouped.reduce_numeric(flags, "any").to_pandas(name="flag")
    flag_all = grouped.reduce_numeric(flags, "all").to_pandas(name="flag")

    assert minimum.tolist()[:2] == [2.0, 5.0]
    assert maximum.tolist()[:2] == [2.0, 10.0]
    assert firsted.tolist()[:2] == [2.0, 10.0]
    assert lasted.tolist()[:2] == [2.0, 5.0]
    assert flag_max.tolist()[:2] == [0.0, 1.0]
    assert flag_last.tolist()[:2] == [0.0, 1.0]
    assert flag_any.tolist() == [False, True, False]
    assert flag_all.tolist() == [False, True, True]
    assert np.isnan(minimum.iloc[2])
    assert np.isnan(maximum.iloc[2])
    assert np.isnan(firsted.iloc[2])
    assert np.isnan(lasted.iloc[2])
    assert np.isnan(flag_max.iloc[2])
    assert np.isnan(flag_last.iloc[2])
    assert minimum.dtype == np.dtype("float64")
    assert firsted.dtype == np.dtype("float64")
    assert flag_max.dtype == np.dtype("float64")
    assert flag_last.dtype == np.dtype("float64")


def test_native_grouped_take_reducers_preserve_host_extension_dtypes() -> None:
    output_index = pd.CategoricalIndex(
        pd.Categorical(["b", "a", "c"], categories=["a", "b", "c"]),
        name="group",
    )
    grouped = NativeGrouped.from_dense_codes(
        np.asarray([0, 1, 0, 1], dtype=np.int32),
        group_count=3,
        output_index=output_index,
    )
    labels = pd.Series(
        ["left", "right", "last-left", "last-right"],
        dtype=object,
        name="label",
    )
    categories = pd.Series(
        pd.Categorical(
            ["x", "y", "z", "w"],
            categories=["w", "x", "y", "z"],
        ),
        name="category",
    )
    strings = pd.Series(pd.array(["aa", "bb", "cc", "dd"], dtype="string"), name="text")

    first_labels = grouped.reduce_take(labels, "first").to_pandas(name="label")
    last_categories = grouped.reduce_take(categories, "last").to_pandas(name="category")
    reduced = grouped.reduce_take_columns(
        {
            "label": labels,
            "category": categories,
            "text": strings,
        },
        {
            "label": "last",
            "category": "first",
            "text": "first",
        },
    ).to_pandas()

    assert first_labels.index.equals(output_index)
    assert first_labels.tolist() == ["left", "right", None]
    assert last_categories.tolist() == ["z", "w", np.nan]
    assert isinstance(last_categories.dtype, pd.CategoricalDtype)
    assert reduced["label"].tolist() == ["last-left", "last-right", None]
    assert reduced["category"].tolist() == ["x", "y", np.nan]
    assert isinstance(reduced["category"].dtype, pd.CategoricalDtype)
    assert reduced["text"].tolist()[:2] == ["aa", "bb"]
    assert pd.isna(reduced["text"].iloc[2])
    assert pd.api.types.is_string_dtype(reduced["text"].dtype)


def test_native_grouped_take_reducers_match_pandas_skip_null_semantics() -> None:
    keys = pd.Series(
        pd.Categorical(["b", "a", "b", "a", "b"], categories=["a", "b", "c"]),
        name="group",
    )
    output_index = pd.CategoricalIndex(
        pd.Categorical(["b", "a", "c"], categories=["a", "b", "c"]),
        name="group",
    )
    grouped = NativeGrouped.from_dense_codes(
        np.asarray([0, 1, 0, 1, 0], dtype=np.int32),
        group_count=3,
        output_index=output_index,
    )
    cases = [
        pd.Series([None, "y", "z", None, "q"], dtype=object, name="object"),
        pd.Series(
            pd.array([pd.NA, "bb", "cc", pd.NA, "qq"], dtype="string"),
            name="string",
        ),
        pd.Series(
            pd.Categorical([None, "y", "z", None, "q"], categories=["q", "y", "z"]),
            name="category",
        ),
        pd.Series(pd.array([pd.NA, 2, 3, pd.NA, 5], dtype="Int64"), name="nullable_int"),
        pd.Series([np.nan, 2.0, 3.0, np.nan, 5.0], name="float"),
        pd.Series(
            [
                pd.NaT,
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-03"),
                pd.NaT,
                pd.Timestamp("2020-01-05"),
            ],
            name="datetime",
        ),
    ]

    for values in cases:
        for reducer in ("first", "last"):
            actual = grouped.reduce_take(values, reducer).to_pandas(name=values.name)
            expected = values.groupby(keys, observed=False, sort=False).agg(reducer)
            expected.name = values.name
            assert_series_equal(actual, expected)


def test_native_grouped_rejects_non_numeric_column_reducer() -> None:
    grouped = NativeGrouped.from_dense_codes([0, 1, 0], group_count=2)

    with pytest.raises(TypeError, match="numeric or bool"):
        grouped.reduce_numeric_columns({"name": np.asarray(["a", "b", "c"])}, "sum")

    with pytest.raises(ValueError, match="reducers missing columns"):
        grouped.reduce_numeric_columns(
            {"a": np.asarray([1, 2, 3]), "b": np.asarray([4, 5, 6])},
            {"a": "sum"},
        )

    with pytest.raises(ValueError, match="first or last"):
        grouped.reduce_take(pd.Series(["a", None, "b"]), "sum")


def test_native_grouped_dense_codes_reject_invalid_host_codes() -> None:
    with pytest.raises(ValueError, match="exceed group_count"):
        NativeGrouped.from_dense_codes([0, 2], group_count=2)

    with pytest.raises(ValueError, match="-1 for dropped null keys"):
        NativeGrouped.from_dense_codes([0, -2], group_count=2)


def test_native_grouped_from_dense_codes_infers_host_size_metadata() -> None:
    grouped = NativeGrouped.from_dense_codes(
        np.asarray([0, 2, 0, 2, 2], dtype=np.int32),
        group_count=3,
    )

    assert grouped.all_groups_observed is False
    assert grouped.group_size_min == 0
    assert grouped.group_size_max == 3

    observed = NativeGrouped.from_dense_codes(
        np.asarray([0, 1, 0, 1, 1], dtype=np.int32),
        group_count=2,
    )

    assert observed.all_groups_observed is True
    assert observed.group_size_min == 2
    assert observed.group_size_max == 3


def test_native_grouped_from_sorted_offsets_preserves_identity_order() -> None:
    grouped = NativeGrouped.from_sorted_offsets(
        np.asarray([0, 2, 2, 5], dtype=np.int64),
        output_index=pd.Index(["a", "b", "c"], name="group"),
    )

    assert grouped.sorted_order_is_identity is True
    assert grouped.resolved_group_count == 3
    assert grouped.row_count == 5
    assert grouped.all_groups_observed is False
    assert grouped.group_size_min == 0
    assert grouped.group_size_max == 3
    np.testing.assert_array_equal(grouped.group_codes, np.asarray([0, 0, 2, 2, 2]))
    np.testing.assert_array_equal(grouped.group_offsets, np.asarray([0, 2, 5]))
    np.testing.assert_array_equal(grouped.group_ids, np.asarray([0, 2]))
    np.testing.assert_array_equal(grouped.sorted_order, np.arange(5))

    reduced = grouped.reduce_numeric(np.asarray([1, 2, 4, 8, 16]), "sum")
    series = reduced.to_pandas()
    assert series.tolist() == [3, 0, 28]
    assert series.index.tolist() == ["a", "b", "c"]


def test_native_grouped_dense_sorted_offsets_retain_empty_group_capacity() -> None:
    grouped = NativeGrouped.from_dense_sorted_offsets(
        np.asarray([0, 2, 2, 5], dtype=np.int64),
        output_index=pd.Index(["a", "b", "c"], name="group"),
        all_groups_observed=False,
        group_size_min=0,
        group_size_max=3,
    )

    assert grouped.sorted_order_is_identity is True
    assert grouped.resolved_group_count == 3
    assert grouped.row_count == 5
    np.testing.assert_array_equal(grouped.group_codes, [0, 0, 2, 2, 2])
    np.testing.assert_array_equal(grouped.group_offsets, [0, 2, 2, 5])
    np.testing.assert_array_equal(grouped.group_ids, [0, 1, 2])
    np.testing.assert_array_equal(grouped.sorted_order, np.arange(5))

    reduced = grouped.reduce_numeric(np.asarray([1, 2, 4, 8, 16]), "sum")
    assert reduced.to_pandas().tolist() == [3, 0, 28]


def test_native_grouped_all_observed_sorted_offsets_skip_dynamic_compaction() -> None:
    grouped = NativeGrouped.from_sorted_offsets(
        np.asarray([0, 2, 5], dtype=np.int64),
        all_groups_observed=True,
        group_size_min=2,
        group_size_max=3,
    )
    source = Path("src/vibespatial/api/_native_grouped.py").read_text()
    branch_start = source.index("if all_groups_observed is True:")
    branch_end = source.index("if resolved_row_count:", branch_start)
    fixed_branch = source[branch_start:branch_end]

    np.testing.assert_array_equal(grouped.group_offsets, [0, 2, 5])
    np.testing.assert_array_equal(grouped.group_ids, [0, 1])
    assert "xp.nonzero" not in fixed_branch.split("else:", 1)[0]


def test_native_grouped_device_reduce_stays_device_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device grouped reducer probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    codes = cp.asarray([1, 0, 1, -1, 2, 0], dtype=cp.int32)
    values = cp.asarray([10, 2, 5, 99, 7, 3], dtype=cp.int64)
    flags = cp.asarray([True, False, True, True, False, False], dtype=cp.bool_)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        grouped = NativeGrouped.from_dense_codes(codes, group_count=3)
        reduced = grouped.reduce_numeric(values, "sum")
        minimized = grouped.reduce_numeric(values, "min")
        maximized = grouped.reduce_numeric(values, "max")
        firsted = grouped.reduce_numeric(values, "first")
        lasted = grouped.reduce_numeric(values, "last")
        anyed = grouped.reduce_numeric(flags, "any")
        alled = grouped.reduce_numeric(flags, "all")
        reduced_table = grouped.reduce_numeric_columns(
            {
                "value": values,
                "rows": values,
                "any_flag": flags,
                "all_flag": flags,
            },
            {
                "value": "sum",
                "rows": "count",
                "any_flag": "any",
                "all_flag": "all",
            },
        )
        reduced_attributes = reduced_table.to_native_attribute_table()
        reduced_attribute_arrays = reduced_attributes.numeric_column_arrays(
            ["value", "rows", "any_flag", "all_flag"]
        )

    assert grouped.is_device
    assert reduced.is_device
    assert minimized.is_device
    assert maximized.is_device
    assert firsted.is_device
    assert lasted.is_device
    assert anyed.is_device
    assert alled.is_device
    assert reduced_table.is_device
    assert reduced_attributes.device_table is not None
    assert reduced_attribute_arrays is not None
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(reduced.values).tolist() == [5, 15, 7]
    assert cp.asnumpy(minimized.values).tolist() == [2, 5, 7]
    assert cp.asnumpy(maximized.values).tolist() == [3, 10, 7]
    assert cp.asnumpy(firsted.values).tolist() == [2, 10, 7]
    assert cp.asnumpy(lasted.values).tolist() == [3, 5, 7]
    assert cp.asnumpy(anyed.values).tolist() == [False, True, False]
    assert cp.asnumpy(alled.values).tolist() == [False, True, False]
    assert cp.asnumpy(reduced_table.columns["value"].values).tolist() == [5, 15, 7]
    assert cp.asnumpy(reduced_table.columns["rows"].values).tolist() == [2, 2, 1]
    assert cp.asnumpy(reduced_table.columns["any_flag"].values).tolist() == [
        False,
        True,
        False,
    ]
    assert cp.asnumpy(reduced_table.columns["all_flag"].values).tolist() == [
        False,
        True,
        False,
    ]
    assert cp.asnumpy(reduced_attribute_arrays["value"]).tolist() == [5, 15, 7]
    assert cp.asnumpy(reduced_attribute_arrays["rows"]).tolist() == [2, 2, 1]
    assert cp.asnumpy(reduced_attribute_arrays["any_flag"]).tolist() == [
        False,
        True,
        False,
    ]
    assert cp.asnumpy(reduced_attribute_arrays["all_flag"]).tolist() == [
        False,
        True,
        False,
    ]
    reset_d2h_transfer_count()


def test_native_grouped_from_device_sorted_offsets_stays_device_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device grouped-offset probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    offsets = cp.asarray([0, 2, 2, 5], dtype=cp.int64)
    values = cp.asarray([1, 2, 4, 8, 16], dtype=cp.int64)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        grouped = NativeGrouped.from_sorted_offsets(offsets, row_count=values.size)
        reduced = grouped.reduce_numeric(values, "sum")
        dense_grouped = NativeGrouped.from_dense_sorted_offsets(
            offsets,
            row_count=values.size,
            all_groups_observed=False,
            group_size_min=0,
        )
        dense_reduced = dense_grouped.reduce_numeric(values, "sum")

    assert grouped.is_device
    assert grouped.sorted_order_is_identity is True
    assert reduced.is_device
    assert get_materialization_events(clear=True) == []

    np.testing.assert_array_equal(cp.asnumpy(grouped.group_codes), [0, 0, 2, 2, 2])
    np.testing.assert_array_equal(cp.asnumpy(grouped.group_offsets), [0, 2, 5])
    np.testing.assert_array_equal(cp.asnumpy(grouped.group_ids), [0, 2])
    np.testing.assert_array_equal(cp.asnumpy(reduced.values), [3, 0, 28])
    np.testing.assert_array_equal(
        cp.asnumpy(dense_grouped.group_offsets),
        [0, 2, 2, 5],
    )
    np.testing.assert_array_equal(cp.asnumpy(dense_grouped.group_ids), [0, 1, 2])
    np.testing.assert_array_equal(cp.asnumpy(dense_reduced.values), [3, 0, 28])
    reset_d2h_transfer_count()


def test_native_grouped_bounded_all_handles_dense_and_empty_segments() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for bounded grouped reducer probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    offsets = cp.asarray([0, 3, 3, 5, 8], dtype=cp.int64)
    flags = cp.asarray(
        [True, True, True, True, False, True, True, True],
        dtype=cp.bool_,
    )
    grouped = NativeGrouped.from_dense_sorted_offsets(
        offsets,
        row_count=flags.size,
        all_groups_observed=False,
        group_size_min=0,
        group_size_max=3,
    )
    reset_d2h_transfer_count()
    with assert_zero_d2h_transfers():
        reduced = grouped.reduce_boolean_all_bounded(flags)

    assert reduced.is_device
    assert int(reduced.values.nbytes) == grouped.resolved_group_count
    assert cp.asnumpy(reduced.values).tolist() == [True, True, False, True]
    reset_d2h_transfer_count()


def test_native_grouped_bounded_all_drops_null_key_rows() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for bounded grouped null-key probe")
    cp = pytest.importorskip("cupy")

    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, -1, 1], dtype=cp.int32),
        group_count=2,
    )
    reduced = grouped.reduce_boolean_all_bounded(
        cp.asarray([True, False, True], dtype=cp.bool_),
    )

    reduced.validate_error_flag()
    assert cp.asnumpy(reduced.values).tolist() == [True, True]


def test_native_grouped_bounded_all_saturates_one_skewed_segment() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for bounded grouped skew probe")
    cp = pytest.importorskip("cupy")

    row_count = 1_000_000
    offsets = cp.asarray([0, row_count], dtype=cp.int64)
    flags = cp.ones(row_count, dtype=cp.bool_)
    flags[-1] = False

    reduced = NativeGrouped.reduce_boolean_all_from_sorted_offsets(
        flags,
        offsets,
        row_count=row_count,
        group_count=1,
    )

    assert cp.asnumpy(reduced.values).tolist() == [False]


@pytest.mark.parametrize(
    "offsets",
    (
        [-1, 2, 3],
        [0, 3, 2],
        [0, 2, 4],
        [1, 2, 3],
        [0, 1, 2],
    ),
)
def test_native_grouped_bounded_all_guards_malformed_offsets(offsets) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for guarded grouped-offset probe")
    cp = pytest.importorskip("cupy")

    reduced = NativeGrouped.reduce_boolean_all_from_sorted_offsets(
        cp.asarray([True, False, True], dtype=cp.bool_),
        cp.asarray(offsets, dtype=cp.int64),
        row_count=3,
        group_count=2,
        source_token="malformed-offset-regression",
    )

    with pytest.raises(ValueError, match="malformed offsets"):
        reduced.validate_error_flag()


def test_native_grouped_bounded_all_rejects_nonempty_rows_with_zero_groups() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for zero-group coverage probe")
    cp = pytest.importorskip("cupy")

    with pytest.raises(ValueError, match="zero groups require zero source rows"):
        NativeGrouped.reduce_boolean_all_from_sorted_offsets(
            cp.asarray([False], dtype=cp.bool_),
            cp.asarray([0], dtype=cp.int64),
            row_count=1,
            group_count=0,
        )


def test_native_grouped_bounded_all_rejects_int32_group_overflow() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for bounded grouped overflow probe")
    cp = pytest.importorskip("cupy")

    with pytest.raises(OverflowError, match="int32 capacity"):
        NativeGrouped.reduce_boolean_all_from_sorted_offsets(
            cp.empty(0, dtype=cp.bool_),
            cp.asarray([0], dtype=cp.int64),
            row_count=0,
            group_count=int(np.iinfo(np.int32).max) + 1,
        )


def test_native_grouped_bounded_all_records_complete_output_admission() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for bounded grouped admission probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()
    runtime.memory_admission_events(clear=True)
    NativeGrouped.reduce_boolean_all_from_sorted_offsets(
        cp.asarray([True, False, True], dtype=cp.bool_),
        cp.asarray([0, 2, 3], dtype=cp.int64),
        row_count=3,
        group_count=2,
    )
    events = runtime.memory_admission_events(clear=True)

    assert any(
        event.stage == "native_grouped.boolean_all_dense_group_ids"
        and event.required_bytes == 2 * np.dtype(np.int32).itemsize
        and event.admitted
        for event in events
    )
    assert any(
        event.stage == "native_grouped.boolean_all_bounded"
        and event.required_bytes >= 2
        and event.admitted
        for event in events
    )


def test_native_grouped_device_export_reports_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for grouped export D2H accounting")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    codes = cp.asarray([0, 1, 0], dtype=cp.int32)
    values = cp.asarray([1, 2, 3], dtype=cp.int64)
    grouped = NativeGrouped.from_dense_codes(codes, group_count=2)
    reduction = grouped.reduce_numeric(values, "sum")
    reduced_table = grouped.reduce_numeric_columns(
        {"value": values, "rows": values},
        {"value": "sum", "rows": "count"},
    )

    clear_materialization_events()
    reset_d2h_transfer_count()

    series = reduction.to_pandas(name="value")

    assert series.tolist() == [4, 2]
    events = get_materialization_events(clear=True)
    assert len(events) == 1
    assert events[0].operation == "grouped_reduction_to_pandas"
    assert "native_export_target=pandas-series" in events[0].detail
    assert "rows=2" in events[0].detail
    d2h_events = get_d2h_transfer_events(clear=True)
    assert d2h_events[-1].reason.endswith("::grouped_reduction_to_pandas")

    clear_materialization_events()
    reset_d2h_transfer_count()

    frame = reduced_table.to_pandas()

    assert frame["value"].tolist() == [4, 2]
    assert frame["rows"].tolist() == [2, 1]
    events = get_materialization_events(clear=True)
    assert len(events) == 1
    assert events[0].operation == "grouped_attribute_reduction_to_pandas"
    assert "native_export_target=pandas-dataframe" in events[0].detail
    assert "rows=2" in events[0].detail
    d2h_events = get_d2h_transfer_events(clear=True)
    assert (
        sum(
            event.reason.endswith("::grouped_attribute_reduction_to_pandas") for event in d2h_events
        )
        == 2
    )
    reset_d2h_transfer_count()


def test_native_dissolve_attribute_prep_admits_device_bool_any_all_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device dissolve attribute probe")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )
    from vibespatial.overlay.dissolve import _try_prepare_native_grouped_dissolve_attributes

    gdf = GeoDataFrame(
        {
            "zone": [1, 0, 1],
            "has_flag": [False, True, False],
            "all_flag": [True, True, False],
            "geometry": GeoSeries.from_wkt(
                ["POINT (1 1)", "POINT (0 0)", "POINT (2 2)"],
                name="geometry",
            ),
        }
    )
    _attach_owned_native_tabular_state(gdf, attribute_storage="device")
    data = gdf.drop(labels=gdf.geometry.name, axis=1)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        prepared = _try_prepare_native_grouped_dissolve_attributes(
            gdf,
            data,
            by="zone",
            aggfunc={"has_flag": "any", "all_flag": "all"},
            level=None,
            observed=False,
            sort=True,
            dropna=True,
            agg_kwargs={},
        )
        assert prepared is not None
        attributes, row_group_codes = prepared
        arrays = attributes.numeric_column_arrays(("has_flag", "all_flag"))

    assert attributes.device_table is not None
    assert row_group_codes.tolist() == [1, 0, 1]
    assert arrays is not None
    assert cp.asnumpy(arrays["has_flag"]).tolist() == [True, False]
    assert cp.asnumpy(arrays["all_flag"]).tolist() == [True, False]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_native_area_expression_feeds_rowset_and_grouped_consumers_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device area expression probe")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            box(0, 0, 2, 2),
            box(0, 0, 3, 3),
            box(0, 0, 4, 4),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native area expression",
    )
    attribute_arrow = pa.table({"group": pa.array([0, 0, 1, 1], type=pa.int32())})
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(attribute_arrow),
        column_override=tuple(attribute_arrow.column_names),
        schema_override=attribute_arrow.schema,
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=attributes,
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("group", "geometry"),
        )
    )
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 0, 1, 1], dtype=cp.int32),
        group_count=2,
        source_token=state.lineage_token,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        expression = state.geometry_area_expression()
        rowset = expression.greater_than(4.0)
        filtered = state.take(rowset, preserve_index=False)
        reduced = grouped.reduce_expression(expression, "sum")

    assert expression.is_device
    assert rowset.is_device
    assert reduced.is_device
    assert filtered.row_count == 2
    assert filtered.attributes.device_table is not None
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(expression.values).tolist() == [1.0, 4.0, 9.0, 16.0]
    assert cp.asnumpy(rowset.positions).tolist() == [2, 3]
    assert cp.asnumpy(reduced.values).tolist() == [5.0, 25.0]
    reset_d2h_transfer_count()


def test_native_binary_predicate_expression_feeds_rowset_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device binary predicate expression")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )
    from vibespatial.predicates.binary import binary_predicate_expression

    left = from_shapely_geometries(
        [
            box(0, 0, 2, 2),
            box(5, 5, 6, 6),
            box(10, 10, 12, 12),
            box(20, 20, 21, 21),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native binary predicate expression left",
    )
    right = from_shapely_geometries(
        [
            box(1, 1, 3, 3),
            box(7, 7, 8, 8),
            box(11, 11, 13, 13),
            box(30, 30, 31, 31),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native binary predicate expression right",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30, 40], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(left, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        expression = binary_predicate_expression(
            "intersects",
            state.geometry.owned,
            right,
            source_token=state.lineage_token,
        )
        assert expression is not None
        rowset = expression.equal_to(True)
        filtered = state.take(rowset, preserve_index=True)

    assert expression.is_device
    assert rowset.is_device
    assert filtered.row_count == 2
    assert filtered.attributes.device_table is not None
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(expression.values).tolist() == [True, False, True, False]
    assert cp.asnumpy(rowset.positions).tolist() == [0, 2]
    reset_d2h_transfer_count()


def test_native_point_pair_predicate_expression_feeds_rowset_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device point predicate expression")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )
    from vibespatial.predicates.binary import binary_predicate_expression

    left = from_shapely_geometries(
        [
            Point(0, 0),
            Point(1, 1),
            Point(2, 2),
            Point(3, 3),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native point pair predicate expression left",
    )
    right = from_shapely_geometries(
        [
            Point(0, 0),
            Point(5, 5),
            Point(2, 2),
            Point(4, 4),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native point pair predicate expression right",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30, 40], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(left, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    reset_d2h_transfer_count()
    clear_materialization_events()
    clear_dispatch_events()

    with assert_zero_d2h_transfers():
        intersects = binary_predicate_expression(
            "intersects",
            state.geometry.owned,
            right,
            source_token=state.lineage_token,
        )
        disjoint = binary_predicate_expression(
            "disjoint",
            state.geometry.owned,
            right,
            source_token=state.lineage_token,
        )
        assert intersects is not None
        assert disjoint is not None
        rowset = intersects.equal_to(True)
        filtered = state.take(rowset, preserve_index=True)
    events = get_dispatch_events(clear=True)

    assert intersects.is_device
    assert disjoint.is_device
    assert rowset.is_device
    assert filtered.row_count == 2
    assert filtered.attributes.device_table is not None
    assert get_materialization_events(clear=True) == []
    assert any(
        event.implementation == "native_point_pair_expression_gpu"
        and "workload_shape=aligned_pairwise_point_point" in event.detail
        for event in events
    )
    assert cp.asnumpy(intersects.values).tolist() == [True, False, True, False]
    assert cp.asnumpy(disjoint.values).tolist() == [False, True, False, True]
    assert cp.asnumpy(rowset.positions).tolist() == [0, 2]
    reset_d2h_transfer_count()


def test_native_frame_state_predicate_expression_feeds_rowset_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native-frame predicate expression")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    left = from_shapely_geometries(
        [
            Point(0, 0),
            Point(1, 1),
            Point(2, 2),
            Point(3, 3),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native-frame predicate expression left",
    )
    right = from_shapely_geometries(
        [
            Point(0, 0),
            Point(5, 5),
            Point(2, 2),
            Point(4, 4),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native-frame predicate expression right",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30, 40], type=pa.int64())})
    left_state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(left, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    right_arrow = pa.table({"rhs": pa.array([0, 1, 2, 3], type=pa.int64())})
    right_state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(arrow_table=right_arrow),
            geometry=GeometryNativeResult.from_owned(right, crs=None),
            geometry_name="geometry",
            column_order=("rhs", "geometry"),
        )
    )
    reset_d2h_transfer_count()
    clear_materialization_events()
    clear_dispatch_events()

    with assert_zero_d2h_transfers():
        expression = left_state.geometry_predicate_expression("intersects", right_state)
        assert expression is not None
        rowset = expression.equal_to(True)
        filtered = left_state.take(rowset, preserve_index=True)
    events = get_dispatch_events(clear=True)

    assert expression.source_token == left_state.lineage_token
    assert expression.is_device
    assert rowset.is_device
    assert rowset.source_token == left_state.lineage_token
    assert filtered.row_count == 2
    assert filtered.attributes.device_table is not None
    assert get_materialization_events(clear=True) == []
    assert any(
        event.implementation == "native_point_pair_expression_gpu"
        and "operation=geometry_predicate.intersects" in event.detail
        and "carrier=NativeExpression" in event.detail
        for event in events
    )
    assert cp.asnumpy(expression.values).tolist() == [True, False, True, False]
    assert cp.asnumpy(rowset.positions).tolist() == [0, 2]
    reset_d2h_transfer_count()


def test_native_frame_state_multi_predicate_expressions_share_de9im_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native-frame multi-predicate expressions")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    left = from_shapely_geometries(
        [
            box(0, 0, 2, 2),
            box(0, 0, 1, 1),
            box(5, 5, 6, 6),
            box(0, 0, 1, 1),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            box(1, 1, 3, 3),
            box(-1, -1, 2, 2),
            box(7, 7, 8, 8),
            box(0, 0, 1, 1),
        ],
        residency=Residency.DEVICE,
    )
    left_arrow = pa.table({"lhs": pa.array([0, 1, 2, 3], type=pa.int64())})
    right_arrow = pa.table({"rhs": pa.array([0, 1, 2, 3], type=pa.int64())})
    left_state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(arrow_table=left_arrow),
            geometry=GeometryNativeResult.from_owned(left, crs=None),
            geometry_name="geometry",
            column_order=("lhs", "geometry"),
        )
    )
    right_state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(arrow_table=right_arrow),
            geometry=GeometryNativeResult.from_owned(right, crs=None),
            geometry_name="geometry",
            column_order=("rhs", "geometry"),
        )
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        expressions = left_state.geometry_predicate_expressions(
            ("intersects", "covered_by", "disjoint"),
            right_state,
        )
        assert expressions is not None
        hit_rows = expressions["intersects"].equal_to(True)
        inside_rows = expressions["covered_by"].equal_to(True)
        disjoint_rows = expressions["disjoint"].equal_to(True)

    assert set(expressions) == {"intersects", "covered_by", "disjoint"}
    assert all(
        expression.source_token == left_state.lineage_token for expression in expressions.values()
    )
    assert all(expression.is_device for expression in expressions.values())
    assert hit_rows.is_device
    assert inside_rows.is_device
    assert disjoint_rows.is_device
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(expressions["intersects"].values).tolist() == [True, True, False, True]
    assert cp.asnumpy(expressions["covered_by"].values).tolist() == [False, True, False, True]
    assert cp.asnumpy(expressions["disjoint"].values).tolist() == [False, False, True, False]
    assert cp.asnumpy(hit_rows.positions).tolist() == [0, 1, 3]
    assert cp.asnumpy(inside_rows.positions).tolist() == [1, 3]
    assert cp.asnumpy(disjoint_rows.positions).tolist() == [2]
    reset_d2h_transfer_count()


def test_native_point_region_predicate_expression_uses_relation_kernel_without_runtime_d2h() -> (
    None
):
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device binary predicate expression")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )
    from vibespatial.predicates.binary import binary_predicate_expression

    regions = from_shapely_geometries(
        [
            box(0, 0, 2, 2),
            box(0, 0, 2, 2),
            box(0, 0, 2, 2),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native point-region predicate expression regions",
    )
    points = from_shapely_geometries(
        [
            Point(1, 1),
            Point(0, 1),
            Point(3, 3),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native point-region predicate expression points",
    )
    reset_d2h_transfer_count()
    clear_materialization_events()
    clear_dispatch_events()

    with assert_zero_d2h_transfers():
        contains = binary_predicate_expression(
            "contains",
            regions,
            points,
            source_token="point-region-token",
        )
        touches = binary_predicate_expression(
            "touches",
            regions,
            points,
            source_token="point-region-token",
        )
        assert contains is not None
        assert touches is not None
        contained_rows = contains.equal_to(True)
        boundary_rows = touches.equal_to(True)
    events = get_dispatch_events(clear=True)

    assert contains.is_device
    assert touches.is_device
    assert contained_rows.is_device
    assert boundary_rows.is_device
    assert get_materialization_events(clear=True) == []
    assert any(
        event.implementation == "native_point_region_relation_expression_gpu"
        and "workload_shape=aligned_pairwise_point_region" in event.detail
        for event in events
    )
    assert cp.asnumpy(contains.values).tolist() == [True, False, False]
    assert cp.asnumpy(touches.values).tolist() == [False, True, False]
    assert cp.asnumpy(contained_rows.positions).tolist() == [0]
    assert cp.asnumpy(boundary_rows.positions).tolist() == [1]
    reset_d2h_transfer_count()


def test_native_multipoint_predicate_expression_feeds_rowset_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device multipoint predicate expression")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_stats,
        reset_d2h_transfer_count,
    )
    from vibespatial.predicates.binary import binary_predicate_expression

    left = from_shapely_geometries(
        [
            MultiPoint([(0, 0), (0.5, 0.5)]),
            MultiPoint([(2, 2), (3, 3)]),
            MultiPoint([(0, 0), (2, 2)]),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            box(-1, -1, 1, 1),
            box(0, 0, 1, 1),
            Point(2, 2),
        ],
        residency=Residency.DEVICE,
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(left, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    reset_d2h_transfer_count()
    clear_materialization_events()
    clear_dispatch_events()

    with assert_zero_d2h_transfers():
        intersects = binary_predicate_expression(
            "intersects",
            state.geometry.owned,
            right,
            source_token=state.lineage_token,
        )
        covered_by = binary_predicate_expression(
            "covered_by",
            state.geometry.owned,
            right,
            source_token=state.lineage_token,
        )
        assert intersects is not None
        assert covered_by is not None
        hit_rows = intersects.equal_to(True)
        covered_rows = covered_by.equal_to(True)
    events = get_dispatch_events(clear=True)

    assert intersects.is_device
    assert covered_by.is_device
    assert hit_rows.is_device
    assert covered_rows.is_device
    reset_d2h_transfer_count()
    filtered = state.take(hit_rows, preserve_index=True)
    d2h_count, d2h_bytes = get_d2h_transfer_stats()
    assert filtered.row_count == 2
    assert filtered.attributes.device_table is not None
    assert d2h_count <= 1
    assert d2h_bytes <= 8
    assert get_materialization_events(clear=True) == []
    assert any(
        event.implementation == "native_point_family_indexed_expression_gpu"
        and "workload_shape=aligned_pairwise_point_family" in event.detail
        for event in events
    )
    assert cp.asnumpy(intersects.values).tolist() == [True, False, True]
    assert cp.asnumpy(covered_by.values).tolist() == [True, False, False]
    assert cp.asnumpy(hit_rows.positions).tolist() == [0, 2]
    assert cp.asnumpy(covered_rows.positions).tolist() == [0]
    reset_d2h_transfer_count()


def test_native_multi_predicate_expressions_share_de9im_pass_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device multi-predicate expressions")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )
    from vibespatial.predicates.binary import binary_predicate_expressions

    left = from_shapely_geometries(
        [
            box(0, 0, 2, 2),
            box(0, 0, 1, 1),
            box(5, 5, 6, 6),
            box(0, 0, 1, 1),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            box(1, 1, 3, 3),
            box(-1, -1, 2, 2),
            box(7, 7, 8, 8),
            box(0, 0, 1, 1),
        ],
        residency=Residency.DEVICE,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        expressions = binary_predicate_expressions(
            ("intersects", "covered_by", "disjoint"),
            left,
            right,
            source_token="multi-predicate-token",
        )
        assert expressions is not None
        hit_rows = expressions["intersects"].equal_to(True)
        inside_rows = expressions["covered_by"].equal_to(True)
        disjoint_rows = expressions["disjoint"].equal_to(True)

    assert set(expressions) == {"intersects", "covered_by", "disjoint"}
    assert all(expression.is_device for expression in expressions.values())
    assert hit_rows.is_device
    assert inside_rows.is_device
    assert disjoint_rows.is_device
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(expressions["intersects"].values).tolist() == [True, True, False, True]
    assert cp.asnumpy(expressions["covered_by"].values).tolist() == [False, True, False, True]
    assert cp.asnumpy(expressions["disjoint"].values).tolist() == [False, False, True, False]
    assert cp.asnumpy(hit_rows.positions).tolist() == [0, 1, 3]
    assert cp.asnumpy(inside_rows.positions).tolist() == [1, 3]
    assert cp.asnumpy(disjoint_rows.positions).tolist() == [2]
    reset_d2h_transfer_count()


def test_native_multi_predicate_expressions_handle_nulls_without_all_valid_fence() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device multi-predicate expressions")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.predicates.binary import binary_predicate_expressions

    left = from_shapely_geometries(
        [
            box(0, 0, 2, 2),
            None,
            box(10, 10, 11, 11),
            box(0, 0, 1, 1),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            box(1, 1, 3, 3),
            box(0, 0, 1, 1),
            box(20, 20, 21, 21),
            box(0, 0, 1, 1),
        ],
        residency=Residency.DEVICE,
    )
    left._validity = None
    left._tags = None
    left._family_row_offsets = None
    right._validity = None
    right._tags = None
    right._family_row_offsets = None
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    with assert_zero_d2h_transfers():
        expressions = binary_predicate_expressions(
            ("intersects", "covered_by", "disjoint"),
            left,
            right,
            source_token="multi-predicate-null-token",
        )
        assert expressions is not None
        hit_rows = expressions["intersects"].equal_to(True)
        inside_rows = expressions["covered_by"].equal_to(True)
        disjoint_rows = expressions["disjoint"].equal_to(True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert "binary predicate all-valid pair scalar fence" not in runtime_reasons
    assert cp.asnumpy(expressions["intersects"].values).tolist() == [True, False, False, True]
    assert cp.asnumpy(expressions["covered_by"].values).tolist() == [False, False, False, True]
    assert cp.asnumpy(expressions["disjoint"].values).tolist() == [False, False, True, False]
    assert cp.asnumpy(hit_rows.positions).tolist() == [0, 3]
    assert cp.asnumpy(inside_rows.positions).tolist() == [3]
    assert cp.asnumpy(disjoint_rows.positions).tolist() == [2]
    reset_d2h_transfer_count()


def test_native_multi_predicate_device_only_admission_avoids_metadata_export() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device multi-predicate expressions")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.predicates.binary import binary_predicate_expressions

    left_base = from_shapely_geometries(
        [
            box(0, 0, 2, 2),
            box(0, 0, 1, 1),
            box(5, 5, 6, 6),
        ],
        residency=Residency.DEVICE,
    )
    right_base = from_shapely_geometries(
        [
            box(1, 1, 3, 3),
            box(-1, -1, 2, 2),
            box(7, 7, 8, 8),
        ],
        residency=Residency.DEVICE,
    )
    d_rows = cp.asarray([0, 1, 2], dtype=cp.int64)
    left = left_base.device_take(d_rows)
    right = right_base.device_take(d_rows)
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    expressions = binary_predicate_expressions(
        ("intersects", "covered_by"),
        left,
        right,
        source_token="device-only-multi-predicate-token",
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert expressions is not None
    assert set(expressions) == {"intersects", "covered_by"}
    assert "binary predicate all-valid pair scalar fence" not in runtime_reasons
    assert "binary predicate tag-pairs host export" not in runtime_reasons
    assert not any(reason.startswith("owned geometry host metadata") for reason in runtime_reasons)


def test_native_point_region_predicate_expression_handles_nulls_without_all_valid_fence() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device point-region expression")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.predicates.binary import binary_predicate_expression

    points = from_shapely_geometries(
        [
            Point(1, 1),
            None,
            Point(3, 3),
        ],
        residency=Residency.DEVICE,
    )
    regions = from_shapely_geometries(
        [
            box(0, 0, 2, 2),
            box(0, 0, 2, 2),
            box(0, 0, 2, 2),
        ],
        residency=Residency.DEVICE,
    )
    points._validity = None
    points._tags = None
    points._family_row_offsets = None
    regions._validity = None
    regions._tags = None
    regions._family_row_offsets = None
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    with assert_zero_d2h_transfers():
        intersects = binary_predicate_expression(
            "intersects",
            points,
            regions,
            source_token="point-region-null-token",
        )
        disjoint = binary_predicate_expression(
            "disjoint",
            points,
            regions,
            source_token="point-region-null-token",
        )
        assert intersects is not None
        assert disjoint is not None
        hit_rows = intersects.equal_to(True)
        disjoint_rows = disjoint.equal_to(True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert "binary predicate all-valid pair scalar fence" not in runtime_reasons
    assert cp.asnumpy(intersects.values).tolist() == [True, False, False]
    assert cp.asnumpy(disjoint.values).tolist() == [False, False, True]
    assert cp.asnumpy(hit_rows.positions).tolist() == [0]
    assert cp.asnumpy(disjoint_rows.positions).tolist() == [2]
    reset_d2h_transfer_count()


def test_native_validity_expression_feeds_rowset_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device validity expression")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            bowtie,
            None,
            MultiPolygon([box(2, 2, 3, 3)]),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native validity expression",
    )
    attribute_arrow = pa.table({"value": pa.array([1, 2, 3, 4], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        expression = state.geometry_validity_expression()
        rowset = expression.equal_to(True)

    assert expression.operation == "geometry.is_valid"
    assert expression.is_device
    assert rowset.is_device
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(expression.values).tolist() == [True, False, False, True]
    assert cp.asnumpy(rowset.positions).tolist() == [0, 3]

    filtered = state.take(rowset, preserve_index=False)
    assert filtered.row_count == 2
    assert filtered.attributes.device_table is not None
    reset_d2h_transfer_count()


def test_native_length_expression_feeds_rowset_and_grouped_consumers_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device length expression probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [
            LineString([(0, 0), (1, 0)]),
            LineString([(0, 0), (3, 4)]),
            box(0, 0, 1, 1),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native length expression",
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(dataframe=pd.DataFrame({"group": [0, 0, 1]})),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("group", "geometry"),
        )
    )
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 0, 1], dtype=cp.int32),
        group_count=2,
        source_token=state.lineage_token,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        expression = state.geometry_length_expression()
        rowset = expression.greater_than(4.0)
        reduced = grouped.reduce_expression(expression, "sum")

    assert expression.is_device
    assert rowset.is_device
    assert reduced.is_device
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(expression.values).tolist() == [1.0, 5.0, 4.0]
    assert cp.asnumpy(rowset.positions).tolist() == [1]
    assert cp.asnumpy(reduced.values).tolist() == [6.0, 4.0]
    reset_d2h_transfer_count()


def test_native_centroid_component_expressions_feed_consumers_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device centroid expression probe")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            box(2, 0, 4, 2),
            box(4, 4, 8, 8),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native centroid expression",
    )
    owned = owned.device_take(cp.asarray([0, 1, 2], dtype=cp.int64))
    assert owned._validity is None
    assert owned._tags is None
    assert owned._family_row_offsets is None
    attribute_arrow = pa.table({"group": pa.array([0, 0, 1], type=pa.int32())})
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(attribute_arrow),
        column_override=tuple(attribute_arrow.column_names),
        schema_override=attribute_arrow.schema,
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=attributes,
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("group", "geometry"),
        )
    )
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 0, 1], dtype=cp.int32),
        group_count=2,
        source_token=state.lineage_token,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        centroid_x, centroid_y = state.geometry_centroid_expressions()
        x_rowset = centroid_x.greater_than(1.0)
        y_rowset = centroid_y.less_equal(1.0)
        compound_rowset = x_rowset.intersection(y_rowset)
        filtered = state.take(compound_rowset, preserve_index=False)
        x_reduced = grouped.reduce_expression(centroid_x, "sum")
        y_reduced = grouped.reduce_expression(centroid_y, "sum")

    assert centroid_x.operation == "geometry.centroid.x"
    assert centroid_y.operation == "geometry.centroid.y"
    assert centroid_x.is_device
    assert centroid_y.is_device
    assert x_rowset.is_device
    assert y_rowset.is_device
    assert compound_rowset.is_device
    assert filtered.row_count == 1
    assert filtered.attributes.device_table is not None
    assert x_reduced.is_device
    assert y_reduced.is_device
    assert get_materialization_events(clear=True) == []
    np.testing.assert_allclose(cp.asnumpy(centroid_x.values), [0.5, 3.0, 6.0])
    np.testing.assert_allclose(cp.asnumpy(centroid_y.values), [0.5, 1.0, 6.0])
    assert cp.asnumpy(compound_rowset.positions).tolist() == [1]
    np.testing.assert_allclose(cp.asnumpy(x_reduced.values), [3.5, 6.0])
    np.testing.assert_allclose(cp.asnumpy(y_reduced.values), [1.5, 6.0])
    reset_d2h_transfer_count()


def test_native_metric_expressions_compose_rowsets_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device metric expression composition")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            box(0, 0, 2, 2),
            box(0, 0, 3, 3),
            box(0, 0, 4, 4),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native metric expression composition",
    )
    attribute_arrow = pa.table({"group": pa.array([0, 0, 1, 1], type=pa.int32())})
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(attribute_arrow),
        column_override=tuple(attribute_arrow.column_names),
        schema_override=attribute_arrow.schema,
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=attributes,
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("group", "geometry"),
        )
    )
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 0, 1, 1], dtype=cp.int32),
        group_count=2,
        source_token=state.lineage_token,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        area_expression = state.geometry_area_expression()
        length_expression = state.geometry_length_expression()
        area_rowset = area_expression.greater_than(4.0)
        length_rowset = length_expression.less_equal(12.0)
        compound_rowset = area_rowset.intersection(length_rowset)
        filtered = state.take(compound_rowset, preserve_index=False)
        area_reduced = grouped.reduce_expression(area_expression, "sum")
        length_reduced = grouped.reduce_expression(length_expression, "sum")

    assert area_expression.is_device
    assert length_expression.is_device
    assert area_rowset.is_device
    assert length_rowset.is_device
    assert compound_rowset.is_device
    assert filtered.row_count == 1
    assert filtered.attributes.device_table is not None
    assert area_reduced.is_device
    assert length_reduced.is_device
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(area_rowset.positions).tolist() == [2, 3]
    assert cp.asnumpy(length_rowset.positions).tolist() == [0, 1, 2]
    assert cp.asnumpy(compound_rowset.positions).tolist() == [2]
    assert cp.asnumpy(area_reduced.values).tolist() == [5.0, 25.0]
    assert cp.asnumpy(length_reduced.values).tolist() == [12.0, 28.0]
    reset_d2h_transfer_count()


def test_native_distance_expression_composes_rowsets_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device distance expression composition")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    left_owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(0.0, 0.0, 1.0, 1.0),
            box(0.0, 0.0, 1.0, 1.0),
            box(5.0, 0.0, 6.0, 1.0),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native distance expression left",
    )
    right_owned = from_shapely_geometries(
        [
            Point(0.5, 0.5),
            Point(3.0, 0.5),
            box(2.0, 0.0, 3.0, 1.0),
            box(3.0, 0.0, 4.0, 1.0),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native distance expression right",
    )
    attribute_arrow = pa.table({"group": pa.array([0, 0, 1, 1], type=pa.int32())})
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(attribute_arrow),
        column_override=tuple(attribute_arrow.column_names),
        schema_override=attribute_arrow.schema,
    )
    left_state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=attributes,
            geometry=GeometryNativeResult.from_owned(left_owned, crs=None),
            geometry_name="geometry",
            column_order=("group", "geometry"),
        )
    )
    right_state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=attributes,
            geometry=GeometryNativeResult.from_owned(right_owned, crs=None),
            geometry_name="geometry",
            column_order=("group", "geometry"),
        )
    )
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 0, 1, 1], dtype=cp.int32),
        group_count=2,
        source_token=left_state.lineage_token,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        expression = left_state.geometry_distance_expression(right_state)
        close_rows = expression.less_equal(1.0)
        filtered = left_state.take(close_rows, preserve_index=False)
        reduced = grouped.reduce_expression(expression, "sum")

    assert expression.operation == "geometry.distance"
    assert expression.is_device
    assert close_rows.is_device
    assert filtered.row_count == 3
    assert filtered.attributes.device_table is not None
    assert reduced.is_device
    assert get_materialization_events(clear=True) == []
    np.testing.assert_allclose(cp.asnumpy(expression.values), [0.0, 2.0, 1.0, 1.0])
    assert cp.asnumpy(close_rows.positions).tolist() == [0, 2, 3]
    np.testing.assert_allclose(cp.asnumpy(reduced.values), [2.0, 2.0])
    reset_d2h_transfer_count()


@pytest.mark.gpu
def test_public_distance_streams_aligned_device_composition_partitions() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native partitioned distance expression")
    cp = pytest.importorskip("cupy")
    from vibespatial.api._native_public_arrays import NativeNumericExpressionArray
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    def _composition(parts):
        offset = 0
        native_parts = []
        for geometries in parts:
            owned = from_shapely_geometries(
                geometries,
                residency=Residency.DEVICE,
            )
            stop = offset + len(geometries)
            native_parts.append(
                NativeGeometryCompositionPart(
                    GeometryNativeResult.from_owned(owned, crs=None),
                    cp.arange(offset, stop, dtype=cp.int64),
                )
            )
            offset = stop
        return GeometryNativeResult.from_composition(
            NativeGeometryComposition(
                parts=tuple(native_parts),
                row_count=offset,
                crs=None,
                contiguous_row_partitions=True,
                trusted_singular_rows=True,
                trusted_family_domain=(GeometryFamily.POINT,),
            ),
            crs=None,
        )

    left_geometry = _composition(
        ([Point(0, 0), Point(1, 0)], [Point(2, 0), Point(3, 0)])
    )
    right_geometry = _composition(
        ([Point(0, 0), Point(1, 1)], [Point(2, 2), Point(3, 3)])
    )
    attributes = NativeAttributeTable.from_value(pd.DataFrame({"row": range(4)}))
    left = NativeTabularResult(
        attributes=attributes,
        geometry=left_geometry,
        geometry_name="geometry",
        column_order=("row", "geometry"),
    ).to_geodataframe()
    right = NativeTabularResult(
        attributes=attributes,
        geometry=right_geometry,
        geometry_name="geometry",
        column_order=("row", "geometry"),
    ).to_geodataframe()
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        distances = left.geometry.distance(right.geometry, align=False)
        scalar_distances = left.geometry.distance(Point(0, 0))

    assert isinstance(distances.array, NativeNumericExpressionArray)
    assert distances.array.expression.is_device
    assert distances.array.expression.precision in {"fp32", "fp64"}
    assert isinstance(scalar_distances.array, NativeNumericExpressionArray)
    assert scalar_distances.array.expression.is_device
    assert get_materialization_events(clear=True) == []
    np.testing.assert_allclose(
        cp.asnumpy(distances.array.expression.values),
        [0.0, 1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        cp.asnumpy(scalar_distances.array.expression.values),
        [0.0, 1.0, 2.0, 3.0],
    )
    reset_d2h_transfer_count()


def test_native_metric_expression_columns_compose_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native metric expression columns")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            box(0, 0, 2, 2),
            box(0, 0, 3, 3),
            box(0, 0, 4, 4),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native metric expression column composition",
    )
    attribute_arrow = pa.table({"group": pa.array([0, 0, 1, 1], type=pa.int32())})
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(attribute_arrow),
        column_override=tuple(attribute_arrow.column_names),
        schema_override=attribute_arrow.schema,
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=attributes,
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("group", "geometry"),
        )
    )
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 0, 1, 1], dtype=cp.int32),
        group_count=2,
        source_token=state.lineage_token,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        area_expression = state.geometry_area_expression()
        length_expression = state.geometry_length_expression()
        augmented = state.assign_expression_columns(
            {
                "area": area_expression,
                "length": length_expression,
            },
        )
        assert augmented is not None
        area_column = augmented.attribute_expression("area")
        length_column = augmented.attribute_expression("length")
        assert area_column is not None
        assert length_column is not None
        rowset = area_column.greater_than(4.0).intersection(
            length_column.less_equal(12.0),
        )
        filtered = augmented.take(rowset, preserve_index=False)
        reduced = grouped.reduce_numeric_columns(
            {
                "area": area_column.values,
                "length": length_column.values,
            },
            "sum",
        )
        reduced_attributes = reduced.to_native_attribute_table()
        reduced_arrays = reduced_attributes.numeric_column_arrays(["area", "length"])

    assert augmented.attributes.device_table is not None
    assert augmented.column_order == ("group", "geometry", "area", "length")
    assert area_column.is_device
    assert length_column.is_device
    assert rowset.is_device
    assert filtered.row_count == 1
    assert filtered.attributes.device_table is not None
    assert reduced_attributes.device_table is not None
    assert reduced_arrays is not None
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(rowset.positions).tolist() == [2]
    assert cp.asnumpy(reduced_arrays["area"]).tolist() == [5.0, 25.0]
    assert cp.asnumpy(reduced_arrays["length"]).tolist() == [12.0, 28.0]
    reset_d2h_transfer_count()


def test_native_tabular_result_to_native_frame_state_preserves_private_carriers() -> None:
    owned = from_shapely_geometries([box(0, 0, 1, 1), box(0, 0, 2, 2)])
    result = NativeTabularResult(
        attributes=NativeAttributeTable(dataframe=pd.DataFrame({"value": [1, 2]})),
        geometry=GeometryNativeResult.from_owned(owned, crs=None),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    clear_materialization_events()

    state = result.to_native_frame_state()

    assert isinstance(state, NativeFrameState)
    assert state.geometry.owned is owned
    assert state.column_order == ("value", "geometry")
    assert get_materialization_events(clear=True) == []


def test_clip_native_tabular_reuses_arrow_source_attributes() -> None:
    from vibespatial.api.tools.clip import evaluate_geopandas_clip_native

    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)", "POINT (5 5)"],
                name="geometry",
            ),
        }
    )
    _attach_native_tabular_state(gdf, attribute_storage="arrow")
    clear_materialization_events()

    result = evaluate_geopandas_clip_native(
        gdf,
        [-0.5, -0.5, 1.5, 1.5],
        keep_geom_type=False,
    )

    assert result.attributes.arrow_table is not None or result.attributes.device_table is not None
    assert result.attributes.to_pandas()["value"].tolist() == [1, 2]


def test_clip_native_tabular_reuses_device_source_attributes_without_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device clip attribute projection")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.api.tools.clip import evaluate_geopandas_clip_native
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)", "POINT (5 5)"],
                name="geometry",
            ),
        }
    )
    _attach_native_tabular_state(gdf, attribute_storage="device")
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        result = evaluate_geopandas_clip_native(
            gdf,
            [-0.5, -0.5, 1.5, 1.5],
            keep_geom_type=False,
        )
        arrays = result.attributes.numeric_column_arrays(("value",))

    assert result.attributes.device_table is not None
    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [1, 2]
    reset_d2h_transfer_count()


def test_pairwise_constructive_native_result_carries_source_row_provenance() -> None:
    left = GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        },
        geometry="geometry",
    )
    right = GeoDataFrame(
        {
            "zone": [1, 2],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 3, 1)],
        },
        geometry="geometry",
    )
    geometry = GeometryNativeResult.from_values(
        [box(2, 0, 3, 1), box(0, 0, 1, 1)],
        crs=None,
        name="geometry",
    )
    relation = RelationIndexResult(
        np.asarray([2, 0], dtype=np.int32),
        np.asarray([1, 0], dtype=np.int32),
    )

    result = _pairwise_constructive_to_native_tabular_result(
        geometry=geometry,
        relation=relation,
        keep_geom_type_applied=True,
        left_df=left,
        right_df=right,
    )
    state = result.to_native_frame_state()
    rowset = NativeRowSet.from_positions(
        np.asarray([1], dtype=np.int64),
        source_token=state.lineage_token,
        source_row_count=state.row_count,
    )
    taken = state.take(rowset, preserve_index=False)

    assert isinstance(result.provenance, NativeGeometryProvenance)
    assert result.provenance.operation == "pairwise_constructive"
    assert result.provenance.keep_geom_type_applied is True
    assert result.provenance.left_rows.tolist() == [2, 0]
    assert result.provenance.right_rows.tolist() == [1, 0]
    assert isinstance(taken.provenance, NativeGeometryProvenance)
    assert taken.provenance.left_rows.tolist() == [0]
    assert taken.provenance.right_rows.tolist() == [0]


def test_pairwise_constructive_native_result_reuses_arrow_source_attributes() -> None:
    left = GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        },
        geometry="geometry",
    )
    right = GeoDataFrame(
        {
            "zone": [1, 2],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 3, 1)],
        },
        geometry="geometry",
    )
    _attach_native_tabular_state(left, attribute_storage="arrow")
    _attach_native_tabular_state(right, attribute_storage="arrow")
    geometry = GeometryNativeResult.from_values(
        [box(2, 0, 3, 1), box(0, 0, 1, 1)],
        crs=None,
        name="geometry",
    )

    result = _pairwise_constructive_to_native_tabular_result(
        geometry=geometry,
        relation=RelationIndexResult(
            np.asarray([2, 0], dtype=np.int32),
            np.asarray([1, 0], dtype=np.int32),
        ),
        keep_geom_type_applied=False,
        left_df=left,
        right_df=right,
    )

    assert result.attributes.arrow_table is not None
    assert result.column_order == ("value", "zone", "geometry")
    assert result.attributes.to_pandas().to_dict("list") == {
        "value": [30, 10],
        "zone": [2, 1],
    }


def test_pairwise_constructive_native_result_reuses_device_source_attributes_without_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device pairwise constructive attributes")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    left = GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        },
        geometry="geometry",
    )
    right = GeoDataFrame(
        {
            "zone": [1, 2],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 3, 1)],
        },
        geometry="geometry",
    )
    _attach_native_tabular_state(left, attribute_storage="device")
    _attach_native_tabular_state(right, attribute_storage="device")
    geometry = GeometryNativeResult.from_values(
        [box(2, 0, 3, 1), box(0, 0, 1, 1)],
        crs=None,
        name="geometry",
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        result = _pairwise_constructive_to_native_tabular_result(
            geometry=geometry,
            relation=RelationIndexResult(
                cp.asarray([2, 0], dtype=cp.int32),
                cp.asarray([1, 0], dtype=cp.int32),
            ),
            keep_geom_type_applied=False,
            left_df=left,
            right_df=right,
        )
        arrays = result.attributes.numeric_column_arrays(("value", "zone"))

    assert result.attributes.parts is not None
    assert all(part.device_table is not None for part in result.attributes.parts)
    assert result.column_order == ("value", "zone", "geometry")
    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [30, 10]
    assert cp.asnumpy(arrays["zone"]).tolist() == [2, 1]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_pairwise_constructive_promotes_numeric_pandas_attrs_for_device_gather() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for mixed native constructive attributes")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    left = GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        },
        geometry="geometry",
    )
    right = GeoDataFrame(
        {
            "zone": [1, 2],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 3, 1)],
        },
        geometry="geometry",
    )
    _attach_native_tabular_state(left, attribute_storage="device")
    _attach_native_tabular_state(right, attribute_storage="pandas")
    geometry = GeometryNativeResult.from_values(
        [box(2, 0, 3, 1), box(0, 0, 1, 1)],
        crs=None,
        name="geometry",
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        result = _pairwise_constructive_to_native_tabular_result(
            geometry=geometry,
            relation=RelationIndexResult(
                cp.asarray([2, 0], dtype=cp.int32),
                cp.asarray([1, 0], dtype=cp.int32),
            ),
            keep_geom_type_applied=False,
            left_df=left,
            right_df=right,
        )
        arrays = result.attributes.numeric_column_arrays(("value", "zone"))

    assert result.attributes.parts is not None
    assert all(part.device_table is not None for part in result.attributes.parts)
    assert result.column_order == ("value", "zone", "geometry")
    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [30, 10]
    assert cp.asnumpy(arrays["zone"]).tolist() == [2, 1]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_relation_constructive_native_result_uses_device_relation_pairs() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for relation constructive native result")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_profile,
        reset_d2h_transfer_count,
    )

    left_owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(1.0, 0.0, 2.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test relation constructive left",
    )
    right_owned = from_shapely_geometries(
        [
            box(0.25, 0.25, 1.25, 1.25),
            box(2.25, 0.25, 3.25, 1.25),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test relation constructive right",
    )
    left_attribute_arrow = pa.table(
        {
            "value": pa.array([10, 20, 30], type=pa.int64()),
            "shared": pa.array([1, 2, 3], type=pa.int32()),
        }
    )
    right_attribute_arrow = pa.table(
        {
            "zone": pa.array([100, 200], type=pa.int64()),
            "shared": pa.array([4, 5], type=pa.int32()),
        }
    )
    left_state = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(left_attribute_arrow),
            column_override=tuple(left_attribute_arrow.column_names),
            schema_override=left_attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(left_owned, crs=None),
        geometry_name="geometry",
        column_order=("value", "shared", "geometry"),
    ).to_native_frame_state()
    right_state = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(right_attribute_arrow),
            column_override=tuple(right_attribute_arrow.column_names),
            schema_override=right_attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(right_owned, crs=None),
        geometry_name="geometry",
        column_order=("zone", "shared", "geometry"),
    ).to_native_frame_state()
    relation = NativeRelation(
        left_indices=cp.asarray([0, 2], dtype=cp.int32),
        right_indices=cp.asarray([0, 1], dtype=cp.int32),
        left_token=left_state.lineage_token,
        right_token=right_state.lineage_token,
        predicate="intersects",
        left_row_count=left_state.row_count,
        right_row_count=right_state.row_count,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    result = _relation_constructive_to_native_tabular_result(
        op="intersection",
        relation=relation,
        left_state=left_state,
        right_state=right_state,
        dispatch_mode=ExecutionMode.GPU,
    )
    assert result is not None
    state = result.to_native_frame_state()
    area = state.geometry_area_expression()
    rowset = area.greater_than(0.5)
    attribute_arrays = state.attributes.numeric_column_arrays(
        ("value", "shared_1", "zone", "shared_2")
    )
    transfer_count, transfer_bytes, _transfer_seconds = get_d2h_transfer_profile()

    assert result.provenance.operation == "relation_intersection"
    assert result.provenance.is_device
    assert result.provenance.source_tokens == (
        left_state.lineage_token,
        right_state.lineage_token,
    )
    assert cp.asnumpy(result.provenance.left_rows).tolist() == [0, 2]
    assert cp.asnumpy(result.provenance.right_rows).tolist() == [0, 1]
    assert state.column_order == ("value", "shared_1", "zone", "shared_2", "geometry")
    assert state.attributes.parts is not None
    assert all(part.device_table is not None for part in state.attributes.parts)
    assert attribute_arrays is not None
    assert cp.asnumpy(attribute_arrays["value"]).tolist() == [10, 30]
    assert cp.asnumpy(attribute_arrays["shared_1"]).tolist() == [1, 3]
    assert cp.asnumpy(attribute_arrays["zone"]).tolist() == [100, 200]
    assert cp.asnumpy(attribute_arrays["shared_2"]).tolist() == [4, 5]
    assert area.is_device
    assert rowset.is_device
    np.testing.assert_allclose(cp.asnumpy(area.values), [0.5625, 0.5625])
    assert cp.asnumpy(rowset.positions).tolist() == [0, 1]
    # Polygon constructive still carries small scalar output-size/admissibility
    # fences; relation pair formatting must not add materialization.
    assert transfer_count <= 8
    assert transfer_bytes <= 128
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_relation_selection_constructive_keeps_pair_capacity_device_resident() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for relation selection constructive result")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_profile,
        reset_d2h_transfer_count,
    )

    left_owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(1.0, 0.0, 2.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
        ],
        residency=Residency.DEVICE,
    )
    right_owned = from_shapely_geometries(
        [
            box(0.25, 0.25, 1.25, 1.25),
            box(2.25, 0.25, 3.25, 1.25),
        ],
        residency=Residency.DEVICE,
    )
    left_state = NativeTabularResult(
        attributes=pd.DataFrame({"value": [10, 20, 30]}),
        geometry=GeometryNativeResult.from_owned(left_owned, crs=None),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    ).to_native_frame_state()
    right_state = NativeTabularResult(
        attributes=pd.DataFrame({"zone": [100, 200]}),
        geometry=GeometryNativeResult.from_owned(right_owned, crs=None),
        geometry_name="geometry",
        column_order=("zone", "geometry"),
    ).to_native_frame_state()
    relation = NativeRelation(
        left_indices=cp.asarray([0, 1, 2], dtype=cp.int32),
        right_indices=cp.asarray([0, 0, 1], dtype=cp.int32),
        left_token=left_state.lineage_token,
        right_token=right_state.lineage_token,
        left_row_count=left_state.row_count,
        right_row_count=right_state.row_count,
    )
    relation_selection = relation.filter_pairs_selection(
        NativeDeviceSelection.from_mask(cp.asarray([True, False, True])),
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    result = _relation_selection_constructive_to_native_tabular_result(
        op="intersection",
        relation_selection=relation_selection,
        left_state=left_state,
        right_state=right_state,
    )
    transfer_count, transfer_bytes, _transfer_seconds = get_d2h_transfer_profile()

    assert isinstance(result, NativeTabularSelection)
    assert result.capacity == 3
    assert cp.asnumpy(result.logical_count).tolist() == [2]
    assert cp.asnumpy(result.selection.positions[:2]).tolist() == [0, 1]
    assert cp.asnumpy(result.provenance.left_rows).tolist() == [0, 2, 0]
    assert cp.asnumpy(result.provenance.right_rows).tolist() == [0, 1, 0]
    assert transfer_count <= 8
    assert transfer_bytes <= 128
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_relation_constructive_native_result_validates_source_lineage() -> None:
    owned = from_shapely_geometries([box(0, 0, 1, 1)])
    state = NativeTabularResult(
        attributes=NativeAttributeTable(dataframe=pd.DataFrame(index=pd.RangeIndex(1))),
        geometry=GeometryNativeResult.from_owned(owned, crs=None),
        geometry_name="geometry",
        column_order=("geometry",),
    ).to_native_frame_state()
    relation = NativeRelation(
        left_indices=np.asarray([0], dtype=np.int32),
        right_indices=np.asarray([0], dtype=np.int32),
        left_token="stale-left",
        right_token=state.lineage_token,
        left_row_count=state.row_count,
        right_row_count=state.row_count,
    )

    with pytest.raises(ValueError, match="left source token"):
        _relation_constructive_to_native_tabular_result(
            op="intersection",
            relation=relation,
            left_state=state,
            right_state=state,
            dispatch_mode=ExecutionMode.CPU,
        )


def test_constructive_native_result_feeds_expression_consumers_with_bounded_fences() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for constructive NativeFrameState consumers")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.constructive.binary_constructive import binary_constructive_owned
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_profile,
        reset_d2h_transfer_count,
    )

    left = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(1.0, 0.0, 2.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
            box(3.0, 0.0, 4.0, 1.0),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test constructive native output left",
    )
    right = from_shapely_geometries(
        [
            box(0.25, 0.25, 1.25, 1.25),
            box(1.25, 0.25, 2.25, 1.25),
            box(2.25, 0.25, 3.25, 1.25),
            box(3.25, 0.25, 4.25, 1.25),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test constructive native output right",
    )
    constructed = binary_constructive_owned(
        "intersection",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )
    seed_homogeneous_host_metadata(constructed, GeometryFamily.POLYGON)
    seed_all_validity_cache(constructed)
    attribute_arrow = pa.table({"group": pa.array([0, 0, 1, 1], type=pa.int32())})
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(attribute_arrow),
        column_override=tuple(attribute_arrow.column_names),
        schema_override=attribute_arrow.schema,
    )
    state = NativeTabularResult(
        attributes=attributes,
        geometry=GeometryNativeResult.from_owned(constructed, crs=None),
        geometry_name="geometry",
        column_order=("group", "geometry"),
        provenance=NativeGeometryProvenance(
            operation="pairwise_constructive",
            row_count=int(constructed.row_count),
            left_rows=cp.arange(4, dtype=cp.int32),
            right_rows=cp.arange(4, dtype=cp.int32),
        ),
    ).to_native_frame_state()
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 0, 1, 1], dtype=cp.int32),
        group_count=2,
        source_token=state.lineage_token,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    area_expression = state.geometry_area_expression()
    rowset = area_expression.greater_than(0.5)
    filtered = state.take(rowset, preserve_index=False)
    reduced = grouped.reduce_expression(area_expression, "sum")
    transfer_count, transfer_bytes, _transfer_seconds = get_d2h_transfer_profile()

    assert constructed.residency is Residency.DEVICE
    assert area_expression.is_device
    assert rowset.is_device
    assert filtered.row_count == 4
    assert filtered.attributes.device_table is not None
    assert isinstance(filtered.provenance, NativeGeometryProvenance)
    assert filtered.provenance.is_device
    assert reduced.is_device
    assert transfer_count <= 2
    assert transfer_bytes <= 16
    assert get_materialization_events(clear=True) == []
    np.testing.assert_allclose(cp.asnumpy(area_expression.values), [0.5625] * 4)
    assert cp.asnumpy(rowset.positions).tolist() == [0, 1, 2, 3]
    assert cp.asnumpy(filtered.provenance.left_rows).tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(cp.asnumpy(reduced.values), [1.125, 1.125])
    reset_d2h_transfer_count()


def test_native_geometry_metadata_host_contract_reuses_owned_bounds() -> None:
    owned = from_shapely_geometries([Point(0, 0), box(0, 0, 2, 3)])

    metadata = NativeGeometryMetadata.from_owned(
        owned,
        source_token="frame",
        prefer_device=False,
    )

    assert metadata.source_token == "frame"
    assert metadata.row_count == 2
    assert metadata.is_device is False
    metadata.validate_row_count(2)
    np.testing.assert_allclose(
        metadata.bounds_to_host(strict_disallowed=False),
        np.asarray([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 3.0]]),
    )
    assert metadata.total_bounds == (0.0, 0.0, 2.0, 3.0)


def test_flat_spatial_index_exports_native_carriers_without_lazy_materialization() -> None:
    owned = from_shapely_geometries([Point(2, 2), Point(0, 0), Point(1, 1)])
    flat_index = build_flat_spatial_index(
        owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            reason="unit test native spatial index export",
        ),
    )

    metadata = flat_index.geometry_metadata(source_token="frame")
    native_index = flat_index.to_native_spatial_index(source_token="frame")
    prepared_marker = object()
    native_index.point_partition_cache["prepared-marker"] = prepared_marker

    assert isinstance(metadata, NativeGeometryMetadata)
    assert isinstance(native_index, NativeSpatialIndex)
    assert metadata.bounds is flat_index._host_bounds
    assert native_index.metadata is not None
    assert native_index.metadata.bounds is flat_index._host_bounds
    assert native_index.order is flat_index._host_order
    assert native_index.morton_keys is flat_index._host_morton_keys
    assert native_index.source_token == "frame"
    assert native_index.to_flat_index() is flat_index
    assert native_index.point_partition_cache["prepared-marker"] is prepared_marker
    assert native_index.to_flat_index().point_grid is None
    reconstructed = replace(native_index, _flat_index=None).to_flat_index()
    assert reconstructed.point_grid is None
    native_index.validate_row_count(3)


def test_flat_spatial_index_publishes_one_native_owner_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock
    from time import sleep

    owned = from_shapely_geometries([Point(0, 0), Point(1, 1)])
    flat_index = build_flat_spatial_index(
        owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            reason="unit test concurrent native spatial index owner",
        ),
    )
    original_factory = NativeSpatialIndex.from_flat_index
    created = 0
    created_lock = Lock()

    def _slow_factory(cls, source_index, *, source_token=None):
        nonlocal created
        with created_lock:
            created += 1
        sleep(0.05)
        return original_factory(source_index, source_token=source_token)

    monkeypatch.setattr(
        NativeSpatialIndex,
        "from_flat_index",
        classmethod(_slow_factory),
    )
    start = Barrier(2)

    def _create_owner():
        start.wait()
        return flat_index.to_native_spatial_index(source_token="shared-tree")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(_create_owner)
        second_future = executor.submit(_create_owner)
        first = first_future.result(timeout=5.0)
        second = second_future.result(timeout=5.0)

    assert created == 1
    assert first is second
    assert flat_index._native_spatial_index is first


def test_native_spatial_index_query_relation_reuses_index_state() -> None:
    tree = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(2.0, 2.0, 3.0, 3.0),
        ]
    )
    flat_index = build_flat_spatial_index(
        tree,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            reason="unit test native spatial index relation query",
        ),
    )
    native_index = flat_index.to_native_spatial_index(source_token="tree")
    query = from_shapely_geometries(
        [
            Point(0.5, 0.5),
            Point(10.0, 10.0),
            Point(2.5, 2.5),
        ]
    )

    relation, execution = native_index.query_relation(
        query,
        predicate="intersects",
        sort=True,
        query_token="query",
        return_device=False,
        return_metadata=True,
    )

    assert isinstance(relation, NativeRelation)
    assert execution.implementation in {
        "owned_cpu_spatial_query",
        "owned_gpu_spatial_query",
    }
    assert relation.left_token == "query"
    assert relation.right_token == "tree"
    assert relation.left_row_count == 3
    assert relation.right_row_count == 2
    assert relation.sorted_by_left is True
    np.testing.assert_array_equal(relation.left_indices, np.asarray([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(relation.right_indices, np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        relation.left_semijoin_rowset().positions,
        np.asarray([0, 2], dtype=np.int64),
    )


def test_public_spatial_index_query_relation_reuses_native_index_cache() -> None:
    tree_owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(10.0, 10.0, 11.0, 11.0),
        ]
    )
    tree = GeoDataFrame(
        {
            "zone": ["a", "b"],
            "geometry": GeoSeries(
                GeometryArray.from_owned(tree_owned),
                name="geometry",
            ),
        }
    )
    tree_state = _attach_native_tabular_state(tree)
    query_owned = from_shapely_geometries(
        [
            Point(0.5, 0.5),
            Point(20.0, 20.0),
            Point(10.5, 10.5),
        ]
    )
    query = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries(
                GeometryArray.from_owned(query_owned),
                name="geometry",
            ),
        }
    )
    query_state = _attach_native_tabular_state(query)

    sindex = tree.sindex
    relation, execution = sindex.query_relation(
        query_state,
        predicate="intersects",
        sort=True,
        source_token=tree_state.lineage_token,
        query_token=query_state.lineage_token,
        return_device=False,
    )
    cached_native_index = sindex._native_spatial_index
    relation_again, _execution_again = sindex.query_relation(
        query_state,
        predicate="intersects",
        sort=True,
        source_token=tree_state.lineage_token,
        query_token=query_state.lineage_token,
        return_device=False,
    )
    semijoin, _semijoin_execution = sindex.query_left_semijoin(
        query_state,
        predicate="intersects",
        source_token=tree_state.lineage_token,
        query_token=query_state.lineage_token,
    )
    antijoin, _antijoin_execution = sindex.query_left_antijoin(
        query_state,
        predicate="intersects",
        source_token=tree_state.lineage_token,
        query_token=query_state.lineage_token,
    )
    counts, _count_execution = sindex.query_left_match_count_expression(
        query_state,
        predicate="intersects",
        source_token=tree_state.lineage_token,
        query_token=query_state.lineage_token,
    )
    right_semijoin, _right_semijoin_execution = sindex.query_right_semijoin(
        query_state,
        predicate="intersects",
        source_token=tree_state.lineage_token,
    )

    assert isinstance(relation, NativeRelation)
    assert execution.implementation in {
        "owned_cpu_spatial_query",
        "owned_gpu_spatial_query",
    }
    assert sindex._native_spatial_index is cached_native_index
    assert relation.left_token == query_state.lineage_token
    assert relation.right_token == tree_state.lineage_token
    np.testing.assert_array_equal(relation.left_indices, np.asarray([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(relation.right_indices, np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(relation_again.left_indices, relation.left_indices)
    np.testing.assert_array_equal(relation_again.right_indices, relation.right_indices)
    semijoin_count = int(_host_array(semijoin.logical_count, dtype=np.int64)[0])
    assert semijoin_count == 2
    np.testing.assert_array_equal(
        _host_array(semijoin.positions[:semijoin_count], dtype=np.int64),
        np.asarray([0, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        _host_array(semijoin.partition_capacity_positions(), dtype=np.int64),
        np.asarray([0, 2, 1], dtype=np.int64),
    )
    antijoin_count = int(_host_array(antijoin.logical_count, dtype=np.int64)[0])
    assert antijoin_count == 1
    np.testing.assert_array_equal(
        _host_array(antijoin.positions[:antijoin_count], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        _host_array(antijoin.partition_capacity_positions(), dtype=np.int64),
        np.asarray([1, 0, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        _host_array(counts.values, dtype=np.int64),
        np.asarray([1, 0, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        _host_array(right_semijoin.positions, dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
    )


def test_public_sindex_query_exports_indices_from_native_relation() -> None:
    tree = GeoDataFrame(
        {
            "zone": ["a", "b"],
            "geometry": GeoSeries(
                GeometryArray.from_owned(
                    from_shapely_geometries(
                        [
                            box(0.0, 0.0, 1.0, 1.0),
                            box(10.0, 10.0, 11.0, 11.0),
                        ]
                    )
                ),
                name="geometry",
            ),
        }
    )
    query = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries(
                GeometryArray.from_owned(
                    from_shapely_geometries(
                        [
                            Point(0.5, 0.5),
                            Point(20.0, 20.0),
                            Point(10.5, 10.5),
                        ]
                    )
                ),
                name="geometry",
            ),
        }
    )
    clear_dispatch_events()
    clear_materialization_events()

    indices = tree.sindex.query(
        query.geometry,
        predicate="intersects",
        sort=True,
    )
    events = [
        event
        for event in get_dispatch_events(clear=True)
        if event.surface == "geopandas.sindex.query"
    ]
    materializations = [
        event
        for event in get_materialization_events(clear=True)
        if event.operation == "sindex_query"
    ]

    assert indices.tolist() == [[0, 2], [0, 1]]
    assert events[-1].implementation == "native_spatial_index"
    assert len(materializations) == 1
    assert "native_export_target=sindex-indices" in materializations[0].detail
    assert "rows=2" in materializations[0].detail


def test_public_sindex_nearest_exports_indices_from_native_relation() -> None:
    tree = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries([Point(0.0, 0.0), Point(10.0, 10.0)])),
        name="geometry",
    )
    query = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries([Point(0.5, 0.5), Point(10.5, 10.5)])),
        name="geometry",
    )
    clear_dispatch_events()
    clear_materialization_events()

    indices = tree.sindex.nearest(query)
    events = [
        event
        for event in get_dispatch_events(clear=True)
        if event.surface == "geopandas.sindex.nearest"
    ]
    materializations = [
        event
        for event in get_materialization_events(clear=True)
        if event.operation == "sindex_nearest"
    ]

    assert indices.tolist() == [[0, 1], [0, 1]]
    assert events[-1].implementation in {
        "native_relation_export",
        "point_tree_gpu_knn",
        "owned_cpu_nearest",
        "strtree_host",
    }
    assert len(materializations) == 1
    assert "native_export_target=sindex-nearest" in materializations[0].detail
    assert "rows=2" in materializations[0].detail


def test_native_geometry_metadata_from_device_owned_stays_device_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device metadata probe")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [Point(0, 0), Point(1, 1)],
        residency=Residency.DEVICE,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        metadata = NativeGeometryMetadata.from_owned(owned, source_token="frame")

    assert metadata.is_device
    assert metadata.bounds is not None
    assert metadata.source_token == "frame"
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_native_frame_geometry_metadata_carries_lineage_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native frame metadata probe")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [Point(0, 0), Point(1, 1)],
        residency=Residency.DEVICE,
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(dataframe=pd.DataFrame({"value": [10, 20]})),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        metadata = state.geometry_metadata()

    assert metadata.is_device
    assert metadata.source_token == state.lineage_token
    assert metadata.row_count == state.row_count
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_native_frame_cached_geometry_metadata_survives_device_rowset_take_without_runtime_d2h() -> (
    None
):
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native frame metadata take")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [Point(0, 0), Point(1, 1), Point(2, 2)],
        residency=Residency.DEVICE,
    )
    metadata = NativeGeometryMetadata.from_owned(owned, source_token="read")
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30], type=pa.int32())})
    result = NativeTabularResult(
        attributes=NativeAttributeTable(
            device_table=plc.Table.from_arrow(attribute_arrow),
            column_override=tuple(attribute_arrow.column_names),
            schema_override=attribute_arrow.schema,
        ),
        geometry=GeometryNativeResult.from_owned(owned, crs=None),
        geometry_name="geometry",
        column_order=("value", "geometry"),
        geometry_metadata=metadata,
    )
    state = NativeFrameState.from_native_tabular_result(result)
    rowset = NativeRowSet.from_positions(
        cp.asarray([2, 0], dtype=cp.int32),
        source_token=state.lineage_token,
        source_row_count=state.row_count,
        ordered=True,
        unique=True,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        cached = state.geometry_metadata()
        taken = state.take(rowset, preserve_index=True)
        taken_metadata = taken.geometry_metadata()

    assert cached.bounds is metadata.bounds
    assert cached.source_token == state.lineage_token
    assert taken.geometry_metadata_cache is not None
    assert taken_metadata.is_device
    assert taken_metadata.source_token == taken.lineage_token
    assert taken_metadata.row_count == 2
    assert taken.index_plan.kind == "device-labels"
    assert taken.attributes.device_table is not None
    assert get_materialization_events(clear=True) == []
    np.testing.assert_allclose(
        cp.asnumpy(taken_metadata.bounds),
        [[2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]],
    )
    reset_d2h_transfer_count()


def test_flat_spatial_index_exports_device_native_index_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native spatial index probe")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [Point(1.0, 2.0)],
        residency=Residency.DEVICE,
    )
    flat_index = build_flat_spatial_index(
        owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="unit test device native spatial index export",
        ),
    )
    assert flat_index.device_bounds is not None
    assert flat_index._host_bounds is None
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        metadata = flat_index.geometry_metadata(source_token="tree")
        native_index = flat_index.to_native_spatial_index(source_token="tree")

    assert metadata.is_device
    assert native_index.is_device
    assert native_index.metadata is not None
    assert metadata.bounds is flat_index.device_bounds
    assert native_index.metadata.bounds is flat_index.device_bounds
    assert native_index.source_token == "tree"
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_native_spatial_index_query_relation_uses_bounded_scalar_fence() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native spatial index query probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_stats,
        reset_d2h_transfer_count,
    )

    tree = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(1.0, 0.0, 2.0, 1.0),
            box(0.0, 1.0, 1.0, 2.0),
            box(1.0, 1.0, 2.0, 2.0),
        ]
    )
    tree.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native spatial index query tree",
    )
    flat_index = build_flat_spatial_index(
        tree,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="unit test device native spatial index relation query",
        ),
    )
    query = from_shapely_geometries(
        [
            Point(0.5, 0.5),
            Point(1.5, 0.5),
            Point(20.0, 20.0),
        ]
    )
    query.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native spatial index query geometry",
    )
    native_index = flat_index.to_native_spatial_index(source_token="tree")
    reset_d2h_transfer_count()
    clear_materialization_events()

    relation, execution = native_index.query_relation(
        query,
        predicate="intersects",
        sort=True,
        query_token="query",
        return_metadata=True,
    )
    rowset = relation.left_semijoin_rowset()
    d2h_count, d2h_bytes = get_d2h_transfer_stats()

    assert execution.selected is ExecutionMode.GPU
    assert relation.left_rowset().is_device
    assert relation.right_rowset().is_device
    assert rowset.is_device
    assert d2h_count <= 1
    assert d2h_bytes <= 64
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(relation.left_indices).tolist() == [0, 1]
    assert cp.asnumpy(relation.right_indices).tolist() == [0, 1]
    assert cp.asnumpy(rowset.positions).tolist() == [0, 1]
    reset_d2h_transfer_count()


def test_native_spatial_index_left_reductions_bypass_dense_relation_on_device() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native spatial semijoin probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_stats,
        reset_d2h_transfer_count,
    )

    tree = from_shapely_geometries([box(0.0, 0.0, 10.0, 10.0)] * 128)
    tree.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test dense native semijoin tree",
    )
    flat_index = build_flat_spatial_index(
        tree,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="unit test dense native semijoin index",
        ),
    )
    query = from_shapely_geometries(
        [Point(5.0, 5.0) if row % 2 == 0 else Point(20.0, 20.0) for row in range(256)]
    )
    query.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test dense native semijoin query",
    )
    native_index = flat_index.to_native_spatial_index(source_token="tree")
    reset_d2h_transfer_count()
    clear_materialization_events()

    rowset, execution = native_index.query_left_semijoin(
        query,
        predicate="intersects",
        query_token="query",
        return_metadata=True,
    )
    semijoin_d2h_count, semijoin_d2h_bytes = get_d2h_transfer_stats()
    reset_d2h_transfer_count()
    anti_rowset, anti_execution = native_index.query_left_antijoin(
        query,
        predicate="intersects",
        query_token="query",
        return_metadata=True,
    )
    antijoin_d2h_count, antijoin_d2h_bytes = get_d2h_transfer_stats()
    reset_d2h_transfer_count()
    count_expression, count_execution = native_index.query_left_match_count_expression(
        query,
        predicate="intersects",
        query_token="query",
        return_metadata=True,
    )
    count_d2h_count, count_d2h_bytes = get_d2h_transfer_stats()
    reset_d2h_transfer_count()
    right_rowset, right_execution = native_index.query_right_semijoin(
        query,
        predicate="intersects",
        return_metadata=True,
    )
    right_d2h_count, right_d2h_bytes = get_d2h_transfer_stats()

    assert execution.implementation == "owned_gpu_spatial_semijoin"
    assert semijoin_d2h_count <= 2
    assert semijoin_d2h_bytes <= 288
    assert rowset.is_device
    assert rowset.source_token == "query"
    rowset_count = int(cp.asnumpy(rowset.logical_count)[0])
    anti_rowset_count = int(cp.asnumpy(anti_rowset.logical_count)[0])
    right_rowset_count = int(cp.asnumpy(right_rowset.logical_count)[0])
    assert cp.asnumpy(rowset.positions[:rowset_count]).tolist() == list(range(0, 256, 2))
    assert anti_execution.implementation == "owned_gpu_spatial_semijoin"
    assert anti_rowset.is_device
    assert cp.asnumpy(anti_rowset.positions[:anti_rowset_count]).tolist() == list(
        range(1, 256, 2)
    )
    assert antijoin_d2h_count <= 2
    assert antijoin_d2h_bytes <= 288
    assert count_execution.implementation == "owned_gpu_spatial_match_count"
    assert count_expression.is_device
    assert cp.asnumpy(count_expression.values).tolist() == [
        128 if row % 2 == 0 else 0 for row in range(256)
    ]
    assert right_execution.implementation == "owned_gpu_spatial_right_semijoin"
    assert right_rowset.is_device
    assert cp.asnumpy(right_rowset.positions[:right_rowset_count]).tolist() == list(range(128))
    assert flat_index._host_bounds is None
    assert flat_index._host_order is None
    assert flat_index._host_morton_keys is None
    assert count_d2h_count <= 2
    assert count_d2h_bytes <= 288
    assert right_d2h_count <= 2
    assert right_d2h_bytes <= 288
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_native_spatial_index_mixed_family_reductions_partition_pair_capacity(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for mixed-family native reduction probe")
    cp = pytest.importorskip("cupy")

    tree = from_shapely_geometries(
        [
            box(0.0, 0.0, 2.0, 2.0),
            LineString([(10.0, 0.0), (10.0, 2.0)]),
            Point(20.0, 1.0),
        ]
    )
    tree.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test mixed-family direct reduction tree",
    )
    flat_index = build_flat_spatial_index(
        tree,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="unit test mixed-family direct reduction index",
        ),
    )
    query = from_shapely_geometries(
        [
            Point(1.0, 1.0),
            LineString([(9.0, 1.0), (11.0, 1.0)]),
            box(19.0, 0.0, 21.0, 2.0),
            Point(100.0, 100.0),
        ]
    )
    query.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test mixed-family direct reduction query",
    )
    native_index = flat_index.to_native_spatial_index(source_token="tree")
    from vibespatial.spatial import spatial_index_device as spatial_index_device_module

    classified_capacities = []
    classify_tile = spatial_index_device_module._classify_homogeneous_reduction_tile

    def classify_group_span(*args, **kwargs):
        if kwargs.get("source_offset") is not None:
            classified_capacities.append(
                (kwargs["pair_capacity"], kwargs["launch_capacity"])
            )
        return classify_tile(*args, **kwargs)

    monkeypatch.setattr(
        spatial_index_device_module,
        "_classify_homogeneous_reduction_tile",
        classify_group_span,
    )

    matched, matched_execution = native_index.query_left_semijoin(
        query,
        predicate="intersects",
        query_token="query",
        return_metadata=True,
    )
    unmatched, unmatched_execution = native_index.query_left_antijoin(
        query,
        predicate="intersects",
        query_token="query",
        return_metadata=True,
    )
    counts, count_execution = native_index.query_left_match_count_expression(
        query,
        predicate="intersects",
        query_token="query",
        return_metadata=True,
    )
    indexed, indexed_execution = native_index.query_right_semijoin(
        query,
        predicate="intersects",
        return_metadata=True,
    )

    assert isinstance(matched, NativeDeviceSelection)
    assert NativeRelationFamilyPartition.__name__ in Path(
        "src/vibespatial/spatial/spatial_index_device.py"
    ).read_text()
    assert "one all-family grouped partition pass per mixed tile" in matched_execution.reason
    assert unmatched_execution.implementation == "owned_gpu_spatial_semijoin"
    assert count_execution.implementation == "owned_gpu_spatial_match_count"
    assert indexed_execution.implementation == "owned_gpu_spatial_right_semijoin"
    matched_count = int(cp.asnumpy(matched.logical_count)[0])
    unmatched_count = int(cp.asnumpy(unmatched.logical_count)[0])
    indexed_count = int(cp.asnumpy(indexed.logical_count)[0])
    assert cp.asnumpy(matched.positions[:matched_count]).tolist() == [0, 1, 2]
    assert cp.asnumpy(unmatched.positions[:unmatched_count]).tolist() == [3]
    assert cp.asnumpy(counts.values).tolist() == [1, 1, 1, 0]
    assert cp.asnumpy(indexed.positions[:indexed_count]).tolist() == [0, 1, 2]
    family_pair_count = 9
    assert classified_capacities
    assert len(classified_capacities) % family_pair_count == 0
    for chunk_start in range(0, len(classified_capacities), family_pair_count):
        chunk = classified_capacities[chunk_start : chunk_start + family_pair_count]
        tile_capacity = chunk[0][0]
        assert all(pair_capacity == tile_capacity for pair_capacity, _ in chunk)
        assert sum(launch_capacity for _, launch_capacity in chunk) <= (
            tile_capacity + family_pair_count
        )


def test_relation_family_partition_preserves_source_positions() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for relation family source-position probe")
    cp = pytest.importorskip("cupy")
    left = cp.asarray([0, 1, 2, 3], dtype=cp.int32)
    right = cp.asarray([3, 2, 1, 0], dtype=cp.int32)
    source_positions = cp.asarray([13, 11, 17, 19], dtype=cp.int32)
    partition = NativeRelationFamilyPartition.from_pair_capacity(
        left,
        right,
        cp.asarray([True, True, False, True], dtype=cp.bool_),
        cp.asarray([0, 1, 2, 0], dtype=cp.int8),
        cp.asarray([0, 1, 2, 1], dtype=cp.int8),
        family_count=3,
        source_positions=source_positions,
    )

    packed = sorted(
        zip(
            cp.asnumpy(partition.left_indices).tolist(),
            cp.asnumpy(partition.right_indices).tolist(),
            cp.asnumpy(partition.source_positions).tolist(),
        )
    )
    assert packed == sorted([(0, 3, 13), (1, 2, 11), (2, 1, 17), (3, 0, 19)])
    group = partition.family_pair(
        left_family=GeometryFamily.POINT,
        right_family=GeometryFamily.LINESTRING,
        left_family_tag=0,
        right_family_tag=1,
        launch_capacity=1,
    )
    assert group.source_positions is partition.source_positions


def test_sjoin_native_spatial_index_device_relation_uses_bounded_fences() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native sjoin query probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_stats,
        reset_d2h_transfer_count,
    )

    left = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0.5 0.5)", "POINT (1.5 0.5)", "POINT (20 20)"],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "zone": ["a", "b", "c", "d"],
            "geometry": GeoSeries.from_wkt(
                [
                    "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
                    "POLYGON ((1 0, 2 0, 2 1, 1 1, 1 0))",
                    "POLYGON ((0 1, 1 1, 1 2, 0 2, 0 1))",
                    "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
                ],
                name="geometry",
            ),
        }
    )
    left_state = _attach_owned_native_tabular_state(left)
    right_state = _attach_owned_native_tabular_state(right)
    left_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native sjoin left geometry",
    )
    right_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native sjoin right geometry",
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    export_result, implementation, execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
        return_device=True,
    )
    relation = export_result.to_native_relation()
    rowset = relation.left_semijoin_rowset()
    d2h_count, d2h_bytes = get_d2h_transfer_stats()

    assert implementation == "native_spatial_index"
    assert execution is not None
    assert execution.selected is ExecutionMode.GPU
    assert relation.left_rowset().is_device
    assert relation.right_rowset().is_device
    assert rowset.is_device
    assert d2h_count <= 5
    assert d2h_bytes <= 160
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(relation.left_indices).tolist() == [0, 1]
    assert cp.asnumpy(relation.right_indices).tolist() == [0, 1]
    assert cp.asnumpy(rowset.positions).tolist() == [0, 1]
    reset_d2h_transfer_count()


def test_sjoin_return_device_multipoint_refine_stays_device_resident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for multipoint relation device probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_stats,
        reset_d2h_transfer_count,
    )

    left = GeoDataFrame(
        {
            "geometry": GeoSeries.from_wkt(
                ["MULTIPOINT ((0 0), (0.25 0.25))"],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "geometry": GeoSeries.from_wkt(
                ["POLYGON ((-1 -1, 1 -1, 1 1, -1 1, -1 -1))"],
                name="geometry",
            ),
        }
    )
    left_state = _attach_owned_native_tabular_state(left)
    right_state = _attach_owned_native_tabular_state(right)
    left_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native sjoin multipoint left geometry",
    )
    right_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native sjoin multipoint right geometry",
    )
    clear_fallback_events()
    clear_materialization_events()
    reset_d2h_transfer_count()

    export_result, implementation, execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
        return_device=True,
    )
    relation = export_result.to_native_relation()
    d2h_count, d2h_bytes = get_d2h_transfer_stats()
    events = get_fallback_events(clear=True)

    assert implementation == "native_spatial_index"
    assert execution is not None
    assert execution.selected is ExecutionMode.GPU
    assert relation.left_rowset().is_device
    assert relation.right_rowset().is_device
    assert d2h_count <= 5
    assert d2h_bytes <= 160
    assert events == []
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(relation.left_indices).tolist() == [0]
    assert cp.asnumpy(relation.right_indices).tolist() == [0]
    reset_d2h_transfer_count()

    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_fallback_events()
    export_result, implementation, execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
        return_device=True,
    )
    strict_relation = export_result.to_native_relation()
    assert implementation == "native_spatial_index"
    assert execution is not None
    assert execution.selected is ExecutionMode.GPU
    assert cp.asnumpy(strict_relation.left_indices).tolist() == [0]
    assert cp.asnumpy(strict_relation.right_indices).tolist() == [0]
    assert get_fallback_events(clear=True) == []
    monkeypatch.delenv(STRICT_NATIVE_ENV_VAR, raising=False)


def test_native_grouped_mixed_host_codes_device_values_reduce_on_device() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for mixed grouped reducer probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    grouped = NativeGrouped.from_dense_codes(
        np.asarray([1, 0, 1, -1, 2, 0], dtype=np.int32),
        group_count=3,
    )
    values = cp.asarray([10, 2, 5, 99, 7, 3], dtype=cp.int64)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        reduced = grouped.reduce_numeric(values, "sum")
        minimized = grouped.reduce_numeric(values, "min")
        maximized = grouped.reduce_numeric(values, "max")
        firsted = grouped.reduce_numeric(values, "first")
        lasted = grouped.reduce_numeric(values, "last")

    assert reduced.is_device
    assert minimized.is_device
    assert maximized.is_device
    assert firsted.is_device
    assert lasted.is_device
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(reduced.values).tolist() == [5, 15, 7]
    assert cp.asnumpy(minimized.values).tolist() == [2, 5, 7]
    assert cp.asnumpy(maximized.values).tolist() == [3, 10, 7]
    assert cp.asnumpy(firsted.values).tolist() == [2, 10, 7]
    assert cp.asnumpy(lasted.values).tolist() == [3, 5, 7]
    reset_d2h_transfer_count()


def test_native_relation_first_order_semijoin_stays_on_device() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device relation rowset probe")
    cp = pytest.importorskip("cupy")

    relation = NativeRelation(
        left_indices=cp.asarray([2, 0, 2], dtype=cp.int32),
        right_indices=cp.asarray([1, 1, 0], dtype=cp.int32),
        left_row_count=4,
        right_row_count=3,
    )

    left_rowset = relation.left_semijoin_rowset(order="first")
    right_rowset = relation.right_semijoin_rowset(order="first")

    assert left_rowset.is_device
    assert right_rowset.is_device
    assert cp.asnumpy(left_rowset.positions).tolist() == [2, 0]
    assert cp.asnumpy(right_rowset.positions).tolist() == [1, 0]


def test_native_relation_right_semijoin_rowset_feeds_owned_take_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device relation rowset take probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)])
    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test relation right-rowset device take",
    )
    relation = NativeRelation(
        left_indices=cp.asarray([0, 1, 2, 3, 4], dtype=cp.int32),
        right_indices=cp.asarray([2, 0, 2, 1, 0], dtype=cp.int32),
        left_row_count=5,
        right_row_count=4,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        rowset = relation.right_semijoin_rowset()
        taken = owned.take(rowset.positions)

    assert rowset.is_device
    assert rowset.unique
    assert rowset.ordered
    assert taken.row_count == 3
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(rowset.positions).tolist() == [0, 1, 2]
    reset_d2h_transfer_count()


def test_native_relation_device_grouped_reduction_stays_device_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device relation grouped reducer probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    relation = NativeRelation(
        left_indices=cp.asarray([0, 0, 2, 2, 2], dtype=cp.int32),
        right_indices=cp.asarray([1, 2, 0, 1, 2], dtype=cp.int32),
        left_row_count=4,
        right_row_count=3,
    )
    right_values = cp.asarray([10, 20, 30], dtype=cp.int64)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        reduced = relation.left_reduce_right_numeric(right_values, "sum")
        reduced_table = relation.left_reduce_right_numeric_columns(
            {
                "score": right_values,
                "matches": right_values,
            },
            {
                "score": "sum",
                "matches": "count",
            },
        )

    assert reduced.is_device
    assert reduced_table.is_device
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(reduced.values).tolist() == [50, 0, 60, 0]
    assert cp.asnumpy(reduced_table.columns["score"].values).tolist() == [50, 0, 60, 0]
    assert cp.asnumpy(reduced_table.columns["matches"].values).tolist() == [2, 0, 3, 0]
    reset_d2h_transfer_count()


def test_native_relation_device_distance_expression_filters_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device relation distance expression probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    relation = NativeRelation(
        left_indices=cp.asarray([0, 0, 1, 2, 2], dtype=cp.int32),
        right_indices=cp.asarray([3, 1, 2, 0, 3], dtype=cp.int32),
        left_token="left",
        right_token="right",
        predicate="nearest",
        distances=cp.asarray([5.0, 1.0, 2.5, 7.0, 1.5], dtype=cp.float64),
        left_row_count=3,
        right_row_count=4,
        sorted_by_left=True,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        expression = relation.distance_expression()
        rowset = expression.less_equal(2.5)
        filtered = relation.filter_pairs(rowset)
        left_min = filtered.left_reduce_distances("min")
        left_sum = filtered.left_reduce_distances("sum")
        right_count = filtered.right_reduce_distances("count")

    assert expression.is_device
    assert rowset.is_device
    assert filtered.left_rowset().is_device
    assert filtered.right_rowset().is_device
    assert left_min.is_device
    assert right_count.is_device
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(rowset.positions).tolist() == [1, 2, 4]
    assert cp.asnumpy(filtered.left_indices).tolist() == [0, 1, 2]
    assert cp.asnumpy(filtered.right_indices).tolist() == [1, 2, 3]
    assert cp.asnumpy(filtered.distances).tolist() == [1.0, 2.5, 1.5]
    assert cp.asnumpy(left_min.values).tolist() == [1.0, 2.5, 1.5]
    assert cp.asnumpy(left_sum.values).tolist() == [1.0, 2.5, 1.5]
    assert cp.asnumpy(right_count.values).tolist() == [0, 1, 1, 1]
    reset_d2h_transfer_count()


def test_native_grouped_selection_reduces_capacity_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for capacity grouped selection")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    selection = NativeDeviceSelection.from_mask(
        cp.asarray([True, False, True, True, False], dtype=cp.bool_),
        source_row_count=5,
    )
    grouped = NativeGroupedSelection(
        group_codes=cp.asarray([0, 0, 1, 1, 2], dtype=cp.int32),
        selection=selection,
        group_count=3,
    )
    values = cp.asarray([2.0, 20.0, 3.0, 5.0, 7.0], dtype=cp.float64)
    flags = cp.asarray([True, False, False, True, True], dtype=cp.bool_)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        sums = grouped.reduce_numeric(values, "sum")
        means = grouped.reduce_numeric(values, "mean")
        minima = grouped.reduce_numeric(values, "min")
        maxima = grouped.reduce_numeric(values, "max")
        any_flags = grouped.reduce_numeric(flags, "any")
        all_flags = grouped.reduce_numeric(flags, "all")

    assert grouped.capacity == 5
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(sums.values).tolist() == [2.0, 8.0, 0.0]
    np.testing.assert_allclose(
        cp.asnumpy(means.values),
        np.asarray([2.0, 4.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        cp.asnumpy(minima.values),
        np.asarray([2.0, 3.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        cp.asnumpy(maxima.values),
        np.asarray([2.0, 5.0, np.nan]),
        equal_nan=True,
    )
    assert cp.asnumpy(any_flags.values).tolist() == [True, True, False]
    assert cp.asnumpy(all_flags.values).tolist() == [True, False, True]
    reset_d2h_transfer_count()


def test_native_relation_capacity_selection_reduces_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for capacity relation selection")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    relation = NativeRelation(
        left_indices=cp.asarray([0, 0, 1, 2, 2], dtype=cp.int32),
        right_indices=cp.asarray([3, 1, 2, 0, 3], dtype=cp.int32),
        left_token="left",
        right_token="right",
        predicate="nearest",
        distances=cp.asarray([5.0, 1.0, 2.5, 7.0, 1.5], dtype=cp.float64),
        left_row_count=3,
        right_row_count=4,
        sorted_by_left=True,
    )
    left_geometry = from_shapely_geometries(
        [Point(0.0, 0.0), Point(10.0, 0.0), Point(20.0, 0.0)],
        residency=Residency.DEVICE,
    )
    right_geometry = from_shapely_geometries(
        [
            Point(999.0, 0.0),
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(20.0, 0.0),
        ],
        residency=Residency.DEVICE,
    )
    reset_d2h_transfer_count()


    clear_materialization_events()

    with assert_zero_d2h_transfers():
        filtered = relation.filter_by_distance_selection("<=", 2.5)
        attribute_filtered = relation.filter_by_equal_columns_selection(
            {"zone": cp.asarray([1, 2, 1], dtype=cp.int32)},
            {"zone": cp.asarray([0, 1, 2, 0], dtype=cp.int32)},
        )
        left_min = filtered.left_reduce_distances("min")
        left_sum = filtered.left_reduce_distances("sum")
        right_count = filtered.right_reduce_distances("count")
        left_matches = filtered.left_match_count_expression()
        right_matches = filtered.right_match_count_expression()
        attribute_left_matches = attribute_filtered.left_match_count_expression()
        pair_geometry = filtered.physicalize_geometries(
            left_geometry,
            right_geometry,
        )
        pair_intersection = pair_geometry.constructive_native("intersection")

    assert filtered.capacity == 5
    with pytest.raises(TypeError, match="logical length is device-resident"):
        len(filtered)
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(filtered.logical_count).tolist() == [3]
    assert cp.asnumpy(attribute_filtered.logical_count).tolist() == [2]
    assert cp.asnumpy(left_min.values).tolist() == [1.0, 2.5, 1.5]
    assert cp.asnumpy(left_sum.values).tolist() == [1.0, 2.5, 1.5]
    assert cp.asnumpy(right_count.values).tolist() == [0, 1, 1, 1]
    assert cp.asnumpy(left_matches.values).tolist() == [1, 1, 1]
    assert cp.asnumpy(right_matches.values).tolist() == [0, 1, 1, 1]
    assert cp.asnumpy(attribute_left_matches.values).tolist() == [1, 1, 0]
    assert pair_geometry.capacity == 5
    assert cp.asnumpy(pair_geometry.logical_count).tolist() == [3]
    assert cp.asnumpy(pair_geometry.left_geometry._ensure_device_state().validity).tolist() == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert cp.asnumpy(pair_geometry.right_geometry._ensure_device_state().validity).tolist() == [
        True,
        True,
        True,
        False,
        False,
    ]
    left_values = pair_geometry.left_geometry.to_shapely()
    right_values = pair_geometry.right_geometry.to_shapely()
    assert [value.x if value is not None else None for value in left_values] == [
        0.0,
        10.0,
        20.0,
        None,
        None,
    ]
    assert [value.x if value is not None else None for value in right_values] == [
        0.0,
        10.0,
        20.0,
        None,
        None,
    ]
    assert pair_intersection.capacity == 5
    assert pair_intersection.geometry.owned is not None
    assert cp.asnumpy(
        pair_intersection.geometry.owned._ensure_device_state().validity
    ).tolist() == [True, True, True, False, False]
    intersection_values = pair_intersection.geometry.owned.to_shapely()
    assert [
        value.x if value is not None and not value.is_empty else None
        for value in intersection_values
    ] == [0.0, 10.0, 20.0, None, None]
    reset_d2h_transfer_count()


@pytest.mark.gpu
def test_relation_selection_preserves_singular_right_constructive_carrier() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for singular-right constructive carrier")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    relation = NativeRelation(
        left_indices=cp.asarray([0, 1], dtype=cp.int32),
        right_indices=cp.asarray([0, 0], dtype=cp.int32),
        left_row_count=2,
        right_row_count=1,
    )
    selected = NativeRelationSelection(
        relation=relation,
        selection=NativeDeviceSelection.identity(2),
    )
    left_geometry = from_shapely_geometries(
        [
            Polygon([(0.0, 0.0), (2.0, 0.0), (1.0, 2.0), (0.0, 0.0)]),
            Polygon([(3.0, 0.0), (5.0, 0.0), (4.0, 2.0), (3.0, 0.0)]),
        ],
        residency=Residency.DEVICE,
    )
    right_geometry = from_shapely_geometries(
        [
            Polygon(
                [
                    (1.0, -1.0),
                    (4.0, -1.0),
                    (4.5, 1.0),
                    (3.5, 3.0),
                    (1.0, 3.0),
                    (1.0, -1.0),
                ]
            )
        ],
        residency=Residency.DEVICE,
    )
    reset_d2h_transfer_count()
    clear_dispatch_events()

    pair_geometry = selected.physicalize_geometries(
        left_geometry,
        right_geometry,
    )
    result = pair_geometry.constructive_native("intersection")

    assert pair_geometry.broadcast_right_geometry is right_geometry
    assert result.geometry.owned is not None
    d2h_events = get_d2h_transfer_events(clear=True)
    assert len(d2h_events) == 1
    assert d2h_events[0].reason == "fused multi-root capacity scatter exact allocation packet"
    assert d2h_events[0].bytes_transferred <= 448
    right_value = right_geometry.to_shapely()[0]
    expected = [
        left.intersection(right_value) for left in left_geometry.to_shapely()
    ]
    actual = result.geometry.owned.to_shapely()
    assert all(left.equals(right) for left, right in zip(actual, expected))
    assert any(
        event.implementation == "broadcast_right_virtual_segment_topology_gpu"
        for event in get_dispatch_events(clear=True)
    )
    reset_d2h_transfer_count()


def test_native_relation_device_match_count_expressions_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device relation match-count probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    relation = NativeRelation(
        left_indices=cp.asarray([2, 0, 2, 4, 4, 4], dtype=cp.int32),
        right_indices=cp.asarray([1, 1, 0, 2, 3, 3], dtype=cp.int32),
        left_token="left",
        right_token="right",
        left_row_count=5,
        right_row_count=4,
    )
    reset_d2h_transfer_count()

    with assert_zero_d2h_transfers():
        left_expression = relation.left_match_count_expression()
        right_expression = relation.right_match_count_expression()
        repeated_left = left_expression.greater_than(1)
        singleton_right = right_expression.equal_to(1)

    assert left_expression.is_device
    assert right_expression.is_device
    assert repeated_left.is_device
    assert singleton_right.is_device
    assert left_expression.source_token == "left"
    assert right_expression.source_token == "right"
    assert cp.asnumpy(left_expression.values).tolist() == [1, 0, 2, 0, 3]
    assert cp.asnumpy(right_expression.values).tolist() == [1, 2, 1, 2]
    assert cp.asnumpy(repeated_left.positions).tolist() == [2, 4]
    assert cp.asnumpy(singleton_right.positions).tolist() == [0, 2]
    reset_d2h_transfer_count()


def test_native_relation_device_attribute_filter_stays_device_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device relation attribute filter probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    relation = NativeRelation(
        left_indices=cp.asarray([0, 0, 1, 2, 2, 2], dtype=cp.int32),
        right_indices=cp.asarray([0, 1, 2, 0, 1, 2], dtype=cp.int32),
        distances=cp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=cp.float64),
        left_row_count=3,
        right_row_count=3,
        sorted_by_left=True,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        filtered = relation.filter_by_equal_columns(
            {
                "zone": cp.asarray([1, 2, 1], dtype=cp.int32),
                "active": cp.asarray([True, False, True], dtype=cp.bool_),
            },
            {
                "zone": cp.asarray([1, 1, 2], dtype=cp.int32),
                "active": cp.asarray([True, False, True], dtype=cp.bool_),
            },
        )

    assert filtered.left_rowset().is_device
    assert filtered.right_rowset().is_device
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(filtered.left_indices).tolist() == [0, 2]
    assert cp.asnumpy(filtered.right_indices).tolist() == [0, 0]
    assert cp.asnumpy(filtered.distances).tolist() == [1.0, 4.0]
    reset_d2h_transfer_count()


def test_native_relation_string_attribute_filter_uses_device_columns() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device relation string attribute filter probe")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    left_table = plc.Table.from_arrow(
        pa.table(
            {
                "zone": pa.array(["a", "b", "a"], type=pa.large_string()),
                "active": pa.array([True, False, True]),
            }
        )
    )
    right_table = plc.Table.from_arrow(
        pa.table(
            {
                "zone": pa.array(["a", "a", "b"], type=pa.large_string()),
                "active": pa.array([True, False, True]),
            }
        )
    )
    relation = NativeRelation(
        left_indices=cp.asarray([0, 0, 1, 2, 2, 2], dtype=cp.int32),
        right_indices=cp.asarray([0, 1, 2, 0, 1, 2], dtype=cp.int32),
        distances=cp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=cp.float64),
        left_row_count=3,
        right_row_count=3,
        sorted_by_left=True,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        filtered = relation.filter_by_equal_columns(
            {
                "zone": left_table.columns()[0],
                "active": left_table.columns()[1],
            },
            {
                "zone": right_table.columns()[0],
                "active": right_table.columns()[1],
            },
        )

    assert filtered.left_rowset().is_device
    assert filtered.right_rowset().is_device
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(filtered.left_indices).tolist() == [0, 2]
    assert cp.asnumpy(filtered.right_indices).tolist() == [0, 0]
    assert cp.asnumpy(filtered.distances).tolist() == [1.0, 4.0]
    reset_d2h_transfer_count()


def test_sjoin_native_on_attribute_filters_device_relation_pairs() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native sjoin attribute filter probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_stats,
        reset_d2h_transfer_count,
    )

    left = GeoDataFrame(
        {
            "zone": [1, 1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 0)", "POINT (2 0)"],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "zone": [1, 2],
            "geometry": GeoSeries.from_wkt(
                [
                    "POLYGON ((-1 -1, 1.5 -1, 1.5 1, -1 1, -1 -1))",
                    "POLYGON ((-0.5 -1, 2.5 -1, 2.5 1, -0.5 1, -0.5 -1))",
                ],
                name="geometry",
            ),
        }
    )
    left_state = _attach_owned_native_tabular_state(left, attribute_storage="device")
    right_state = _attach_owned_native_tabular_state(right, attribute_storage="device")
    left_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native sjoin on_attribute left geometry",
    )
    right_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native sjoin on_attribute right geometry",
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    export_result, implementation, execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
        on_attribute=["zone"],
        return_device=True,
    )
    relation = export_result.to_native_relation()
    d2h_count, d2h_bytes = get_d2h_transfer_stats()

    assert implementation == "native_spatial_index"
    assert execution is not None
    assert execution.selected is ExecutionMode.GPU
    assert relation.left_rowset().is_device
    assert relation.right_rowset().is_device
    assert d2h_count <= 5
    assert d2h_bytes <= 160
    assert get_materialization_events(clear=True) == []
    assert sorted(
        zip(
            cp.asnumpy(relation.left_indices).tolist(),
            cp.asnumpy(relation.right_indices).tolist(),
            strict=True,
        )
    ) == [(0, 0), (1, 0), (2, 1)]
    reset_d2h_transfer_count()


def test_sjoin_on_attribute_waits_for_cross_stream_device_columns() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native sjoin attribute stream ordering")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")

    from vibespatial.cuda._runtime import pylibcudf_column_from_device

    delay = cp.RawKernel(
        r'''
        extern "C" __global__ void delay(unsigned long long ticks) {
            unsigned long long start = clock64();
            while (clock64() - start < ticks) {}
        }
        ''',
        "delay",
    )
    blocker = cp.cuda.Stream(non_blocking=True)
    producer = cp.cuda.Stream(non_blocking=True)
    consumer = cp.cuda.Stream(non_blocking=True)
    ready = cp.cuda.Event(disable_timing=True)
    left_values = cp.zeros(1, dtype=cp.int32)
    right_values = cp.zeros(1, dtype=cp.int32)
    right_attributes = NativeAttributeTable(
        device_table=plc.Table([pylibcudf_column_from_device(right_values)]),
        index_override=pd.RangeIndex(1),
        column_override=("zone",),
        schema_override=pa.schema([pa.field("zone", pa.int32())]),
    )
    with blocker:
        delay((1,), (1,), (100_000_000,))
        ready.record(blocker)
    with producer:
        producer.wait_event(ready)
        left_values.fill(123)
        left_attributes = NativeAttributeTable(
            device_table=plc.Table([pylibcudf_column_from_device(left_values)]),
            index_override=pd.RangeIndex(1),
            column_override=("zone",),
            schema_override=pa.schema([pa.field("zone", pa.int32())]),
        )

    relation = NativeRelation(
        left_indices=cp.asarray([0], dtype=cp.int32),
        right_indices=cp.asarray([0], dtype=cp.int32),
        left_row_count=1,
        right_row_count=1,
    )
    with consumer:
        left_columns = _device_columns_from_native_attributes(left_attributes, ("zone",))
        right_columns = _device_columns_from_native_attributes(right_attributes, ("zone",))
        assert left_columns is not None
        assert right_columns is not None
        filtered = relation.filter_by_equal_columns(left_columns, right_columns)

    assert int(filtered.left_indices.size) == 0
    assert int(filtered.right_indices.size) == 0


def test_sjoin_public_string_on_attribute_filters_device_relation_in_strict_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for public native sjoin attribute filter probe")
    cp = pytest.importorskip("cupy")

    left = GeoDataFrame(
        {
            "zone": ["a", "a", "b"],
            "active": [True, False, True],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 0)", "POINT (2 0)"],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "zone": ["a", "b"],
            "active": [True, True],
            "geometry": GeoSeries.from_wkt(
                [
                    "POLYGON ((-1 -1, 1.5 -1, 1.5 1, -1 1, -1 -1))",
                    "POLYGON ((-1 -1, 2.5 -1, 2.5 1, -1 1, -1 -1))",
                ],
                name="geometry",
            ),
        }
    )
    clear_fallback_events()
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    export_result, _implementation, _execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
        on_attribute=["zone", "active"],
        return_device=True,
    )
    monkeypatch.delenv(STRICT_NATIVE_ENV_VAR, raising=False)
    relation = export_result.to_native_relation()

    assert relation.left_rowset().is_device
    assert relation.right_rowset().is_device
    assert get_fallback_events(clear=True) == []
    assert sorted(
        zip(
            cp.asnumpy(relation.left_indices).tolist(),
            cp.asnumpy(relation.right_indices).tolist(),
            strict=True,
        )
    ) == [(0, 0), (2, 1)]


def test_sjoin_nearest_native_on_attribute_filters_device_relation_pairs() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native nearest attribute filter probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_stats,
        reset_d2h_transfer_count,
    )

    left = GeoDataFrame(
        {
            "zone": [1, 2, 1],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 0)", "POINT (2 0)"],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "zone": [1, 1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0.1 0)", "POINT (1.1 0)", "POINT (2.1 0)"],
                name="geometry",
            ),
        }
    )
    _attach_native_tabular_state(left, attribute_storage="device")
    _attach_native_tabular_state(right, attribute_storage="device")
    reset_d2h_transfer_count()
    clear_materialization_events()

    native_result, selected = _sjoin_nearest_relation_result(
        left,
        right,
        max_distance=0.25,
        how="inner",
        return_distance=True,
        exclusive=False,
        on_attribute=["zone"],
    )
    relation = native_result.to_native_relation(
        left_token="left",
        right_token="right",
        predicate="nearest",
        left_row_count=len(left),
        right_row_count=len(right),
    )
    d2h_count, d2h_bytes = get_d2h_transfer_stats()

    assert selected is ExecutionMode.GPU
    assert relation.left_rowset().is_device
    assert relation.right_rowset().is_device
    assert d2h_count <= 1
    assert d2h_bytes <= 8
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(relation.left_indices).tolist() == [0]
    assert cp.asnumpy(relation.right_indices).tolist() == [0]
    assert np.allclose(cp.asnumpy(relation.distances), np.asarray([0.1]))
    reset_d2h_transfer_count()


def test_sjoin_native_on_attribute_null_key_is_observable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {
            "zone": pd.Series(["a", None], dtype=object),
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 0)"],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "zone": pd.Series(["a"], dtype=object),
            "geometry": GeoSeries.from_wkt(
                ["POLYGON ((-1 -1, 2 -1, 2 1, -1 1, -1 -1))"],
                name="geometry",
            ),
        }
    )
    _attach_owned_native_tabular_state(left)
    _attach_owned_native_tabular_state(right)
    clear_fallback_events()

    export_result, _implementation, _execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
        on_attribute=["zone"],
        return_device=True,
    )
    relation = export_result.to_native_relation()
    events = get_fallback_events(clear=True)

    assert relation.left_indices.tolist() == [0]
    assert relation.right_indices.tolist() == [0]
    assert events
    assert events[-1].surface == "geopandas.sjoin"
    assert "on_attribute" in events[-1].reason
    assert events[-1].pipeline == "spatial_join/native_on_attribute"

    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    with pytest.raises(StrictNativeFallbackError, match="on_attribute"):
        _sjoin_export_result(
            left,
            right,
            "inner",
            "intersects",
            None,
            "left",
            "right",
            on_attribute=["zone"],
            return_device=True,
        )
    monkeypatch.delenv(STRICT_NATIVE_ENV_VAR, raising=False)


def test_unary_constructive_native_tabular_wrappers_carry_source_provenance() -> None:
    from vibespatial.constructive.boundary import boundary_native_tabular_result
    from vibespatial.constructive.convex_hull import convex_hull_native_tabular_result
    from vibespatial.constructive.envelope import envelope_native_tabular_result
    from vibespatial.constructive.minimum_rotated_rectangle import (
        minimum_rotated_rectangle_native_tabular_result,
    )

    owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 2.0),
            box(2.0, 2.0, 4.0, 5.0),
        ]
    )
    source_rows = np.asarray([9, 12], dtype=np.int32)

    cases = (
        (envelope_native_tabular_result, "envelope"),
        (boundary_native_tabular_result, "boundary"),
        (convex_hull_native_tabular_result, "convex_hull"),
        (
            minimum_rotated_rectangle_native_tabular_result,
            "minimum_rotated_rectangle",
        ),
    )
    for factory, operation in cases:
        result = factory(
            owned,
            dispatch_mode=ExecutionMode.CPU,
            source_rows=source_rows,
            source_tokens=("unary-source",),
        )

        assert result.geometry.row_count == owned.row_count
        assert result.column_order == ("geometry",)
        assert isinstance(result.provenance, NativeGeometryProvenance)
        assert result.provenance.operation == operation
        assert result.provenance.source_tokens == ("unary-source",)
        assert result.provenance.source_rows.tolist() == [9, 12]
        assert result.geometry_metadata is not None
        assert result.geometry_metadata.row_count == owned.row_count


def test_row_aligned_unary_native_tabular_wrappers_cover_transform_family() -> None:
    from vibespatial.constructive.affine_transform import (
        translate_native_tabular_result,
    )
    from vibespatial.constructive.centroid import centroid_native_tabular_result
    from vibespatial.constructive.exterior import exterior_native_tabular_result
    from vibespatial.constructive.extract_unique_points import (
        extract_unique_points_native_tabular_result,
    )
    from vibespatial.constructive.interiors import interiors_native_tabular_result
    from vibespatial.constructive.line_merge import line_merge_native_tabular_result
    from vibespatial.constructive.linear_ref import interpolate_native_tabular_result
    from vibespatial.constructive.minimum_bounding_circle import (
        minimum_bounding_circle_native_tabular_result,
    )
    from vibespatial.constructive.minimum_clearance import (
        minimum_clearance_line_native_tabular_result,
    )
    from vibespatial.constructive.normalize import normalize_native_tabular_result
    from vibespatial.constructive.orient import orient_native_tabular_result
    from vibespatial.constructive.remove_repeated_points import (
        remove_repeated_points_native_tabular_result,
    )
    from vibespatial.constructive.representative_point import (
        representative_point_native_tabular_result,
    )
    from vibespatial.constructive.reverse import reverse_native_tabular_result
    from vibespatial.constructive.segmentize import segmentize_native_tabular_result
    from vibespatial.constructive.set_precision import set_precision_native_tabular_result
    from vibespatial.constructive.simplify import simplify_native_tabular_result
    from vibespatial.constructive.stroke import offset_curve_native_tabular_result

    lines = [
        LineString([(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (2.0, 0.0)]),
        LineString([(0.0, 0.0), (0.0, 2.0)]),
    ]
    owned = from_shapely_geometries(lines)
    source_rows = np.asarray([4, 8], dtype=np.int32)

    cases = (
        (
            lambda **kwargs: translate_native_tabular_result(
                owned,
                xoff=1.0,
                yoff=2.0,
                dispatch_mode=ExecutionMode.CPU,
                **kwargs,
            ),
            "translate",
        ),
        (
            lambda **kwargs: reverse_native_tabular_result(
                owned,
                dispatch_mode=ExecutionMode.CPU,
                **kwargs,
            ),
            "reverse",
        ),
        (
            lambda **kwargs: normalize_native_tabular_result(
                owned,
                dispatch_mode=ExecutionMode.CPU,
                **kwargs,
            ),
            "normalize",
        ),
        (
            lambda **kwargs: orient_native_tabular_result(
                owned,
                dispatch_mode=ExecutionMode.CPU,
                **kwargs,
            ),
            "orient",
        ),
        (
            lambda **kwargs: simplify_native_tabular_result(
                owned,
                0.1,
                preserve_topology=False,
                dispatch_mode=ExecutionMode.CPU,
                **kwargs,
            ),
            "simplify",
        ),
        (
            lambda **kwargs: segmentize_native_tabular_result(
                owned,
                0.75,
                dispatch_mode=ExecutionMode.CPU,
                **kwargs,
            ),
            "segmentize",
        ),
        (
            lambda **kwargs: remove_repeated_points_native_tabular_result(
                owned,
                tolerance=0.0,
                dispatch_mode=ExecutionMode.CPU,
                **kwargs,
            ),
            "remove_repeated_points",
        ),
        (
            lambda **kwargs: set_precision_native_tabular_result(
                owned,
                0.25,
                mode="pointwise",
                dispatch_mode=ExecutionMode.CPU,
                **kwargs,
            ),
            "set_precision",
        ),
        (
            lambda **kwargs: offset_curve_native_tabular_result(
                lines,
                0.1,
                join_style="mitre",
                **kwargs,
            ),
            "offset_curve",
        ),
        (
            lambda **kwargs: exterior_native_tabular_result(
                owned,
                dispatch_mode=ExecutionMode.CPU,
                **kwargs,
            ),
            "exterior",
        ),
        (
            lambda **kwargs: interiors_native_tabular_result(
                owned,
                dispatch_mode=ExecutionMode.CPU,
                **kwargs,
            ),
            "interiors",
        ),
        (
            lambda **kwargs: extract_unique_points_native_tabular_result(
                owned,
                dispatch_mode=ExecutionMode.CPU,
                **kwargs,
            ),
            "extract_unique_points",
        ),
        (
            lambda **kwargs: representative_point_native_tabular_result(
                owned,
                dispatch_mode=ExecutionMode.CPU,
                **kwargs,
            ),
            "representative_point",
        ),
        (
            lambda **kwargs: minimum_bounding_circle_native_tabular_result(
                owned,
                dispatch_mode=ExecutionMode.CPU,
                **kwargs,
            ),
            "minimum_bounding_circle",
        ),
        (
            lambda **kwargs: centroid_native_tabular_result(
                owned,
                dispatch_mode=ExecutionMode.CPU,
                **kwargs,
            ),
            "centroid",
        ),
        (
            lambda **kwargs: line_merge_native_tabular_result(
                owned,
                dispatch_mode=ExecutionMode.CPU,
                **kwargs,
            ),
            "line_merge",
        ),
        (
            lambda **kwargs: interpolate_native_tabular_result(
                owned,
                0.5,
                dispatch_mode=ExecutionMode.CPU,
                **kwargs,
            ),
            "interpolate",
        ),
        (
            lambda **kwargs: minimum_clearance_line_native_tabular_result(
                owned,
                dispatch_mode=ExecutionMode.CPU,
                **kwargs,
            ),
            "minimum_clearance_line",
        ),
    )

    for factory, operation in cases:
        result = factory(
            source_rows=source_rows,
            source_tokens=("row-aligned-unary-source",),
        )

        assert result.geometry.row_count == owned.row_count
        assert result.column_order == ("geometry",)
        assert isinstance(result.provenance, NativeGeometryProvenance)
        assert result.provenance.operation == operation
        assert result.provenance.source_tokens == ("row-aligned-unary-source",)
        assert result.provenance.source_rows.tolist() == [4, 8]
        assert result.geometry_metadata is not None
        assert result.geometry_metadata.row_count == owned.row_count


def test_point_parts_native_result_feeds_grouped_consumer_without_public_export() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native point part expansion")
    cp = pytest.importorskip("cupy")
    from vibespatial.constructive.binary_constructive import (
        point_parts_native_tabular_result,
    )
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [
            Point(0, 0),
            MultiPoint([(1, 0), (3, 0)]),
            LineString([(0, 0), (1, 0)]),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native point part expansion",
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    result = point_parts_native_tabular_result(
        owned,
        source_tokens=("point-parts-source",),
    )
    producer_events = get_d2h_transfer_events(clear=True)

    assert isinstance(result, NativeTabularSelection)
    assert isinstance(result.provenance, NativeGeometryProvenance)
    assert result.provenance.operation == "point_parts"
    assert result.provenance.source_tokens == ("point-parts-source",)
    assert result.provenance.is_device
    assert cp.asnumpy(result.provenance.part_family_tags).tolist() == [
        FAMILY_TAGS[GeometryFamily.POINT],
        FAMILY_TAGS[GeometryFamily.POINT],
        FAMILY_TAGS[GeometryFamily.POINT],
    ]
    assert producer_events == []

    reset_d2h_transfer_count()
    clear_materialization_events()
    with assert_zero_d2h_transfers():
        state = result.capacity_result.to_native_frame_state()
        centroid_x, _centroid_y = state.geometry_centroid_expressions()
        grouped = NativeGroupedSelection(
            group_codes=result.provenance.source_rows,
            selection=result.selection,
            group_count=owned.row_count,
            source_token=state.lineage_token,
        )
        reduced = grouped.reduce_expression(centroid_x, "sum")

    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(result.provenance.source_rows).tolist() == [0, 1, 1]
    assert cp.asnumpy(reduced.values).tolist() == [0.0, 4.0, 0.0]
    reset_d2h_transfer_count()


def test_polygonal_parts_native_result_feeds_grouped_consumer_without_public_export() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native polygonal part expansion")
    cp = pytest.importorskip("cupy")
    from vibespatial.constructive.binary_constructive import (
        polygonal_parts_native_tabular_result,
    )
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)]),
            Point(0, 0),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native polygonal part expansion",
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    result = polygonal_parts_native_tabular_result(
        owned,
        source_tokens=("polygonal-parts-source",),
    )
    producer_events = get_d2h_transfer_events(clear=True)

    assert isinstance(result, NativeTabularSelection)
    assert isinstance(result.provenance, NativeGeometryProvenance)
    assert result.provenance.operation == "polygonal_parts"
    assert result.provenance.source_tokens == ("polygonal-parts-source",)
    assert result.provenance.is_device
    assert cp.asnumpy(result.provenance.part_family_tags).tolist() == [
        FAMILY_TAGS[GeometryFamily.POLYGON],
        FAMILY_TAGS[GeometryFamily.POLYGON],
        FAMILY_TAGS[GeometryFamily.POLYGON],
    ]
    assert producer_events == []

    reset_d2h_transfer_count()
    clear_materialization_events()
    with assert_zero_d2h_transfers():
        state = result.capacity_result.to_native_frame_state()
        area = state.geometry_area_expression()
        grouped = NativeGroupedSelection(
            group_codes=result.provenance.source_rows,
            selection=result.selection,
            group_count=owned.row_count,
            source_token=state.lineage_token,
        )
        reduced = grouped.reduce_expression(area, "sum")

    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(result.provenance.source_rows).tolist() == [0, 1, 1]
    assert cp.asnumpy(reduced.values).tolist() == [1.0, 2.0, 0.0]
    reset_d2h_transfer_count()


def test_lineal_parts_native_result_feeds_grouped_consumer_without_public_export() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native lineal part expansion")
    cp = pytest.importorskip("cupy")
    from vibespatial.constructive.binary_constructive import (
        lineal_parts_native_tabular_result,
    )
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [
            LineString([(0, 0), (1, 0)]),
            MultiLineString([[(0, 0), (0, 2)], [(0, 0), (3, 0)]]),
            Point(0, 0),
        ]
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native lineal part expansion",
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    result = lineal_parts_native_tabular_result(
        owned,
        source_tokens=("lineal-parts-source",),
    )
    producer_events = get_d2h_transfer_events(clear=True)

    assert isinstance(result, NativeTabularSelection)
    assert isinstance(result.provenance, NativeGeometryProvenance)
    assert result.provenance.operation == "lineal_parts"
    assert result.provenance.source_tokens == ("lineal-parts-source",)
    assert result.provenance.is_device
    assert cp.asnumpy(result.provenance.part_family_tags).tolist() == [
        FAMILY_TAGS[GeometryFamily.LINESTRING],
        FAMILY_TAGS[GeometryFamily.LINESTRING],
        FAMILY_TAGS[GeometryFamily.LINESTRING],
    ]
    assert producer_events == []

    reset_d2h_transfer_count()
    clear_materialization_events()
    with assert_zero_d2h_transfers():
        state = result.capacity_result.to_native_frame_state()
        length = state.geometry_length_expression()
        grouped = NativeGroupedSelection(
            group_codes=result.provenance.source_rows,
            selection=result.selection,
            group_count=owned.row_count,
            source_token=state.lineage_token,
        )
        reduced = grouped.reduce_expression(length, "sum")

    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(result.provenance.source_rows).tolist() == [0, 1, 1]
    assert cp.asnumpy(reduced.values).tolist() == [1.0, 5.0, 0.0]
    reset_d2h_transfer_count()


def test_sjoin_export_result_exposes_private_native_relation_rowsets() -> None:
    left = GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (5 5)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "zone": ["a"],
            "geometry": GeoSeries.from_wkt(
                ["POLYGON ((-1 -1, 2 -1, 2 2, -1 2, -1 -1))"],
                name="geometry",
            ),
        }
    )

    export_result, _implementation, _execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
    )
    relation = export_result.to_native_relation()

    assert relation.predicate == "intersects"
    assert relation.left_row_count == 3
    assert relation.right_row_count == 1
    assert relation.left_semijoin_rowset().positions.tolist() == [0, 2]
    assert relation.left_antijoin_rowset().positions.tolist() == [1]


def test_sjoin_native_relation_uses_attached_native_state_lineage() -> None:
    left, left_state = _native_backed_geodataframe()
    right = GeoDataFrame(
        {
            "zone": ["a"],
            "geometry": GeoSeries.from_wkt(
                ["POLYGON ((-1 -1, 2 -1, 2 2, -1 2, -1 -1))"],
                name="geometry",
            ),
        }
    )
    right_result = NativeTabularResult(
        attributes=NativeAttributeTable(dataframe=right.drop(columns=["geometry"])),
        geometry=GeometryNativeResult.from_geoseries(right.geometry),
        geometry_name="geometry",
        column_order=("zone", "geometry"),
    )
    right_state = NativeFrameState.from_native_tabular_result(right_result)
    attach_native_state(right, right_state)

    export_result, _implementation, _execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
    )
    relation = export_result.to_native_relation()
    rowset = relation.left_semijoin_rowset()
    left_count_expression = export_result.left_match_count_expression()
    right_count_expression = export_result.right_match_count_expression()
    taken = left_state.take(rowset, preserve_index=False)

    assert relation.left_token == left_state.lineage_token
    assert relation.right_token == right_state.lineage_token
    assert rowset.source_token == left_state.lineage_token
    assert left_count_expression.source_token == left_state.lineage_token
    assert right_count_expression.source_token == right_state.lineage_token
    assert left_count_expression.values.tolist() == [1, 1]
    assert right_count_expression.values.tolist() == [2]
    assert left_count_expression.greater_than(0).positions.tolist() == [0, 1]
    assert taken.row_count == 2
    assert taken.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        2,
    ]


def test_sjoin_native_small_device_query_skips_index_metadata_fences() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device spatial index metadata test")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    def _attach_device_owned_state(frame: GeoDataFrame) -> NativeFrameState:
        result = NativeTabularResult(
            attributes=NativeAttributeTable(
                dataframe=frame.drop(columns=[frame._geometry_column_name])
            ),
            geometry=GeometryNativeResult.from_owned(
                from_shapely_geometries(
                    list(frame.geometry),
                    residency=Residency.DEVICE,
                ),
                crs=frame.crs,
            ),
            geometry_name=frame._geometry_column_name,
            column_order=tuple(frame.columns),
        )
        state = NativeFrameState.from_native_tabular_result(result)
        attach_native_state(frame, state)
        return state

    left = GeoDataFrame(
        {
            "parcel": np.arange(128, dtype=np.int64),
            "geometry": [
                box(
                    float(i % 16),
                    float(i // 16),
                    float(i % 16) + 0.75,
                    float(i // 16) + 0.75,
                )
                for i in range(128)
            ],
        }
    )
    right = GeoDataFrame(
        {
            "zone": np.arange(7, dtype=np.int64),
            "geometry": [box(float(i * 2), 0.0, float(i * 2) + 1.25, 8.0) for i in range(7)],
        }
    )
    _attach_device_owned_state(left)
    _attach_device_owned_state(right)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    export_result, implementation, execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
        return_device=True,
    )
    relation = export_result.relation_result.relation
    events = get_d2h_transfer_events(clear=True)

    assert implementation == "native_spatial_index"
    assert execution.selected is ExecutionMode.GPU
    assert relation.left_indices.size > 0
    assert hasattr(relation.left_indices, "__cuda_array_interface__")
    assert hasattr(relation.right_indices, "__cuda_array_interface__")
    assert not any(
        event.reason
        in {
            "spatial index regular-grid summary scalar fence",
            "flat spatial index device total-bounds scalar fence",
            "device spatial-index candidate-pair allocation fence",
        }
        for event in events
    )


def test_sjoin_query_prefers_attached_native_spatial_index() -> None:
    left = GeoDataFrame(
        {
            "value": [1, 2],
            "name": ["a", "b"],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    left_state = _attach_owned_native_tabular_state(left)
    right = GeoDataFrame(
        {
            "zone": ["a"],
            "geometry": GeoSeries.from_wkt(
                ["POLYGON ((-1 -1, 2 -1, 2 2, -1 2, -1 -1))"],
                name="geometry",
            ),
        }
    )
    right_state = _attach_owned_native_tabular_state(right)

    export_result, implementation, execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
    )
    relation = export_result.to_native_relation()

    assert implementation == "native_spatial_index"
    assert execution is not None
    assert relation.left_token == left_state.lineage_token
    assert relation.right_token == right_state.lineage_token
    assert relation.left_semijoin_rowset().positions.tolist() == [0, 1]
    assert relation.right_semijoin_rowset().positions.tolist() == [0]


def test_sjoin_export_result_consumes_attached_left_state_without_joined_export() -> None:
    left, _left_state = _native_backed_geodataframe()
    right = GeoDataFrame(
        {
            "zone": ["a", "b"],
            "geometry": GeoSeries.from_wkt(
                [
                    "POLYGON ((-1 -1, 0.5 -1, 0.5 0.5, -1 0.5, -1 -1))",
                    "POLYGON ((10 10, 11 10, 11 11, 10 11, 10 10))",
                ],
                name="geometry",
            ),
        }
    )
    right_result = NativeTabularResult(
        attributes=NativeAttributeTable(dataframe=right.drop(columns=["geometry"])),
        geometry=GeometryNativeResult.from_geoseries(right.geometry),
        geometry_name="geometry",
        column_order=("zone", "geometry"),
    )
    attach_native_state(right, NativeFrameState.from_native_tabular_result(right_result))
    clear_materialization_events()

    export_result, _implementation, _execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
    )
    matched = export_result.left_semijoin_native_frame(order="first")
    unmatched = export_result.left_antijoin_native_frame()
    matched_right = export_result.right_semijoin_native_frame(order="first")
    unmatched_right = export_result.right_antijoin_native_frame()

    assert matched is not None
    assert unmatched is not None
    assert matched_right is not None
    assert unmatched_right is not None
    assert matched.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [1]
    assert unmatched.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [2]
    assert matched_right.to_native_tabular_result().attributes.to_pandas()["zone"].tolist() == ["a"]
    assert unmatched_right.to_native_tabular_result().attributes.to_pandas()["zone"].tolist() == [
        "b"
    ]
    assert all(
        event.surface != "vibespatial.api.NativeTabularResult.to_geodataframe"
        for event in get_materialization_events(clear=True)
    )


def test_sjoin_export_result_reduces_attached_source_attributes() -> None:
    left = GeoDataFrame(
        {
            "left_value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "score": [10.0, 5.0],
            "weight": [2.0, 8.0],
            "geometry": GeoSeries.from_wkt(
                [
                    "POLYGON ((-1 -1, 2 -1, 2 2, -1 2, -1 -1))",
                    "POLYGON ((0.5 0.5, 2 0.5, 2 2, 0.5 2, 0.5 0.5))",
                ],
                name="geometry",
            ),
        }
    )
    _attach_native_tabular_state(left)
    _attach_native_tabular_state(right)
    clear_materialization_events()

    export_result, _implementation, _execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
    )
    left_reduced = export_result.left_reduce_right_numeric_columns(
        {
            "score_sum": "score",
            "score_mean": "score",
            "match_count": "score",
            "weight_sum": "weight",
        },
        {
            "score_sum": "sum",
            "score_mean": "mean",
            "match_count": "count",
            "weight_sum": "sum",
        },
    )
    right_reduced = export_result.right_reduce_left_numeric_columns(
        {"left_value_sum": "left_value"},
        {"left_value_sum": "sum"},
    )

    assert left_reduced is not None
    assert right_reduced is not None
    left_frame = left_reduced.to_pandas()
    right_frame = right_reduced.to_pandas()
    assert left_frame["score_sum"].tolist() == [10.0, 15.0]
    assert left_frame["score_mean"].tolist() == [10.0, 7.5]
    assert left_frame["match_count"].tolist() == [1, 2]
    assert left_frame["weight_sum"].tolist() == [2.0, 10.0]
    assert right_frame["left_value_sum"].tolist() == [3, 2]
    assert all(
        event.surface != "vibespatial.api.NativeTabularResult.to_geodataframe"
        for event in get_materialization_events(clear=True)
    )


def test_sjoin_export_result_lowers_joined_rows_to_native_frame_without_public_export() -> None:
    left = GeoDataFrame(
        {
            "left_value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "score": [10.0, 5.0],
            "geometry": GeoSeries.from_wkt(
                [
                    "POLYGON ((-1 -1, 0.5 -1, 0.5 0.5, -1 0.5, -1 -1))",
                    "POLYGON ((0.5 0.5, 2 0.5, 2 2, 0.5 2, 0.5 0.5))",
                ],
                name="geometry",
            ),
        }
    )
    _attach_native_tabular_state(left)
    _attach_native_tabular_state(right)
    clear_materialization_events()

    export_result, _implementation, _execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
    )
    payload = export_result.to_native_tabular_result(attribute_storage="pandas")
    joined_state = export_result.to_native_frame_state(attribute_storage="pandas")

    assert payload.geometry.row_count == 2
    assert payload.geometry_metadata is not None
    assert joined_state.row_count == 2
    assert joined_state.column_order == (
        "left_value",
        "geometry",
        "index_right",
        "score",
    )
    assert joined_state.to_native_tabular_result().attributes.to_pandas()["score"].tolist() == [
        10.0,
        5.0,
    ]
    assert all(
        event.surface != "vibespatial.api.NativeTabularResult.to_geodataframe"
        for event in get_materialization_events(clear=True)
    )


def test_relation_join_export_result_can_build_device_attribute_frame_with_distance() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device relation join attribute storage")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    left = GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "score": [10.0, 5.0],
            "geometry": GeoSeries.from_wkt(
                ["POINT (2 2)", "POINT (3 3)"],
                name="geometry",
            ),
        }
    )
    left_state = _attach_owned_native_tabular_state(left, attribute_storage="device")
    right_state = _attach_owned_native_tabular_state(right, attribute_storage="device")
    left_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test relation join export left geometry",
    )
    right_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test relation join export right geometry",
    )
    export_result = RelationJoinExportResult(
        relation_result=RelationJoinResult(
            RelationIndexResult(
                cp.asarray([0, 1], dtype=cp.int32),
                cp.asarray([1, 0], dtype=cp.int32),
            ),
            distances=cp.asarray([0.5, 1.5], dtype=cp.float64),
        ),
        left_df=left,
        right_df=right,
        how="inner",
        lsuffix="left",
        rsuffix="right",
        distance_col="dist",
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        payload = export_result.to_native_tabular_result(attribute_storage="device")
        state = payload.to_native_frame_state()
        attributes = payload.attributes
        arrays = attributes.numeric_column_arrays(("value", "index_right", "score", "dist"))

    assert payload.index_plan is not None
    assert payload.index_plan.kind == "device-labels"
    assert payload.attributes.index.equals(pd.RangeIndex(2))
    assert attributes.parts is not None
    assert all(part.device_table is not None for part in attributes.parts)
    assert state.index_plan.kind == "device-labels"
    assert state.column_order == ("value", "geometry", "index_right", "score", "dist")
    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [1, 2]
    assert cp.asnumpy(arrays["index_right"]).tolist() == [1, 0]
    assert cp.asnumpy(arrays["score"]).tolist() == [5.0, 10.0]
    assert cp.asnumpy(arrays["dist"]).tolist() == [0.5, 1.5]
    assert get_materialization_events(clear=True) == []
    exported = payload.to_geodataframe()
    assert exported.index.tolist() == [0, 1]
    assert exported["index_right"].tolist() == [1, 0]
    reset_d2h_transfer_count()


def test_public_sjoin_transiently_lowers_owned_sources_with_on_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for transient relation source lowering")
    pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")

    left_index = pd.RangeIndex(10, 12, name="left_id")
    right_index = pd.RangeIndex(20, 22, name="right_id")
    left = GeoDataFrame(
        {
            "zone": ["a", "b"],
            "value": [1, 2],
            "nullable": pd.array([1, None], dtype="Int64"),
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (2 0)"],
                index=left_index,
                name="geometry",
            ),
        },
        index=left_index,
    )
    right = GeoDataFrame(
        {
            "zone": ["a", "b"],
            "value": [10, 20],
            "label": pd.array(["x", None], dtype="string"),
            "geometry": GeoSeries.from_wkt(
                [
                    "POLYGON ((-1 -1, 1 -1, 1 1, -1 1, -1 -1))",
                    "POLYGON ((1 -1, 3 -1, 3 1, 1 1, 1 -1))",
                ],
                index=right_index,
                name="geometry",
            ),
        },
        index=right_index,
    )
    assert get_native_state(left) is None
    assert get_native_state(right) is None

    def _fail_relation_host_export(self, **kwargs):
        raise AssertionError("transient relation lowering must not export pair rows")

    monkeypatch.setattr(RelationIndexResult, "to_host", _fail_relation_host_export)
    joined = geopandas.sjoin(
        left,
        right,
        how="inner",
        predicate="intersects",
        lsuffix="L",
        rsuffix="R",
        on_attribute="zone",
    )
    joined_state = get_native_state(joined)

    assert joined.index.tolist() == [10, 11]
    assert tuple(joined.columns) == (
        "zone",
        "value_L",
        "nullable",
        "geometry",
        "right_id",
        "value_R",
        "label",
    )
    assert joined["right_id"].tolist() == [20, 21]
    assert joined["value_L"].tolist() == [1, 2]
    assert joined["value_R"].tolist() == [10, 20]
    assert joined["nullable"].isna().tolist() == [False, True]
    assert joined["label"].isna().tolist() == [False, True]
    assert joined_state is not None
    assert joined_state.index_plan.kind == "range"
    assert joined_state.column_order == tuple(joined.columns)
    assert NativeAttributeTable.from_value(joined_state.attributes).is_device_backed


def test_public_sjoin_nearest_transiently_lowers_owned_sources_with_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for transient nearest source lowering")
    pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")

    left_index = pd.RangeIndex(30, 32, name="left_id")
    right_index = pd.RangeIndex(40, 42, name="right_id")
    left = GeoDataFrame(
        {
            "zone": ["a", "b"],
            "value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (10 0)"],
                index=left_index,
                name="geometry",
            ),
        },
        index=left_index,
    )
    right = GeoDataFrame(
        {
            "zone": ["a", "b"],
            "value": [10, 20],
            "label": pd.array(["x", None], dtype="string"),
            "geometry": GeoSeries.from_wkt(
                ["POINT (1 0)", "POINT (12 0)"],
                index=right_index,
                name="geometry",
            ),
        },
        index=right_index,
    )
    assert get_native_state(left) is None
    assert get_native_state(right) is None

    def _fail_relation_host_export(self, **kwargs):
        raise AssertionError("transient nearest lowering must not export pair rows")

    monkeypatch.setattr(RelationIndexResult, "to_host", _fail_relation_host_export)
    joined = geopandas.sjoin_nearest(
        left,
        right,
        how="inner",
        lsuffix="L",
        rsuffix="R",
        distance_col="distance",
    )
    joined_state = get_native_state(joined)

    assert joined.index.tolist() == [30, 31]
    assert tuple(joined.columns) == (
        "zone_L",
        "value_L",
        "geometry",
        "right_id",
        "zone_R",
        "value_R",
        "label",
        "distance",
    )
    assert joined["right_id"].tolist() == [40, 41]
    assert np.issubdtype(joined["right_id"].dtype, np.integer)
    assert joined["label"].isna().tolist() == [False, True]
    assert joined["distance"].tolist() == [1.0, 2.0]
    assert joined_state is not None
    assert joined_state.index_plan.kind == "range"
    assert joined_state.column_order == tuple(joined.columns)
    assert NativeAttributeTable.from_value(joined_state.attributes).is_device_backed


def test_public_sjoin_transient_state_suffixes_colliding_output_index_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for transient relation index naming")
    pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")

    left_index = pd.Index(["a", "b"], name="myidx")
    left = GeoDataFrame(
        {
            "geometry": GeoSeries(
                [Point(1, 1), Point(2, 2)],
                index=left_index,
            )
        },
        index=left_index,
    )
    right = GeoDataFrame(
        {
            "myidx": [1, 2],
            "geometry": GeoSeries([Point(1, 1), Point(2, 2)]),
        }
    )

    def _fail_relation_host_export(self, **kwargs):
        raise AssertionError("native join index naming must not export relation pairs")

    monkeypatch.setattr(RelationIndexResult, "to_host", _fail_relation_host_export)
    joined = geopandas.sjoin(left, right, how="inner")
    joined_state = get_native_state(joined)

    assert joined.index.tolist() == ["a", "b"]
    assert joined.index.name == "myidx_left"
    assert tuple(joined.columns) == (
        "geometry",
        "index_right",
        "myidx_right",
    )
    assert joined_state is not None
    assert joined_state.index_plan.name == "myidx_left"
    assert joined_state.index_plan.to_public_index().name == "myidx_left"
    assert NativeAttributeTable.from_value(joined_state.attributes).is_device_backed


def test_public_sjoin_export_attaches_joined_private_native_state() -> None:
    left = GeoDataFrame(
        {
            "left_value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "score": [10.0, 5.0],
            "geometry": GeoSeries.from_wkt(
                [
                    "POLYGON ((-1 -1, 0.5 -1, 0.5 0.5, -1 0.5, -1 -1))",
                    "POLYGON ((0.5 0.5, 2 0.5, 2 2, 0.5 2, 0.5 0.5))",
                ],
                name="geometry",
            ),
        }
    )
    _attach_native_tabular_state(left)
    _attach_native_tabular_state(right)

    joined = sjoin(left, right, predicate="intersects")
    joined_state = get_native_state(joined)

    assert joined["left_value"].tolist() == [1, 2]
    assert joined["score"].tolist() == [10.0, 5.0]
    assert joined_state is not None
    assert joined_state.row_count == len(joined)
    assert joined_state.column_order == tuple(joined.columns)
    assert joined_state.to_native_tabular_result().attributes.to_pandas()[
        "left_value"
    ].tolist() == [1, 2]


def test_public_sjoin_keeps_candidate_refinement_device_resident(monkeypatch) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device sjoin relation probe")

    from vibespatial.spatial.query_types import _DeviceCandidates

    left = GeoDataFrame(
        {
            "left_value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "score": [10.0, 5.0],
            "geometry": GeoSeries.from_wkt(
                [
                    "POLYGON ((-1 -1, 0.5 -1, 0.5 0.5, -1 0.5, -1 -1))",
                    "POLYGON ((0.5 0.5, 2 0.5, 2 2, 0.5 2, 0.5 0.5))",
                ],
                name="geometry",
            ),
        }
    )
    left_state = _attach_owned_native_tabular_state(left)
    right_state = _attach_owned_native_tabular_state(right)
    left_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public sjoin left geometry device refinement",
    )
    right_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public sjoin right geometry device refinement",
    )

    def _fail_candidate_host_export(self):
        raise AssertionError(
            "public sjoin should refine device candidates before the final export boundary"
        )

    monkeypatch.setattr(_DeviceCandidates, "to_host", _fail_candidate_host_export)

    joined = sjoin(left, right, predicate="intersects")

    assert joined["left_value"].tolist() == [1, 2]
    assert joined["score"].tolist() == [10.0, 5.0]


def test_sjoin_builds_large_query_index_from_resident_geometry_on_device() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for resident sjoin index probe")

    from vibespatial.api.tools.sjoin import _native_spatial_index_for_sjoin_right

    right = GeoDataFrame(
        {
            "geometry": GeoSeries.from_wkt(
                [
                    "POLYGON ((0 0, 2 0, 1 1, 0 0))",
                    "POLYGON ((4 0, 7 0, 6 2, 4 0))",
                    "POLYGON ((9 1, 12 0, 11 3, 9 1))",
                ],
                name="geometry",
            ),
        }
    )
    right_state = _attach_owned_native_tabular_state(right)
    right_owned = right_state.geometry.owned
    right_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test resident sjoin index build",
    )

    native_index = _native_spatial_index_for_sjoin_right(
        right,
        right_state,
        right_owned,
        query_row_count=1_000_000,
    )

    assert native_index.metadata is not None
    assert hasattr(native_index.metadata.bounds, "__cuda_array_interface__")
    assert hasattr(native_index.morton_keys, "__cuda_array_interface__")
    assert hasattr(native_index.order, "__cuda_array_interface__")
    assert np.isfinite(np.asarray(native_index.total_bounds)).all()


def test_public_sjoin_lowers_nonempty_relation_without_internal_pair_export(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device sjoin relation export probe")
    pytest.importorskip("pylibcudf")

    left = GeoDataFrame(
        {
            "left_value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "score": [10.0, 5.0],
            "geometry": GeoSeries.from_wkt(
                [
                    "POLYGON ((-1 -1, 0.5 -1, 0.5 0.5, -1 0.5, -1 -1))",
                    "POLYGON ((0.5 0.5, 2 0.5, 2 2, 0.5 2, 0.5 0.5))",
                ],
                name="geometry",
            ),
        }
    )
    left_state = _attach_owned_native_tabular_state(left, attribute_storage="device")
    right_state = _attach_owned_native_tabular_state(right, attribute_storage="device")
    left_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public sjoin nonempty left geometry device export",
    )
    right_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public sjoin nonempty right geometry device export",
    )

    def _fail_relation_host_export(self, **kwargs):
        raise AssertionError(
            "nonempty relation joins should lower through device native state before terminal export"
        )

    monkeypatch.setattr(RelationIndexResult, "to_host", _fail_relation_host_export)
    clear_materialization_events()

    joined = sjoin(left, right, predicate="intersects")
    joined_state = get_native_state(joined)
    assert joined_state is not None
    joined_attributes = NativeAttributeTable.from_value(joined_state.attributes)
    assert joined_attributes.parts is not None
    assert all(part.device_table is not None for part in joined_attributes.parts)

    assert joined["left_value"].tolist() == [1, 2]
    assert joined["score"].tolist() == [10.0, 5.0]
    assert all(
        event.surface
        != "vibespatial.api._native_results._relation_join_export_result_to_native_tabular_result"
        for event in get_materialization_events(clear=True)
    )


def test_public_sjoin_small_relation_uses_native_frame_before_public_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device sjoin native export probe")
    pytest.importorskip("pylibcudf")

    left = GeoDataFrame(
        {
            "left_value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "score": [10.0, 5.0],
            "geometry": GeoSeries.from_wkt(
                [
                    "POLYGON ((-1 -1, 0.5 -1, 0.5 0.5, -1 0.5, -1 -1))",
                    "POLYGON ((0.5 0.5, 2 0.5, 2 2, 0.5 2, 0.5 0.5))",
                ],
                name="geometry",
            ),
        }
    )
    left_state = _attach_owned_native_tabular_state(left, attribute_storage="device")
    right_state = _attach_owned_native_tabular_state(right, attribute_storage="device")
    left_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public sjoin small relation left geometry device export",
    )
    right_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public sjoin small relation right geometry device export",
    )

    def _fail_relation_host_export(self, **kwargs):
        raise AssertionError(
            "small relation joins should still lower through device native state before public export"
        )

    monkeypatch.setattr(RelationIndexResult, "to_host", _fail_relation_host_export)
    clear_materialization_events()

    joined = sjoin(left, right, predicate="intersects")
    events = get_materialization_events(clear=True)
    joined_state = get_native_state(joined)

    assert joined["left_value"].tolist() == [1, 2]
    assert joined["score"].tolist() == [10.0, 5.0]
    assert joined_state is not None
    joined_attributes = NativeAttributeTable.from_value(joined_state.attributes)
    assert joined_attributes.parts is not None
    assert all(part.device_table is not None for part in joined_attributes.parts)
    assert all(
        event.surface
        != "vibespatial.api._native_results._relation_join_export_result_to_native_tabular_result"
        for event in events
    )
    assert all(event.surface != "vibespatial.api.NativeAttributeTable.to_arrow" for event in events)
    assert all(event.operation != "index_plan_to_host" for event in events)


def test_relation_join_public_export_keeps_device_relation_for_public_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device relation export guard")
    cp = pytest.importorskip("cupy")

    left, left_state = _native_backed_geodataframe()
    assert left_state.geometry.owned is None
    right = GeoDataFrame(
        {
            "score": [10.0, 5.0],
            "geometry": GeoSeries.from_wkt(
                ["POINT (2 2)", "POINT (3 3)"],
                name="geometry",
            ),
        }
    )
    right_state = _attach_owned_native_tabular_state(right, attribute_storage="device")
    right_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test relation join export right geometry",
    )
    export_result = RelationJoinExportResult(
        relation_result=RelationJoinResult(
            RelationIndexResult(
                cp.asarray([0, 1], dtype=cp.int32),
                cp.asarray([1, 0], dtype=cp.int32),
            ),
        ),
        left_df=left,
        right_df=right,
        how="inner",
        lsuffix="left",
        rsuffix="right",
    )
    to_host_calls = 0
    original_to_host = RelationIndexResult.to_host

    def _record_to_host(self, **kwargs):
        nonlocal to_host_calls
        to_host_calls += 1
        return original_to_host(self, **kwargs)

    monkeypatch.setattr(RelationIndexResult, "to_host", _record_to_host)
    clear_materialization_events()

    joined = export_result.to_geodataframe()

    relation_events = [
        event
        for event in get_materialization_events(clear=True)
        if event.surface
        == "vibespatial.api._native_results._relation_join_export_result_to_native_tabular_result"
    ]
    assert to_host_calls == 0
    assert relation_events == []
    assert joined["score"].tolist() == [5.0, 10.0]


def test_public_sjoin_device_relation_preserves_host_label_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device sjoin host-label index probe")
    pytest.importorskip("pylibcudf")

    left = GeoDataFrame(
        {
            "left_value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "score": [10.0, 5.0],
            "geometry": GeoSeries.from_wkt(
                [
                    "POLYGON ((-1 -1, 0.5 -1, 0.5 0.5, -1 0.5, -1 -1))",
                    "POLYGON ((0.5 0.5, 2 0.5, 2 2, 0.5 2, 0.5 0.5))",
                ],
                name="geometry",
            ),
        }
    )
    left.index = pd.Index([20, 10])
    right.index = pd.Index([300, 400])
    left_state = _attach_owned_native_tabular_state(left, attribute_storage="device")
    right_state = _attach_owned_native_tabular_state(right, attribute_storage="device")
    left_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public sjoin host-label left geometry device export",
    )
    right_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public sjoin host-label right geometry device export",
    )
    clear_materialization_events()

    joined = sjoin(left, right, predicate="intersects")

    index_events = [
        event
        for event in get_materialization_events(clear=True)
        if event.operation == "index_plan_take_positions_to_host"
    ]
    assert joined.index.tolist() == [20, 10]
    assert joined["index_right"].tolist() == [300, 400]
    assert index_events == []


def test_public_sjoin_unique_index_loc_bridge_preserves_device_rowflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device sjoin unique-index loc bridge")
    pytest.importorskip("pylibcudf")
    from vibespatial.api._native_public_arrays import NativeIndexLabelsArray
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    left = GeoDataFrame(
        {
            "left_value": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (5 5)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "score": [10.0, 5.0],
            "geometry": GeoSeries.from_wkt(
                [
                    "POLYGON ((-1 -1, 0.5 -1, 0.5 0.5, -1 0.5, -1 -1))",
                    "POLYGON ((-1 -1, 2 -1, 2 2, -1 2, -1 -1))",
                ],
                name="geometry",
            ),
        }
    )
    left_state = _attach_owned_native_tabular_state(left, attribute_storage="device")
    right_state = _attach_owned_native_tabular_state(right, attribute_storage="device")
    left_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public sjoin unique loc bridge left geometry",
    )
    right_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public sjoin unique loc bridge right geometry",
    )
    joined = sjoin(left, right, predicate="intersects")
    joined_state = get_native_state(joined)
    assert joined_state is not None
    assert joined_state.index_plan.selection_grouped is True

    import cupy as cp

    def _reject_global_unique(*_args, **_kwargs):
        raise AssertionError("grouped relation labels must use adjacent unique")

    monkeypatch.setattr(cp, "unique", _reject_global_unique)
    clear_materialization_events()
    reset_d2h_transfer_count()

    with assert_zero_d2h_transfers():
        candidate_rows = joined.index.unique()
        selected = left.loc[candidate_rows].copy()

    assert isinstance(getattr(candidate_rows, "array", None), NativeIndexLabelsArray)
    selected_state = get_native_state(selected)
    assert selected_state is not None
    assert NativeAttributeTable.from_value(selected_state.attributes).device_table is not None
    assert get_materialization_events(clear=True) == []
    assert selected["left_value"].tolist() == [1, 3]


def test_native_index_plan_roundtrip_preserves_selection_lineage() -> None:
    from vibespatial.api._native_public_arrays import native_public_index_from_plan
    from vibespatial.api._native_rowset import NativeIndexPlan

    source = NativeIndexPlan.from_index(pd.Index([10, 20, 30])).with_selection(
        np.asarray([0, 2], dtype=np.int64),
        source_token="source-token",
        source_row_count=3,
        unique=True,
    )
    selected = source.take(
        np.asarray([0, 2], dtype=np.int64),
        preserve_index=True,
        unique=True,
    ).with_selection(
        np.asarray([0, 2], dtype=np.int64),
        source_token="source-token",
        source_row_count=3,
        unique=True,
    )
    public = native_public_index_from_plan(selected)

    roundtrip = NativeIndexPlan.from_index(public)

    assert roundtrip.selection_source_token == "source-token"
    assert roundtrip.selection_source_row_count == 3
    assert np.array_equal(roundtrip.selection_positions, np.asarray([0, 2]))


def test_public_sjoin_empty_device_candidates_stay_device_resident(monkeypatch) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for empty device sjoin probe")

    from vibespatial.api._native_results import RelationIndexResult
    from vibespatial.spatial.query_types import _DeviceCandidates

    left = GeoDataFrame(
        {
            "left_value": [1],
            "geometry": GeoSeries.from_wkt(["POINT (100 100)"], name="geometry"),
        }
    )
    right = GeoDataFrame(
        {
            "score": [10.0],
            "geometry": GeoSeries.from_wkt(
                ["POLYGON ((-1 -1, 1 -1, 1 1, -1 1, -1 -1))"],
                name="geometry",
            ),
        }
    )
    left_state = _attach_owned_native_tabular_state(left)
    right_state = _attach_owned_native_tabular_state(right)
    left_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public sjoin empty left geometry device refinement",
    )
    right_state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public sjoin empty right geometry device refinement",
    )

    def _fail_candidate_host_export(self):
        raise AssertionError(
            "empty device candidate pairs should not cross to host before relation export"
        )

    def _fail_relation_host_export(self, **kwargs):
        raise AssertionError(
            "empty relation joins should assemble from known zero-cardinality shape"
        )

    monkeypatch.setattr(_DeviceCandidates, "to_host", _fail_candidate_host_export)
    monkeypatch.setattr(RelationIndexResult, "to_host", _fail_relation_host_export)

    joined = sjoin(left, right, predicate="intersects")

    assert joined.empty


def test_sjoin_owned_query_filters_empty_geometries_before_strict_strtree_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_fallback_events()
    left = GeoDataFrame(
        {"left_value": [1, 2]},
        geometry=[Point(0.0, 0.0), Point(5.0, 5.0)],
    )
    right = GeoDataFrame(
        {"score": [10.0]},
        geometry=[box(-1.0, -1.0, 1.0, 1.0)],
    )
    empty_left = GeoDataFrame(
        {"left_value": [3]},
        geometry=[GeometryCollection()],
        crs=left.crs,
    )

    left_join = sjoin(pd.concat([left, empty_left]), right, how="left")

    assert left_join.shape[0] == 3
    assert left_join["index_right"].iloc[0] == 0
    assert left_join["index_right"].iloc[1:].isna().tolist() == [True, True]
    assert get_fallback_events(clear=True) == []


def test_grouped_constructive_result_lowers_to_native_frame_without_public_export() -> None:
    group_index = pd.Index(["a", "b"], name="group")
    geometry = GeometryNativeResult.from_owned(
        from_shapely_geometries(
            [
                box(0.0, 0.0, 1.0, 1.0),
                box(2.0, 2.0, 3.0, 3.0),
            ],
        ),
        crs=None,
    )
    attributes = NativeAttributeTable(
        dataframe=pd.DataFrame({"value": [10, 20]}, index=group_index)
    )
    result = GroupedConstructiveResult(
        geometry=geometry,
        attributes=attributes,
        geometry_name="geometry",
        as_index=True,
    )
    clear_materialization_events()

    payload = result.to_native_tabular_result()
    state = result.to_native_frame_state()

    assert payload.geometry.row_count == 2
    assert payload.geometry_metadata is not None
    assert state.row_count == 2
    assert state.column_order == ("geometry", "value")
    assert state.index_plan.name == "group"
    assert state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        10,
        20,
    ]
    assert all(
        event.surface != "vibespatial.api.NativeTabularResult.to_geodataframe"
        for event in get_materialization_events(clear=True)
    )


def test_sjoin_attribute_reducer_bridge_declines_unadmitted_sources() -> None:
    left, _left_state = _native_backed_geodataframe()
    right = GeoDataFrame(
        {
            "label": ["a"],
            "geometry": GeoSeries.from_wkt(
                ["POLYGON ((-1 -1, 2 -1, 2 2, -1 2, -1 -1))"],
                name="geometry",
            ),
        }
    )
    _attach_native_tabular_state(right)

    export_result, _implementation, _execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
    )

    assert (
        export_result.left_reduce_right_numeric_columns(
            {"missing_sum": "missing"},
            {"missing_sum": "sum"},
        )
        is None
    )
    assert (
        export_result.left_reduce_right_numeric_columns(
            {"label_count": "label"},
            {"label_count": "count"},
        )
        is None
    )
    drop_native_state(right)
    assert (
        export_result.left_reduce_right_numeric_columns(
            {"label_count": "label"},
            {"label_count": "count"},
        )
        is None
    )


def test_relation_export_attribute_reducer_bridge_stays_device_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device relation attribute bridge probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    left = GeoDataFrame(
        {
            "geometry": GeoSeries.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ],
                name="geometry",
            ),
        }
    )
    right = GeoDataFrame(
        {
            "score": [10.0, 20.0, 30.0],
            "weight": [1.0, 2.0, 4.0],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)"],
                name="geometry",
            ),
        }
    )
    _attach_native_tabular_state(left, attribute_storage="arrow")
    _attach_native_tabular_state(right, attribute_storage="device")
    export_result = RelationJoinExportResult(
        relation_result=RelationJoinResult(
            RelationIndexResult(
                cp.asarray([0, 0, 2, 2, 2], dtype=cp.int32),
                cp.asarray([1, 2, 0, 1, 2], dtype=cp.int32),
            )
        ),
        left_df=left,
        right_df=right,
        how="inner",
        lsuffix="left",
        rsuffix="right",
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        reduced = export_result.left_reduce_right_numeric_columns(
            {
                "score_sum": "score",
                "match_count": "score",
                "weight_mean": "weight",
            },
            {
                "score_sum": "sum",
                "match_count": "count",
                "weight_mean": "mean",
            },
        )
        reduced_attributes = reduced.to_native_attribute_table()
        reduced_attribute_arrays = reduced_attributes.numeric_column_arrays(
            ["score_sum", "match_count", "weight_mean"]
        )

    assert reduced is not None
    assert reduced.is_device
    assert reduced_attributes.device_table is not None
    assert reduced_attribute_arrays is not None
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(reduced.columns["score_sum"].values).tolist() == [
        50.0,
        0.0,
        60.0,
        0.0,
    ]
    assert cp.asnumpy(reduced.columns["match_count"].values).tolist() == [2, 0, 3, 0]
    assert np.allclose(
        cp.asnumpy(reduced.columns["weight_mean"].values),
        np.asarray([3.0, np.nan, 7.0 / 3.0, np.nan]),
        equal_nan=True,
    )
    assert cp.asnumpy(reduced_attribute_arrays["score_sum"]).tolist() == [
        50.0,
        0.0,
        60.0,
        0.0,
    ]
    assert cp.asnumpy(reduced_attribute_arrays["match_count"]).tolist() == [
        2,
        0,
        3,
        0,
    ]
    reset_d2h_transfer_count()


def test_native_rowset_host_export_is_explicit() -> None:
    rowset = NativeRowSet.from_positions([2, 0, 1], source_token="frame")

    assert rowset.to_host_positions().tolist() == [2, 0, 1]


def test_native_index_plan_tracks_range_and_duplicate_indexes() -> None:
    range_plan = NativeIndexPlan.from_index(pd.RangeIndex(3))
    duplicate_plan = NativeIndexPlan.from_index(pd.Index(["a", "a", "b"], name="zone"))

    assert range_plan.kind == "range"
    assert range_plan.has_duplicates is False
    assert duplicate_plan.kind == "host-labels"
    assert duplicate_plan.name == "zone"
    assert duplicate_plan.has_duplicates is True


def test_native_index_plan_takes_range_index_labels_on_device() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device index-plan labels")
    cp = pytest.importorskip("cupy")
    clear_materialization_events()

    plan = NativeIndexPlan.from_index(pd.RangeIndex(start=10, stop=20, step=2, name="site"))
    taken = plan.take(cp.asarray([3, 0], dtype=cp.int32), unique=True)
    retaken = taken.take(cp.asarray([1], dtype=cp.int32), unique=True)

    assert taken.kind == "device-labels"
    assert taken.has_duplicates is False
    assert retaken.kind == "device-labels"
    assert get_materialization_events() == []
    public_index = taken.to_public_index()
    events = get_materialization_events(clear=True)
    assert public_index.tolist() == [16, 10]
    assert public_index.name == "site"
    assert len(events) == 1
    assert events[0].operation == "index_plan_to_host"


def test_strict_native_disallows_hidden_device_index_plan_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device index-plan labels")
    cp = pytest.importorskip("cupy")
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_materialization_events()

    plan = NativeIndexPlan.from_index(pd.RangeIndex(start=10, stop=20, step=2, name="site"))
    taken = plan.take(cp.asarray([3, 0], dtype=cp.int32), unique=True)

    with pytest.raises(StrictNativeMaterializationError):
        taken.to_public_index()

    events = get_materialization_events(clear=True)
    assert len(events) == 1
    assert events[0].operation == "index_plan_to_host"
    assert events[0].strict_disallowed is True


def test_native_index_plan_takes_host_labels_with_device_positions_explicitly() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device index-plan host label take")
    cp = pytest.importorskip("cupy")
    clear_materialization_events()

    plan = NativeIndexPlan.from_index(pd.Index([20, 10, 30], name="parcel"))

    taken = plan.take(
        cp.asarray([2, 0], dtype=cp.int32),
        unique=True,
        strict_disallowed=False,
    )

    events = get_materialization_events(clear=True)
    assert taken.kind == "host-labels-take"
    assert taken.index is None
    assert taken.name == "parcel"
    assert events == []

    public_index = taken.to_public_index(strict_disallowed=False)
    events = get_materialization_events(clear=True)

    assert public_index.tolist() == [30, 20]
    assert public_index.name == "parcel"
    assert len(events) == 1
    assert events[0].operation == "index_plan_take_positions_to_host"
    assert events[0].strict_disallowed is False


def test_strict_native_disallows_hidden_host_label_index_plan_take(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for strict device index-plan host label take")
    cp = pytest.importorskip("cupy")
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_materialization_events()

    plan = NativeIndexPlan.from_index(pd.Index([20, 10, 30], name="parcel"))

    taken = plan.take(cp.asarray([2, 0], dtype=cp.int32), unique=True)
    assert taken.kind == "host-labels-take"
    assert get_materialization_events(clear=True) == []

    with pytest.raises(StrictNativeMaterializationError):
        taken.to_public_index()

    events = get_materialization_events(clear=True)
    assert len(events) == 1
    assert events[0].operation == "index_plan_take_positions_to_host"
    assert events[0].strict_disallowed is True


def test_strict_native_defers_admitted_native_frame_index_bridge_until_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native frame index bridge")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    attributes = NativeAttributeTable(
        arrow_table=pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    )
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(2, 2)]).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test strict native admitted index bridge",
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=attributes,
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    rowset = NativeRowSet.from_positions(
        cp.asarray([2, 0], dtype=cp.int32),
        source_token=state.lineage_token,
        source_row_count=state.row_count,
        ordered=True,
        unique=True,
    )
    taken_state = state.take(rowset, preserve_index=True)
    clear_materialization_events()

    payload = taken_state.to_native_tabular_result()

    events = get_materialization_events(clear=True)
    assert payload.attributes.index.tolist() == [0, 1]
    assert payload.index_plan is not None
    assert payload.index_plan.kind == "device-labels"
    assert events == []

    exported = payload.to_geodataframe()
    assert exported.index.tolist() == [2, 0]
    events = get_materialization_events(clear=True)
    index_events = [event for event in events if event.operation == "index_plan_to_host"]
    assert len(index_events) == 1
    assert index_events[0].strict_disallowed is False


def test_strict_native_tabular_device_take_preserves_range_index_without_host_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native tabular take")
    cp = pytest.importorskip("cupy")
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    attributes = NativeAttributeTable(
        arrow_table=pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    )
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1), Point(2, 2)]).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test strict native tabular device take",
    )
    payload = NativeTabularResult(
        attributes=attributes,
        geometry=GeometryNativeResult.from_owned(owned, crs=None),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    clear_materialization_events()

    taken = payload.take(cp.asarray([2, 0], dtype=cp.int32), preserve_index=True)

    events = get_materialization_events(clear=True)
    assert taken.attributes.index.tolist() == [0, 1]
    assert taken.index_plan is not None
    assert taken.index_plan.kind == "device-labels"
    assert events == []

    exported = taken.to_geodataframe()
    assert exported.index.tolist() == [2, 0]
    index_events = [
        event
        for event in get_materialization_events(clear=True)
        if event.operation == "index_plan_to_host"
    ]
    assert len(index_events) == 1
    assert index_events[0].strict_disallowed is False


def test_simple_native_geodataframe_export_avoids_concat_and_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.Index([10, 20], name="site")
    attributes = NativeAttributeTable(
        dataframe=pd.DataFrame(
            {"value": [1, 2], "score": [3, 4]},
            index=index,
        )
    )
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1)])
    geometry = GeometryNativeResult.from_owned(owned, crs="EPSG:4326")
    frame = attributes.to_pandas(copy=False)
    frame.attrs["source"] = "native"

    def fail_geodataframe_init(self, *args, **kwargs):
        raise AssertionError("simple native export should not call GeoDataFrame()")

    monkeypatch.setattr(
        GeoDataFrame,
        "__init__",
        fail_geodataframe_init,
    )

    def fail_concat(*args, **kwargs):
        raise AssertionError("simple native export should not concatenate columns")

    monkeypatch.setattr(native_result_core_module.pd, "concat", fail_concat)

    exported = native_result_core_module._materialize_attribute_geometry_frame(
        attributes,
        (NativeGeometryColumn("geometry", geometry),),
        geometry_name="geometry",
        column_order=("value", "geometry", "score"),
    )

    assert isinstance(exported, GeoDataFrame)
    assert exported.columns.tolist() == ["value", "geometry", "score"]
    assert exported.index.equals(index)
    assert exported.geometry.name == "geometry"
    assert exported.geometry.to_wkt().tolist() == ["POINT (0 0)", "POINT (1 1)"]
    assert exported.crs == "EPSG:4326"
    assert getattr(exported.geometry.values, "_owned", None) is owned
    assert exported["score"].tolist() == [3, 4]
    assert exported.attrs == {"source": "native"}


def test_native_geometry_composition_exports_concrete_and_collection_rows() -> None:
    polygons = GeometryNativeResult.from_owned(
        from_shapely_geometries([box(0, 0, 1, 1), box(2, 0, 3, 1)]),
        crs=None,
    )
    remnants = GeometryNativeResult.from_owned(
        from_shapely_geometries(
            [LineString([(0, 0), (1, 0)]), Point(4, 4)],
        ),
        crs=None,
    )
    geometry = GeometryNativeResult.from_composition(
        NativeGeometryComposition(
            parts=(
                NativeGeometryCompositionPart(polygons, np.asarray([0, 1])),
                NativeGeometryCompositionPart(remnants, np.asarray([0, 2])),
            ),
            row_count=4,
            crs=None,
        ),
        crs=None,
    )

    exported = geometry.to_geoseries(index=pd.RangeIndex(4), name="geometry")

    assert exported.geom_type.iloc[:3].tolist() == [
        "GeometryCollection",
        "Polygon",
        "Point",
    ]
    assert pd.isna(exported.geom_type.iloc[3])
    assert [part.geom_type for part in exported.iloc[0].geoms] == [
        "Polygon",
        "LineString",
    ]


@pytest.mark.gpu
def test_native_geometry_composition_measurements_stay_device_resident() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native composition measurements")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import assert_zero_d2h_transfers

    polygons = GeometryNativeResult.from_owned(
        from_shapely_geometries([box(0, 0, 1, 1), box(2, 0, 3, 1)]).move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="unit test native composition polygon measurements",
        ),
        crs=None,
    )
    remnants = GeometryNativeResult.from_owned(
        from_shapely_geometries(
            [LineString([(0, 0), (1, 0)]), Point(4, 4)],
        ).move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="unit test native composition remnant measurements",
        ),
        crs=None,
    )
    geometry = GeometryNativeResult.from_composition(
        NativeGeometryComposition(
            parts=(
                NativeGeometryCompositionPart(
                    polygons,
                    cp.asarray([0, 1], dtype=cp.int64),
                ),
                NativeGeometryCompositionPart(
                    remnants,
                    cp.asarray([0, 2], dtype=cp.int64),
                ),
            ),
            row_count=4,
            crs=None,
        ),
        crs=None,
    )
    metadata = NativeGeometryMetadata.from_native_geometry(geometry)
    state = NativeTabularResult(
        attributes=NativeAttributeTable.from_value(
            pd.DataFrame({"value": [1, 2, 3, 4]})
        ),
        geometry=geometry,
        geometry_name="geometry",
        column_order=("value", "geometry"),
        geometry_metadata=metadata,
    ).to_native_frame_state()

    with assert_zero_d2h_transfers():
        area = state.geometry_area_expression()
        length = state.geometry_length_expression()
        selected_metadata = metadata.take(cp.asarray([0, 2], dtype=cp.int64))
        physicalized_metadata = selected_metadata.physicalize_logical_rows()

    assert metadata.bounds is None
    assert len(metadata.composition_parts) == 2
    assert selected_metadata.bounds is None
    assert selected_metadata.row_count == 2
    assert len(selected_metadata.composition_parts) == 2
    assert physicalized_metadata.composition_parts == ()
    np.testing.assert_allclose(
        cp.asnumpy(physicalized_metadata.bounds),
        [[0.0, 0.0, 1.0, 1.0], [4.0, 4.0, 4.0, 4.0]],
    )
    assert area.is_device
    assert length.is_device
    np.testing.assert_allclose(cp.asnumpy(area.values[:3]), [1.0, 1.0, 0.0])
    np.testing.assert_allclose(cp.asnumpy(length.values[:3]), [5.0, 4.0, 0.0])
    assert bool(cp.asnumpy(cp.isnan(area.values[3])))
    assert bool(cp.asnumpy(cp.isnan(length.values[3])))


def test_native_geometry_composition_preserves_typed_empty_fallback() -> None:
    empty_lines = GeometryNativeResult.from_owned(
        from_shapely_geometries([LineString(), LineString(), None]),
        crs=None,
    )
    points = GeometryNativeResult.from_owned(
        from_shapely_geometries([None, Point(1, 1), None]),
        crs=None,
    )
    geometry = GeometryNativeResult.from_composition(
        NativeGeometryComposition(
            parts=(
                NativeGeometryCompositionPart(empty_lines, np.asarray([0, 1, 2])),
                NativeGeometryCompositionPart(points, np.asarray([0, 1, 2])),
            ),
            row_count=3,
            crs=None,
        ),
        crs=None,
    )

    exported = geometry.to_geoseries(index=pd.RangeIndex(3), name="geometry")

    assert exported.iloc[0].geom_type == "LineString"
    assert exported.iloc[0].is_empty
    assert exported.iloc[1].equals(Point(1, 1))
    assert exported.iloc[2] is None


def test_native_ordered_geometry_collection_preserves_empty_slots() -> None:
    forward = from_shapely_geometries(
        [MultiLineString([[(0, 0), (1, 0)]]), MultiLineString(), None],
    )
    backward = from_shapely_geometries(
        [MultiLineString(), MultiLineString([[(2, 0), (1, 0)]]), None],
    )
    rows = np.arange(3, dtype=np.int64)

    geometry = _ordered_geometry_collection_from_owned_parts_at_capacity(
        ((forward, rows), (backward, rows)),
        row_count=3,
        crs=None,
    )

    assert geometry is not None
    assert geometry.composition is not None
    assert [part.collection_position for part in geometry.composition.parts] == [0, 1]
    exported = geometry.to_geoseries(index=pd.RangeIndex(3), name="geometry")
    assert [part.geom_type for part in exported.iloc[0].geoms] == [
        "MultiLineString",
        "MultiLineString",
    ]
    assert not exported.iloc[0].geoms[0].is_empty
    assert exported.iloc[0].geoms[1].is_empty
    assert exported.iloc[1].geoms[0].is_empty
    assert not exported.iloc[1].geoms[1].is_empty
    assert exported.iloc[2] is None


def test_native_geometry_composition_maps_parts_at_fixed_row_capacity() -> None:
    polygons = from_shapely_geometries(
        [box(0, 0, 1, 1), box(3, 0, 4, 1)],
    )
    lines = from_shapely_geometries(
        [LineString([(0, 0), (1, 0)])],
    )

    geometry = _geometry_composition_from_owned_parts_at_capacity(
        (
            (polygons, np.asarray([0, 2], dtype=np.int64)),
            (lines, np.asarray([0], dtype=np.int64)),
        ),
        row_count=4,
        crs=None,
    )

    assert geometry is not None
    assert geometry.row_count == 4
    assert [part.geometry.row_count for part in geometry.composition.parts] == [2, 1]
    exported = geometry.to_geoseries(index=pd.RangeIndex(4), name="geometry")
    assert exported.geom_type.iloc[0] == "GeometryCollection"
    assert pd.isna(exported.geom_type.iloc[1])
    assert exported.geom_type.iloc[2] == "Polygon"


def test_singular_device_composition_physicalizes_only_selected_spans() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for exact composition physicalization")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.geometry.owned import device_mask_owned_capacity

    owned = from_shapely_geometries(
        [box(0, 0, 1, 1), box(2, 0, 3, 1)],
    ).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test exact composition physicalization",
    )
    polygon_buffer = owned.device_state.families[GeometryFamily.POLYGON]
    polygon_buffer.x = cp.concatenate((polygon_buffer.x, cp.full(10_000, 999.0)))
    polygon_buffer.y = cp.concatenate((polygon_buffer.y, cp.full(10_000, 999.0)))
    first = device_mask_owned_capacity(owned, cp.asarray([True, False]))
    second = device_mask_owned_capacity(owned, cp.asarray([False, True]))
    rows = cp.arange(2, dtype=cp.int64)
    composition = NativeGeometryComposition(
        parts=(
            NativeGeometryCompositionPart(
                GeometryNativeResult.from_owned(first, crs=None),
                rows,
            ),
            NativeGeometryCompositionPart(
                GeometryNativeResult.from_owned(second, crs=None),
                rows,
            ),
        ),
        row_count=2,
        crs=None,
        trusted_all_ogc_valid=True,
    )

    reset_d2h_transfer_count()
    physical = composition._singular_owned_device()
    events = get_d2h_transfer_events(clear=True)

    assert physical is not None
    assert physical.is_indexed_view
    root = physical._base
    assert root is not None
    root_buffer = root.device_state.families[GeometryFamily.POLYGON]
    assert int(root_buffer.x.size) == 10
    assert int(root_buffer.y.size) == 10
    assert root_buffer.fixed_size.max_coord_count_per_row == 5
    assert physical.device_state.trusted_all_ogc_valid is True
    assert [
        geometry.equals(expected)
        for geometry, expected in zip(
            physical.to_shapely(),
            [box(0, 0, 1, 1), box(2, 0, 3, 1)],
            strict=True,
        )
    ] == [True, True]
    reasons = [event.reason for event in events]
    assert reasons.count(
        "native geometry composition exact physicalization allocation packet"
    ) == 1

    reset_d2h_transfer_count()
    cached = composition._singular_owned_device()

    assert cached is physical
    assert get_d2h_transfer_events(clear=True) == []

    taken = composition.take(cp.asarray([1], dtype=cp.int64), unique=True)
    assert taken._singular_owned_device() is taken._singular_owned_cache
    assert taken._singular_owned_cache.to_shapely()[0].equals(box(2, 0, 3, 1))

    frame = NativeTabularResult(
        attributes=pd.DataFrame({"value": [1, 2]}),
        geometry=GeometryNativeResult.from_composition(composition, crs=None),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    ).to_geodataframe()
    reset_d2h_transfer_count()
    clear_materialization_events()

    polygon_mask = frame.geometry.geom_type.isin({"Polygon", "MultiPolygon"})
    filtered = frame[polygon_mask].copy()

    filtered_state = get_native_state(filtered)
    assert filtered_state is not None
    filtered_owned = filtered_state.geometry.cached_owned()
    assert filtered_owned is not None
    assert set(filtered_owned.device_state.trusted_family_domain or ()) <= {
        GeometryFamily.POLYGON,
        GeometryFamily.MULTIPOLYGON,
    }
    assert filtered_owned.device_state.trusted_polygonal_only is True
    assert filtered_owned.device_state.trusted_all_valid is True
    assert get_materialization_events(clear=True) == []
    assert get_d2h_transfer_events(clear=True) == []


@pytest.mark.gpu
def test_singular_device_composition_keeps_partition_capacity_until_export() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device composition partitioning")
    cp = pytest.importorskip("cupy")
    from vibespatial.geometry.owned import device_mask_owned_capacity

    polygons = from_shapely_geometries(
        [box(0, 0, 1, 1), box(2, 0, 3, 1), Polygon(), box(6, 0, 7, 1)],
        residency=Residency.DEVICE,
    )
    points = from_shapely_geometries(
        [Point(10, 10), Point(1, 1), Point(12, 12), Point(13, 13)],
        residency=Residency.DEVICE,
    )
    rows = cp.arange(4, dtype=cp.int64)
    composition = NativeGeometryComposition(
        parts=(
            NativeGeometryCompositionPart(
                GeometryNativeResult.from_owned(
                    device_mask_owned_capacity(
                        polygons,
                        cp.asarray([True, False, True, False]),
                    ),
                    crs=None,
                ),
                rows,
            ),
            NativeGeometryCompositionPart(
                GeometryNativeResult.from_owned(
                    device_mask_owned_capacity(
                        points,
                        cp.asarray([False, True, False, False]),
                    ),
                    crs=None,
                ),
                rows,
            ),
        ),
        row_count=4,
        crs=None,
    )

    partitioned = composition._singular_partitioned_device()

    assert partitioned is not None
    assert partitioned.trusted_singular_rows
    assert [part.geometry.row_count for part in partitioned.parts] == [4, 4]
    assert all(not part.geometry.owned.is_indexed_view for part in partitioned.parts)
    exported = GeometryNativeResult.from_composition(
        partitioned,
        crs=None,
    ).to_geoseries(index=pd.RangeIndex(4), name="geometry")
    assert exported.geom_type.iloc[:3].tolist() == ["Polygon", "Point", "Polygon"]
    assert pd.isna(exported.geom_type.iloc[3])
    assert exported.is_empty.tolist() == [False, False, True, False]
    np.testing.assert_allclose(
        exported.bounds.to_numpy(),
        np.asarray(
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        ),
        equal_nan=True,
    )
    np.testing.assert_allclose(exported.total_bounds, [0.0, 0.0, 1.0, 1.0])
    assert exported.iloc[0].equals(box(0, 0, 1, 1))
    assert exported.iloc[1].equals(Point(1, 1))
    assert exported.iloc[2].is_empty
    assert exported.iloc[3] is None

    method_source = Path(
        "src/vibespatial/api/_native_result_core.py"
    ).read_text().split("    def _singular_partitioned_device", 1)[1].split(
        "\n    def _singular_owned_device",
        1,
    )[0]
    assert "flatnonzero" not in method_source
    assert "_device_indexed_take" not in method_source


def test_native_geometry_composition_take_preserves_duplicates_and_part_order() -> None:
    polygons = GeometryNativeResult.from_owned(
        from_shapely_geometries([box(0, 0, 1, 1), box(2, 0, 3, 1)]),
        crs=None,
    )
    lines = GeometryNativeResult.from_owned(
        from_shapely_geometries([LineString([(0, 0), (1, 0)])]),
        crs=None,
    )
    geometry = GeometryNativeResult.from_composition(
        NativeGeometryComposition(
            parts=(
                NativeGeometryCompositionPart(polygons, np.asarray([0, 1])),
                NativeGeometryCompositionPart(lines, np.asarray([0])),
            ),
            row_count=2,
            crs=None,
        ),
        crs=None,
    )

    taken = geometry.take(np.asarray([1, 0, 0], dtype=np.int64))
    exported = taken.to_geoseries(index=pd.RangeIndex(3), name="geometry")

    assert exported.geom_type.tolist() == [
        "Polygon",
        "GeometryCollection",
        "GeometryCollection",
    ]
    assert [part.geom_type for part in exported.iloc[1].geoms] == [
        "Polygon",
        "LineString",
    ]
    assert exported.iloc[1].equals(exported.iloc[2])


@pytest.mark.gpu
def test_native_geometry_composition_device_take_uses_relation_pairs() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device composition take")
    cp = pytest.importorskip("cupy")

    points = GeometryNativeResult.from_owned(
        from_shapely_geometries(
            [Point(0, 0), Point(1, 1), Point(2, 2)],
            residency=Residency.DEVICE,
        ),
        crs=None,
    )
    geometry = GeometryNativeResult.from_composition(
        NativeGeometryComposition(
            parts=(
                NativeGeometryCompositionPart(
                    points,
                    cp.asarray([0, 0, 1], dtype=cp.int64),
                ),
            ),
            row_count=2,
            crs=None,
        ),
        crs=None,
    )

    taken = geometry.take(cp.asarray([1, 0, 0], dtype=cp.int64))

    assert taken.composition is not None
    part = taken.composition.parts[0]
    assert cp.asarray(part.output_rows).tolist() == [0, 1, 1, 2, 2]
    assert part.geometry.owned is not None
    assert part.geometry.owned.is_indexed_view
    assert cp.asarray(part.geometry.owned._index_map).tolist() == [2, 0, 1, 0, 1]


@pytest.mark.gpu
def test_singular_device_composition_certifies_ordered_contiguous_parts_once() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device composition certification")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    first = from_shapely_geometries(
        [box(0, 0, 1, 1), box(2, 0, 3, 1)],
        residency=Residency.DEVICE,
    )
    second = from_shapely_geometries(
        [box(4, 0, 5, 1)],
        residency=Residency.DEVICE,
    )
    composition = NativeGeometryComposition(
        parts=(
            NativeGeometryCompositionPart(
                GeometryNativeResult.from_owned(first, crs=None),
                cp.asarray([0, 1], dtype=cp.int64),
            ),
            NativeGeometryCompositionPart(
                GeometryNativeResult.from_owned(second, crs=None),
                cp.asarray([2], dtype=cp.int64),
            ),
        ),
        row_count=3,
        crs=None,
        trusted_singular_rows=True,
    )

    reset_d2h_transfer_count()
    spans = composition.ordered_contiguous_device_parts()
    events = get_d2h_transfer_events(clear=True)

    assert spans is not None
    assert [(start, stop, owned.row_count) for start, stop, owned in spans] == [
        (0, 2, 2),
        (2, 3, 1),
    ]
    assert composition.contiguous_row_partitions
    assert [event.reason for event in events] == [
        "vibespatial.api.NativeGeometryComposition::ordered_contiguous_partition_certification"
    ]

    reset_d2h_transfer_count()
    assert composition.ordered_contiguous_device_parts() == spans
    assert get_d2h_transfer_events(clear=True) == []
    assert composition._singular_partitioned_device() is composition


@pytest.mark.gpu
def test_contiguous_device_composition_unique_take_keeps_one_row_indirected_root() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device composition take")
    cp = pytest.importorskip("cupy")

    first = GeometryNativeResult.from_owned(
        from_shapely_geometries(
            [box(0, 0, 1, 1), box(2, 0, 3, 1)],
            residency=Residency.DEVICE,
        ),
        crs=None,
    )
    second = GeometryNativeResult.from_owned(
        from_shapely_geometries(
            [box(4, 0, 5, 1), box(6, 0, 7, 1)],
            residency=Residency.DEVICE,
        ),
        crs=None,
    )
    geometry = GeometryNativeResult.from_composition(
        NativeGeometryComposition.concat([first, second], crs=None),
        crs=None,
    )

    taken = geometry.take(
        cp.asarray([3, 1], dtype=cp.int32),
        unique=True,
    )

    assert taken.composition is not None
    assert taken.composition.trusted_singular_rows
    assert taken.owned is None
    assert all(
        part.geometry.owned is not None
        and part.geometry.owned.row_count == 2
        and (
            part.geometry.owned._base is None
            or part.geometry.owned._base.row_count == 2
        )
        for part in taken.composition.parts
    )
    actual = np.asarray(
        taken.to_geoseries(index=pd.RangeIndex(2), name="geometry"),
        dtype=object,
    )
    assert actual[0].equals(box(6, 0, 7, 1))
    assert actual[1].equals(box(2, 0, 3, 1))


@pytest.mark.gpu
def test_native_geometry_composition_empty_device_take_drops_part_capacity() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device composition take")
    cp = pytest.importorskip("cupy")

    geometry = GeometryNativeResult.from_composition(
        NativeGeometryComposition(
            parts=(
                NativeGeometryCompositionPart(
                    GeometryNativeResult.from_owned(
                        from_shapely_geometries(
                            [box(0, 0, 1, 1), box(2, 0, 3, 1)],
                            residency=Residency.DEVICE,
                        ),
                        crs=None,
                    ),
                    cp.asarray([0, 1], dtype=cp.int64),
                ),
            ),
            row_count=2,
            crs=None,
        ),
        crs=None,
    )

    taken = geometry.take(cp.empty(0, dtype=cp.int64), unique=True)

    assert taken.composition is not None
    assert taken.composition.row_count == 0
    assert taken.composition.parts == ()
    exported = taken.to_geoseries(index=pd.RangeIndex(0), name="geometry")
    assert exported.empty


def test_native_geometry_composition_concat_stays_native_until_export() -> None:
    from vibespatial.api._native_results import _concat_geometry_result_sequence

    first = GeometryNativeResult.from_composition(
        NativeGeometryComposition(
            parts=(
                NativeGeometryCompositionPart(
                    GeometryNativeResult.from_owned(
                        from_shapely_geometries([Point(0, 0)]),
                        crs=None,
                    ),
                    np.asarray([0]),
                ),
            ),
            row_count=1,
            crs=None,
        ),
        crs=None,
    )
    second_owned = from_shapely_geometries([Point(1, 1), Point(2, 2)])
    second = GeometryNativeResult.from_owned(second_owned, crs=None)

    combined = _concat_geometry_result_sequence(
        [first, second],
        geometry_name="geometry",
        crs=None,
    )

    assert combined.composition is not None
    assert combined.owned is None
    assert combined.row_count == 3
    assert combined.composition.parts[-1].geometry.owned is second_owned
    exported = combined.to_geoseries(index=pd.RangeIndex(3), name="geometry")
    assert exported.to_wkt().tolist() == [
        "POINT (0 0)",
        "POINT (1 1)",
        "POINT (2 2)",
    ]


def test_ordered_geometry_collection_concat_preserves_row_local_assembly() -> None:
    from vibespatial.api._native_results import _concat_geometry_result_sequence

    ordered = _ordered_geometry_collection_from_owned_parts_at_capacity(
        (
            (from_shapely_geometries([MultiLineString()]), np.asarray([0])),
            (
                from_shapely_geometries(
                    [MultiLineString([[(1, 0), (0, 0)]])],
                ),
                np.asarray([0]),
            ),
        ),
        row_count=1,
        crs=None,
    )
    point = GeometryNativeResult.from_owned(
        from_shapely_geometries([Point(2, 2)]),
        crs=None,
    )

    combined = _concat_geometry_result_sequence(
        [ordered, point],
        geometry_name="geometry",
        crs=None,
    )
    exported = combined.to_geoseries(index=pd.RangeIndex(2), name="geometry")

    assert exported.iloc[0].geom_type == "GeometryCollection"
    assert len(exported.iloc[0].geoms) == 2
    assert exported.iloc[0].geoms[0].is_empty
    assert exported.iloc[0].geoms[1].geom_type == "MultiLineString"
    assert exported.iloc[1].equals(Point(2, 2))


def test_native_geometry_composition_tabular_take_and_export() -> None:
    geometry = GeometryNativeResult.from_composition(
        NativeGeometryComposition(
            parts=(
                NativeGeometryCompositionPart(
                    GeometryNativeResult.from_owned(
                        from_shapely_geometries(
                            [box(0, 0, 1, 1), box(2, 0, 3, 1)],
                        ),
                        crs=None,
                    ),
                    np.asarray([0, 1]),
                ),
                NativeGeometryCompositionPart(
                    GeometryNativeResult.from_owned(
                        from_shapely_geometries(
                            [LineString([(0, 0), (1, 0)])],
                        ),
                        crs=None,
                    ),
                    np.asarray([0]),
                ),
            ),
            row_count=2,
            crs=None,
        ),
        crs=None,
    )
    payload = NativeTabularResult(
        attributes=NativeAttributeTable(
            dataframe=pd.DataFrame({"value": [10, 20]}, index=[5, 6]),
        ),
        geometry=geometry,
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )

    taken = payload.take(np.asarray([1, 0]), preserve_index=True)
    exported = taken.to_geodataframe()

    assert exported.index.tolist() == [6, 5]
    assert exported["value"].tolist() == [20, 10]
    assert exported.geom_type.tolist() == ["Polygon", "GeometryCollection"]
    assert get_native_state(exported).residency is Residency.HOST


def test_native_frame_state_from_native_tabular_result() -> None:
    attrs = NativeAttributeTable(dataframe=pd.DataFrame({"value": [1, 2]}))
    geometry = GeometryNativeResult.from_geoseries(
        GeoSeries.from_wkt(["POINT (0 0)", "POINT (1 1)"], name="geometry")
    )
    secondary = GeometryNativeResult.from_geoseries(
        GeoSeries.from_wkt(["POINT (2 2)", "POINT (3 3)"], name="other_geometry")
    )
    result = NativeTabularResult(
        attributes=attrs,
        geometry=geometry,
        geometry_name="geometry",
        column_order=("value", "geometry", "other_geometry"),
        secondary_geometry=(NativeGeometryColumn("other_geometry", secondary),),
    )

    state = NativeFrameState.from_native_tabular_result(result)

    assert state.row_count == 2
    assert state.geometry_name == "geometry"
    assert state.column_order == ("value", "geometry", "other_geometry")
    assert len(state.to_native_tabular_result().secondary_geometry) == 1
    assert state.index_plan.kind == "range"


def test_public_metadata_export_helper_preserves_cached_geometry_metadata() -> None:
    from vibespatial.api.geodataframe import _native_tabular_result_with_public_metadata

    gdf = GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1)])
    metadata = NativeGeometryMetadata.from_owned(owned, source_token="frame")
    result = NativeTabularResult(
        attributes=NativeAttributeTable(dataframe=gdf.drop(columns=["geometry"])),
        geometry=GeometryNativeResult.from_owned(owned, crs=None),
        geometry_name="geometry",
        column_order=("value", "geometry"),
        geometry_metadata=metadata,
    )
    state = NativeFrameState.from_native_tabular_result(result)
    gdf.attrs["source"] = "public"

    payload = _native_tabular_result_with_public_metadata(gdf, state)

    assert payload.attrs == {"source": "public"}
    assert payload.geometry_metadata is metadata


def _native_export_test_frame() -> GeoDataFrame:
    attrs = NativeAttributeTable(dataframe=pd.DataFrame({"value": [1, 2]}))
    geometry = GeometryNativeResult.from_geoseries(
        GeoSeries.from_wkt(["POINT (0 0)", "POINT (1 1)"], name="geometry")
    )
    result = NativeTabularResult(
        attributes=attrs,
        geometry=geometry,
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    return result.to_geodataframe()


def test_native_tabular_geodataframe_materialization_records_storage_detail() -> None:
    clear_materialization_events()
    materialized = _native_export_test_frame()

    event = get_materialization_events(clear=True)[0]
    state = get_native_state(materialized)

    assert len(materialized) == 2
    assert state is not None
    assert state.row_count == 2
    assert state.column_order == ("value", "geometry")
    assert "native_export_target=geodataframe" in event.detail
    assert "rows=2" in event.detail
    assert "attribute_columns=1" in event.detail
    assert "attribute_storage=pandas" in event.detail
    assert "geometry_storage=geoseries" in event.detail


def test_native_geodataframe_repr_records_export_boundary_contract() -> None:
    frame = _native_export_test_frame()

    clear_materialization_events()
    text = repr(frame)
    events = get_materialization_events(clear=True)
    repr_events = [event for event in events if event.operation == "geodataframe_repr"]

    assert "POINT" in text
    assert len(repr_events) == 1
    assert repr_events[0].surface == "vibespatial.api.GeoDataFrame.__repr__"
    assert "native_export_target=repr" in repr_events[0].detail
    assert "rows=2" in repr_events[0].detail
    assert "state_residency=" in repr_events[0].detail


def test_native_geodataframe_html_repr_records_export_boundary_contract() -> None:
    frame = _native_export_test_frame()

    clear_materialization_events()
    html = frame._repr_html_()
    events = get_materialization_events(clear=True)
    html_events = [event for event in events if event.operation == "geodataframe_html_repr"]

    assert "geometry" in html
    assert len(html_events) == 1
    assert html_events[0].surface == "vibespatial.api.GeoDataFrame._repr_html_"
    assert "native_export_target=html-repr" in html_events[0].detail
    assert "rows=2" in html_events[0].detail


@pytest.mark.parametrize(
    ("method_name", "operation", "target"),
    [
        ("to_json", "geodataframe_to_json", "geojson"),
        ("to_geo_dict", "geodataframe_to_geo_dict", "geo-dict"),
        ("__geo_interface__", "geodataframe_geo_interface", "geo-interface"),
        ("to_wkb", "geodataframe_to_wkb", "wkb-dataframe"),
        ("to_wkt", "geodataframe_to_wkt", "wkt-dataframe"),
    ],
)
def test_native_geodataframe_terminal_exports_record_boundary_contract(
    method_name: str,
    operation: str,
    target: str,
) -> None:
    frame = _native_export_test_frame()

    clear_materialization_events()
    if method_name == "__geo_interface__":
        exported = frame.__geo_interface__
    else:
        exported = getattr(frame, method_name)()
    events = get_materialization_events(clear=True)
    export_events = [event for event in events if event.operation == operation]

    assert exported is not None
    assert len(export_events) == 1
    assert f"native_export_target={target}" in export_events[0].detail
    assert "rows=2" in export_events[0].detail


def test_native_geoseries_repr_records_export_boundary_contract() -> None:
    owned = from_shapely_geometries([Point(0, 0)])
    series = GeoSeries(GeometryArray.from_owned(owned), name="geometry")

    clear_materialization_events()
    text = repr(series)
    events = get_materialization_events(clear=True)
    repr_events = [event for event in events if event.operation == "geoseries_repr"]

    assert "POINT" in text
    assert len(repr_events) == 1
    assert repr_events[0].surface == "vibespatial.api.GeoSeries.__repr__"
    assert "native_export_target=repr" in repr_events[0].detail
    assert "rows=1" in repr_events[0].detail
    assert "geometry_residency=" in repr_events[0].detail


@pytest.mark.parametrize(
    ("method_name", "operation", "target"),
    [
        ("to_json", "geoseries_to_json", "geojson"),
        ("__geo_interface__", "geoseries_geo_interface", "geo-interface"),
        ("to_arrow", "geoseries_to_arrow", "arrow"),
        ("to_wkb", "geoseries_to_wkb", "wkb-series"),
        ("to_wkt", "geoseries_to_wkt", "wkt-series"),
    ],
)
def test_native_geoseries_terminal_exports_record_boundary_contract(
    method_name: str,
    operation: str,
    target: str,
) -> None:
    owned = from_shapely_geometries([Point(0, 0)])
    series = GeoSeries(GeometryArray.from_owned(owned), name="geometry")

    clear_materialization_events()
    if method_name == "__geo_interface__":
        exported = series.__geo_interface__
    else:
        exported = getattr(series, method_name)()
    events = get_materialization_events(clear=True)
    export_events = [event for event in events if event.operation == operation]

    assert exported is not None
    assert len(export_events) == 1
    assert f"native_export_target={target}" in export_events[0].detail
    assert "rows=1" in export_events[0].detail


def test_native_geometry_geodataframe_arrow_export_records_boundary_without_frame_state() -> None:
    owned = from_shapely_geometries([Point(0, 0)])
    series = GeoSeries(GeometryArray.from_owned(owned), name="geometry")
    frame = GeoDataFrame({"geometry": series})
    assert get_native_state(frame) is None

    clear_materialization_events()
    exported = frame.to_arrow()
    events = get_materialization_events(clear=True)
    export_events = [event for event in events if event.operation == "geodataframe_to_arrow"]

    assert exported is not None
    assert len(export_events) == 1
    assert "native_export_target=arrow" in export_events[0].detail
    assert "rows=1" in export_events[0].detail


def test_native_geometry_metric_properties_record_public_series_export_boundary() -> None:
    owned = from_shapely_geometries([box(0, 0, 2, 3)])
    series = GeoSeries(GeometryArray.from_owned(owned), name="geometry")
    frame = GeoDataFrame({"geometry": series})

    clear_materialization_events()
    area = frame.area
    events = get_materialization_events(clear=True)
    area_events = [event for event in events if event.operation == "geodataframe_area"]

    assert area.tolist() == [6.0]
    assert len(area_events) == 1
    assert "native_export_target=series" in area_events[0].detail
    assert "rows=1" in area_events[0].detail

    clear_materialization_events()
    length = series.length
    events = get_materialization_events(clear=True)
    length_events = [event for event in events if event.operation == "geoseries_length"]

    assert length.tolist() == [10.0]
    assert len(length_events) == 1
    assert "native_export_target=series" in length_events[0].detail
    assert "rows=1" in length_events[0].detail


def test_native_geometry_bounds_properties_record_public_export_boundaries() -> None:
    owned = from_shapely_geometries([box(1, 2, 3, 5)])
    series = GeoSeries(GeometryArray.from_owned(owned), name="geometry")
    frame = GeoDataFrame({"geometry": series})

    clear_materialization_events()
    bounds = frame.bounds
    events = get_materialization_events(clear=True)
    bounds_events = [event for event in events if event.operation == "geodataframe_bounds"]

    assert bounds.iloc[0].tolist() == [1.0, 2.0, 3.0, 5.0]
    assert len(bounds_events) == 1
    assert "native_export_target=dataframe" in bounds_events[0].detail
    assert "rows=1" in bounds_events[0].detail

    clear_materialization_events()
    total_bounds = series.total_bounds
    events = get_materialization_events(clear=True)
    total_bounds_events = [event for event in events if event.operation == "geoseries_total_bounds"]

    assert total_bounds.tolist() == [1.0, 2.0, 3.0, 5.0]
    assert len(total_bounds_events) == 1
    assert "native_export_target=numpy" in total_bounds_events[0].detail
    assert "rows=1" in total_bounds_events[0].detail


def test_native_geometry_array_numpy_protocol_records_export_boundary() -> None:
    owned = from_shapely_geometries([Point(0, 0)])

    clear_materialization_events()
    host_values = np.asarray(GeometryArray.from_owned(owned))
    events = get_materialization_events(clear=True)
    geometry_array_events = [
        event for event in events if event.operation == "geometryarray_to_numpy"
    ]

    assert str(host_values[0]) == "POINT (0 0)"
    assert len(geometry_array_events) == 1
    assert "native_export_target=numpy" in geometry_array_events[0].detail
    assert "rows=1" in geometry_array_events[0].detail

    clear_materialization_events()
    device_values = np.asarray(DeviceGeometryArray._from_owned(owned))
    events = get_materialization_events(clear=True)
    device_array_events = [
        event for event in events if event.operation == "device_geometryarray_to_numpy"
    ]

    assert str(device_values[0]) == "POINT (0 0)"
    assert len(device_array_events) == 1
    assert "native_export_target=numpy" in device_array_events[0].detail
    assert "rows=1" in device_array_events[0].detail


def test_native_geometry_binary_predicate_records_public_series_export_boundary() -> None:
    owned = from_shapely_geometries([Point(0, 0), Point(2, 2)])
    series = GeoSeries(GeometryArray.from_owned(owned), name="geometry")
    if has_gpu_runtime():
        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )

        reset_d2h_transfer_count()
        get_d2h_transfer_events(clear=True)

    clear_materialization_events()
    result = series.intersects(series)
    events = get_materialization_events(clear=True)
    predicate_events = [event for event in events if event.operation == "geoseries_intersects"]
    if has_gpu_runtime():
        d2h_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
        assert "geometry analysis mixed row-bounds host export" not in d2h_reasons

    assert result.tolist() == [True, True]
    assert len(predicate_events) == 1
    assert "native_export_target=series" in predicate_events[0].detail
    assert "rows=2" in predicate_events[0].detail


@pytest.mark.parametrize(
    ("method_name", "operation", "target", "suffix"),
    [
        ("to_parquet", "geodataframe_to_parquet", "geoparquet", "parquet"),
        ("to_feather", "geodataframe_to_feather", "feather", "feather"),
    ],
)
def test_native_geometry_geodataframe_file_exports_record_boundary_without_frame_state(
    tmp_path,
    method_name: str,
    operation: str,
    target: str,
    suffix: str,
) -> None:
    owned = from_shapely_geometries([Point(0, 0)])
    series = GeoSeries(GeometryArray.from_owned(owned), name="geometry")
    frame = GeoDataFrame({"geometry": series})
    assert get_native_state(frame) is None

    path = tmp_path / f"native-export.{suffix}"
    clear_materialization_events()
    getattr(frame, method_name)(path)
    events = get_materialization_events(clear=True)
    export_events = [event for event in events if event.operation == operation]

    assert path.exists()
    assert len(export_events) == 1
    assert f"native_export_target={target}" in export_events[0].detail
    assert "rows=1" in export_events[0].detail


def test_native_geoseries_file_export_records_boundary_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    owned = from_shapely_geometries([Point(0, 0)])
    series = GeoSeries(DeviceGeometryArray._from_owned(owned), name="geometry")
    captured = {}

    def _capture_write_vector_file(df, filename, driver=None, schema=None, index=None, **kwargs):
        captured["filename"] = filename
        captured["driver"] = driver
        captured["index"] = index
        captured["kwargs"] = kwargs
        captured["geometry_dtype"] = getattr(df.geometry.values.dtype, "name", None)

    monkeypatch.setattr("vibespatial.io.file.write_vector_file", _capture_write_vector_file)

    clear_materialization_events()
    series.to_file(tmp_path / "series.geojson", driver="GeoJSON", index=False)
    events = get_materialization_events(clear=True)
    export_events = [event for event in events if event.operation == "geoseries_to_file"]

    assert len(export_events) == 1
    assert "native_export_target=file" in export_events[0].detail
    assert "rows=1" in export_events[0].detail
    assert [event.operation for event in events] == ["geoseries_to_file"]
    assert captured["driver"] == "GeoJSON"
    assert captured["index"] is False
    assert captured["geometry_dtype"] == "device_geometry"


def test_native_tabular_arrow_export_records_boundary_contract() -> None:
    clear_materialization_events()
    attrs = NativeAttributeTable(dataframe=pd.DataFrame({"value": [1, 2]}))
    geometry = GeometryNativeResult.from_geoseries(
        GeoSeries.from_wkt(["POINT (0 0)", "POINT (1 1)"], name="geometry")
    )
    result = NativeTabularResult(
        attributes=attrs,
        geometry=geometry,
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )

    result.to_arrow()
    events = get_materialization_events(clear=True)
    arrow_events = [event for event in events if event.operation == "native_tabular_to_arrow"]

    assert len(arrow_events) == 1
    assert "native_export_target=arrow" in arrow_events[0].detail
    assert "rows=2" in arrow_events[0].detail
    assert "attribute_columns=1" in arrow_events[0].detail


def test_owned_geometry_shapely_export_records_boundary_contract() -> None:
    clear_materialization_events()
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1)])

    got = owned.to_shapely()
    events = get_materialization_events(clear=True)
    shapely_events = [event for event in events if event.operation == "owned_geometry_to_shapely"]

    assert len(got) == 2
    assert len(shapely_events) == 1
    assert "native_export_target=shapely" in shapely_events[0].detail
    assert "rows=2" in shapely_events[0].detail


def test_native_state_registry_rejects_stale_handles() -> None:
    owner = _Owner()
    attrs = NativeAttributeTable(dataframe=pd.DataFrame({"value": [1]}))
    geometry = GeometryNativeResult.from_geoseries(
        GeoSeries.from_wkt(["POINT (0 0)"], name="geometry")
    )
    result = NativeTabularResult(
        attributes=attrs,
        geometry=geometry,
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    state = NativeFrameState.from_native_tabular_result(result)
    handle = attach_native_state(owner, state)

    assert get_native_state(owner, handle) is state
    drop_native_state(owner)
    assert get_native_state(owner, handle) is None


def _native_backed_geodataframe() -> tuple[GeoDataFrame, NativeFrameState]:
    gdf = GeoDataFrame(
        {
            "value": [1, 2],
            "name": ["a", "b"],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    attrs = NativeAttributeTable(dataframe=gdf.drop(columns=["geometry"]))
    geometry = GeometryNativeResult.from_geoseries(gdf.geometry)
    result = NativeTabularResult(
        attributes=attrs,
        geometry=geometry,
        geometry_name="geometry",
        column_order=("value", "name", "geometry"),
    )
    state = NativeFrameState.from_native_tabular_result(result)
    attach_native_state(gdf, state)
    return gdf, state


def _native_backed_geodataframe_only() -> GeoDataFrame:
    gdf, _state = _native_backed_geodataframe()
    return gdf


def _device_native_expression_geodataframe(
    values: list[int] | None = None,
) -> GeoDataFrame:
    plc = pytest.importorskip("pylibcudf")
    source_values = [1, 2, 3] if values is None else values
    owned = from_shapely_geometries(
        [Point(position, position) for position in range(len(source_values))],
        residency=Residency.DEVICE,
    )
    arrow = pa.table(
        {
            "value": pa.array(source_values, type=pa.int64()),
            "other": pa.array(
                [position + 10 for position in range(len(source_values))],
                type=pa.int64(),
            ),
        }
    )
    index = pd.Index(
        [100 + position for position in range(len(source_values))],
        name="source_id",
    )
    gdf = GeoDataFrame(
        {
            "value": source_values,
            "other": [position + 10 for position in range(len(source_values))],
            "geometry": GeoSeries(
                DeviceGeometryArray._from_owned(owned),
                index=index,
                name="geometry",
            ),
        },
        index=index,
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(arrow),
                index_override=index,
                column_override=tuple(arrow.column_names),
                schema_override=arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "other", "geometry"),
            index_plan=NativeIndexPlan.from_index(index),
        )
    )
    attach_native_state(gdf, state)
    return gdf


def _native_backed_nullable_geodataframe() -> GeoDataFrame:
    gdf = GeoDataFrame(
        {
            "value": [1.0, None],
            "name": ["a", "b"],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    _attach_native_tabular_state(gdf)
    return gdf


def _two_geometry_geodataframe() -> GeoDataFrame:
    return GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
            "other_geometry": GeoSeries.from_wkt(
                ["POINT (10 10)", "POINT (20 20)"],
                name="other_geometry",
            ),
        },
        geometry="geometry",
    )


def _attach_two_geometry_native_tabular_state(
    gdf: GeoDataFrame,
    *,
    owned: bool = False,
    attribute_storage: str = "pandas",
) -> NativeFrameState:
    non_geometry = gdf.drop(columns=["geometry", "other_geometry"])
    if attribute_storage == "pandas":
        attributes = NativeAttributeTable(dataframe=non_geometry)
    elif attribute_storage == "arrow":
        attributes = NativeAttributeTable(
            arrow_table=pa.Table.from_pandas(non_geometry, preserve_index=False),
            index_override=gdf.index,
            column_override=tuple(non_geometry.columns),
        )
    elif attribute_storage == "device":
        import pylibcudf as plc

        arrow_table = pa.Table.from_pandas(non_geometry, preserve_index=False)
        attributes = NativeAttributeTable(
            device_table=plc.Table.from_arrow(arrow_table),
            index_override=gdf.index,
            column_override=tuple(non_geometry.columns),
            schema_override=arrow_table.schema,
        )
    else:
        raise ValueError(f"unsupported attribute storage {attribute_storage!r}")

    if owned:
        primary = GeometryNativeResult.from_owned(
            from_shapely_geometries(list(gdf["geometry"])),
            crs=gdf["geometry"].crs,
        )
        secondary = GeometryNativeResult.from_owned(
            from_shapely_geometries(list(gdf["other_geometry"])),
            crs=gdf["other_geometry"].crs,
        )
    else:
        primary = GeometryNativeResult.from_geoseries(gdf["geometry"])
        secondary = GeometryNativeResult.from_geoseries(gdf["other_geometry"])

    result = NativeTabularResult(
        attributes=attributes,
        geometry=primary,
        geometry_name="geometry",
        column_order=("value", "geometry", "other_geometry"),
        secondary_geometry=(NativeGeometryColumn("other_geometry", secondary),),
    )
    state = NativeFrameState.from_native_tabular_result(result)
    attach_native_state(gdf, state)
    return state


def _attach_native_tabular_state(
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
    elif attribute_storage == "device":
        pytest.importorskip("pylibcudf")
        import pylibcudf as plc

        arrow_table = pa.Table.from_pandas(non_geometry, preserve_index=False)
        attributes = NativeAttributeTable(
            device_table=plc.Table.from_arrow(arrow_table),
            index_override=gdf.index,
            column_override=tuple(non_geometry.columns),
            schema_override=arrow_table.schema,
        )
    else:
        raise ValueError(f"unsupported attribute storage {attribute_storage!r}")
    result = NativeTabularResult(
        attributes=attributes,
        geometry=GeometryNativeResult.from_geoseries(gdf.geometry),
        geometry_name=gdf._geometry_column_name,
        column_order=tuple(gdf.columns),
    )
    state = NativeFrameState.from_native_tabular_result(result)
    attach_native_state(gdf, state)
    return state


def _attach_owned_native_tabular_state(
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
    elif attribute_storage == "device":
        pytest.importorskip("pylibcudf")
        import pylibcudf as plc

        arrow_table = pa.Table.from_pandas(non_geometry, preserve_index=False)
        attributes = NativeAttributeTable(
            device_table=plc.Table.from_arrow(arrow_table),
            index_override=gdf.index,
            column_override=tuple(non_geometry.columns),
            schema_override=arrow_table.schema,
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


def test_geodataframe_shallow_copy_preserves_private_native_state() -> None:
    gdf, state = _native_backed_geodataframe()

    copied = gdf.copy(deep=False)

    assert get_native_state(copied) is state


def test_geodataframe_deep_copy_preserves_valid_private_native_state() -> None:
    gdf, state = _native_backed_geodataframe()

    copied = gdf.copy(deep=True)

    assert copied is not gdf
    assert copied.dtypes.equals(gdf.dtypes)
    assert get_native_state(copied) is state

    copied["new_value"] = [3, 4]

    copied_state = get_native_state(copied)
    assert copied_state is not None
    assert copied_state is not state
    assert copied_state.column_order == ("value", "name", "geometry", "new_value")
    assert get_native_state(gdf) is state


def test_geodataframe_deep_copy_native_fast_path_bypasses_public_setitem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gdf, state = _native_backed_geodataframe()

    def _fail_public_setitem(self, key, value):
        raise AssertionError("native GeoDataFrame.copy should not re-enter __setitem__")

    monkeypatch.setattr(GeoDataFrame, "__setitem__", _fail_public_setitem)

    copied = gdf.copy(deep=True)

    assert copied is not gdf
    assert copied["value"].tolist() == [1, 2]
    assert copied.geometry.tolist() == gdf.geometry.tolist()
    assert get_native_state(copied) is state


def test_geodataframe_deep_copy_rewraps_device_geometry_without_cloning_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device-backed copy fast path")

    owned = from_shapely_geometries(
        [box(0, 0, 1, 1), box(2, 2, 3, 3)],
        residency=Residency.DEVICE,
    )
    gdf = GeoDataFrame(
        {"value": [1, 2]},
        geometry=GeoSeries(
            DeviceGeometryArray._from_owned(owned),
            name="geometry",
        ),
    )
    source_owned = gdf.geometry.values._owned
    attrs = NativeAttributeTable(dataframe=gdf.drop(columns=["geometry"]))
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=attrs,
            geometry=GeometryNativeResult.from_owned(source_owned, crs=gdf.crs),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    attach_native_state(gdf, state)

    def _fail_device_copy(self):
        raise AssertionError("native GeoDataFrame.copy should not clone device geometry")

    monkeypatch.setattr(DeviceGeometryArray, "copy", _fail_device_copy)

    copied = gdf.copy(deep=True)

    assert copied is not gdf
    assert get_native_state(copied) is state
    assert copied.geometry.values is not gdf.geometry.values
    assert copied.geometry.values._owned is source_owned

    copied.loc[0, "value"] = 99
    assert int(gdf.loc[0, "value"]) == 1


def test_geodataframe_projection_preserves_projected_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    projected = gdf[["name", "geometry"]]
    projected_state = get_native_state(projected)

    assert projected_state is not None
    assert projected_state.column_order == ("name", "geometry")
    table = pa.table(projected.to_arrow())
    assert "name" in table.column_names
    assert "value" not in table.column_names


def test_geodataframe_device_attribute_projection_preserves_private_native_state_without_runtime_d2h() -> (
    None
):
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device attribute projection")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    gdf = GeoDataFrame(
        {
            "score": [1.5, 2.5, 3.5],
            "rows": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)"],
                name="geometry",
            ),
        }
    )
    _attach_native_tabular_state(gdf, attribute_storage="device")
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        projected = gdf[["rows", "geometry"]]
        projected_state = get_native_state(projected)
        arrays = projected_state.attributes.numeric_column_arrays(("rows",))

    assert projected_state is not None
    assert projected_state.column_order == ("rows", "geometry")
    assert projected_state.attributes.device_table is not None
    assert arrays is not None
    assert cp.asnumpy(arrays["rows"]).tolist() == [1, 2, 3]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_geodataframe_boolean_filter_preserves_range_index_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()
    clear_materialization_events()

    filtered = gdf[gdf["value"] > 1]
    filtered_state = get_native_state(filtered)

    assert len(filtered) == 1
    assert filtered.index.tolist() == [1]
    assert filtered_state is not None
    assert filtered_state.row_count == 1
    assert filtered_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [2]
    assert get_materialization_events(clear=True) == []


def test_geodataframe_all_true_boolean_filter_reuses_private_native_state() -> None:
    gdf, state = _native_backed_geodataframe()
    clear_materialization_events()

    filtered = gdf[pd.Series([True, True], index=gdf.index)]
    filtered_state = get_native_state(filtered)

    assert filtered.index.equals(gdf.index)
    assert filtered_state is state
    assert get_materialization_events(clear=True) == []


def test_geodataframe_boolean_filter_without_sidecar_uses_native_before_pandas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)"],
                name="geometry",
            ),
        }
    )
    _attach_owned_native_tabular_state(gdf, attribute_storage="arrow")
    mask = pd.Series([True, False, True], index=gdf.index)
    clear_materialization_events()

    original_getitem = pd.DataFrame.__getitem__

    def _fail_pandas_boolean_filter(self, key):
        if isinstance(key, pd.Series) and pd.api.types.is_bool_dtype(key.dtype):
            raise AssertionError("native boolean filter should run before pandas")
        return original_getitem(self, key)

    monkeypatch.setattr(pd.DataFrame, "__getitem__", _fail_pandas_boolean_filter)

    filtered = gdf[mask]
    filtered_state = get_native_state(filtered)

    assert filtered.index.tolist() == [0, 2]
    assert filtered["value"].tolist() == [1, 3]
    assert filtered_state is not None
    assert filtered_state.row_count == 2
    assert filtered_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        3,
    ]


def test_geodataframe_boolean_filter_unique_index_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        },
        index=pd.Index(["a", "b"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    filtered = gdf[pd.Series([True, False], index=gdf.index)]
    filtered_state = get_native_state(filtered)

    assert filtered.index.tolist() == ["a"]
    assert filtered_state is not None
    assert filtered_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [1]


def test_native_frame_state_take_validates_rowset_source() -> None:
    _gdf, state = _native_backed_geodataframe()

    rowset = NativeRowSet.from_positions(
        [1],
        source_token=state.lineage_token,
        source_row_count=state.row_count,
    )
    taken = state.take(rowset)

    assert taken.row_count == 1
    assert taken.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [2]

    stale = NativeRowSet.from_positions(
        [0],
        source_token="stale",
        source_row_count=state.row_count,
    )
    with pytest.raises(ValueError, match="source token"):
        state.take(stale)


def test_native_attribute_take_respects_preserve_index_for_host_storage() -> None:
    source_index = pd.Index(["a", "b", "c"], name="source")
    pandas_attributes = NativeAttributeTable(
        dataframe=pd.DataFrame({"value": [10, 20, 30]}, index=source_index)
    )
    arrow_attributes = NativeAttributeTable(
        arrow_table=pa.table({"value": pa.array([10, 20, 30], type=pa.int64())}),
        index_override=source_index,
    )
    row_positions = np.asarray([2, 0], dtype=np.int64)

    pandas_preserved = pandas_attributes.take(row_positions, preserve_index=True)
    pandas_rebased = pandas_attributes.take(row_positions, preserve_index=False)
    arrow_preserved = arrow_attributes.take(row_positions, preserve_index=True)
    arrow_rebased = arrow_attributes.take(row_positions, preserve_index=False)

    assert pandas_preserved.index.tolist() == ["c", "a"]
    assert arrow_preserved.index.tolist() == ["c", "a"]
    assert pandas_rebased.index.equals(pd.RangeIndex(2))
    assert arrow_rebased.index.equals(pd.RangeIndex(2))
    assert pandas_rebased.to_pandas()["value"].tolist() == [30, 10]
    assert arrow_rebased.to_pandas()["value"].tolist() == [30, 10]


def test_native_attribute_take_keeps_device_storage_for_host_positions() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device attribute host-position take")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    source_index = pd.Index(["a", "b", "c"], name="source")
    arrow = pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(arrow),
        index_override=source_index,
        column_override=tuple(arrow.column_names),
        schema_override=arrow.schema,
    )
    row_positions = np.asarray([2, 0], dtype=np.int64)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        taken = attributes.take(row_positions, preserve_index=True)
        arrays = taken.numeric_column_arrays(("value",))

    assert taken.index.tolist() == ["c", "a"]
    assert taken.device_table is not None
    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [30, 10]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_native_attribute_host_take_composes_device_row_view() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for nested device attribute take")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    arrow = pa.table(
        {"value": pa.array([10, 20, 30, 40, 50, 60], type=pa.int64())}
    )
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(arrow),
        index_override=pd.RangeIndex(6),
        column_override=tuple(arrow.column_names),
        schema_override=arrow.schema,
    )
    row_view = attributes.take(
        cp.asarray([5, 1, 4, 2], dtype=cp.int64),
        preserve_index=False,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        taken = row_view.take(
            np.asarray([2, 0], dtype=np.int64),
            preserve_index=True,
        )
        arrays = taken.numeric_column_arrays(("value",))

    assert taken.device_table is not None
    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [50, 60]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_geodataframe_private_native_rowset_selector_preserves_taken_state() -> None:
    gdf, state = _native_backed_geodataframe()
    rowset = NativeRowSet.from_positions(
        [1],
        source_token=state.lineage_token,
        source_row_count=state.row_count,
    )

    filtered = gdf[rowset]
    filtered_state = get_native_state(filtered)

    assert filtered["value"].tolist() == [2]
    assert filtered_state is not None
    assert filtered_state.row_count == 1


def test_geodataframe_explode_lineal_preserves_private_native_state() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for public explode native state preservation")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    geometries = [
        LineString([(0, 0), (1, 0)]),
        MultiLineString([[(0, 0), (0, 2)], [(0, 0), (3, 0)]]),
    ]
    gdf = GeoDataFrame(
        {
            "value": [10, 20],
            "geometry": GeoSeries(geometries, name="geometry"),
        }
    )
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public lineal explode native state",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    attach_native_state(gdf, state)
    reset_d2h_transfer_count()
    clear_materialization_events()

    exploded = gdf.explode(ignore_index=True)
    producer_events = get_d2h_transfer_events(clear=True)
    exploded_state = get_native_state(exploded)

    assert exploded["value"].tolist() == [10, 20, 20]
    assert exploded_state is not None
    assert exploded_state.row_count == 3
    assert exploded_state.index_plan.kind == "range"
    assert exploded_state.attributes.device_table is not None
    assert isinstance(exploded_state.provenance, NativeGeometryProvenance)
    assert cp.asnumpy(exploded_state.provenance.source_rows).tolist() == [0, 1, 1]
    assert producer_events == []

    reset_d2h_transfer_count()
    clear_materialization_events()
    with assert_zero_d2h_transfers():
        length = exploded_state.geometry_length_expression()
        arrays = exploded_state.attributes.numeric_column_arrays(("value",))
        rowset = length.greater_than(1.5)

    assert get_materialization_events(clear=True) == []
    assert arrays is not None
    assert cp.asnumpy(length.values).tolist() == [1.0, 2.0, 3.0]
    assert cp.asnumpy(arrays["value"]).tolist() == [10, 20, 20]
    assert cp.asnumpy(rowset.positions).tolist() == [1, 2]

    reset_d2h_transfer_count()
    clear_materialization_events()
    filtered = exploded_state.take(rowset, preserve_index=False)
    take_events = get_d2h_transfer_events(clear=True)

    assert filtered.row_count == 2
    assert get_materialization_events(clear=True) == []
    assert len(take_events) <= 2
    assert sum(event.bytes_transferred for event in take_events) <= 64
    reset_d2h_transfer_count()


def test_geodataframe_explode_point_preserves_private_native_state() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for public point explode native state")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    geometries = [Point(0, 0), MultiPoint([(1, 0), (3, 0)])]
    gdf = GeoDataFrame(
        {
            "value": [10, 20],
            "geometry": GeoSeries(geometries, name="geometry"),
        }
    )
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public point explode native state",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    attach_native_state(gdf, state)
    reset_d2h_transfer_count()
    clear_materialization_events()

    exploded = gdf.explode(ignore_index=True)
    producer_events = get_d2h_transfer_events(clear=True)
    exploded_state = get_native_state(exploded)

    assert exploded["value"].tolist() == [10, 20, 20]
    assert exploded_state is not None
    assert exploded_state.row_count == 3
    assert exploded_state.attributes.device_table is not None
    assert isinstance(exploded_state.provenance, NativeGeometryProvenance)
    assert cp.asnumpy(exploded_state.provenance.source_rows).tolist() == [0, 1, 1]
    assert producer_events == []

    reset_d2h_transfer_count()
    clear_materialization_events()
    with assert_zero_d2h_transfers():
        centroid_x, _centroid_y = exploded_state.geometry_centroid_expressions()
        arrays = exploded_state.attributes.numeric_column_arrays(("value",))
        rowset = centroid_x.greater_than(0.5)

    assert get_materialization_events(clear=True) == []
    assert arrays is not None
    assert cp.asnumpy(centroid_x.values).tolist() == [0.0, 1.0, 3.0]
    assert cp.asnumpy(arrays["value"]).tolist() == [10, 20, 20]
    assert cp.asnumpy(rowset.positions).tolist() == [1, 2]
    reset_d2h_transfer_count()


def test_geodataframe_explode_mixed_family_preserves_private_native_state() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for public mixed explode native state")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    geometries = [
        Point(0, 0),
        MultiLineString([[(0, 0), (0, 2)], [(0, 0), (3, 0)]]),
        box(0, 0, 1, 1),
    ]
    gdf = GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": GeoSeries(geometries, name="geometry"),
        }
    )
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public mixed explode native state",
    )
    attribute_arrow = pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    attach_native_state(gdf, state)
    reset_d2h_transfer_count()
    clear_materialization_events()

    exploded = gdf.explode(ignore_index=True)
    producer_events = get_d2h_transfer_events(clear=True)
    exploded_state = get_native_state(exploded)

    assert exploded["value"].tolist() == [10, 20, 20, 30]
    assert exploded.geometry.geom_type.tolist() == [
        "Point",
        "LineString",
        "LineString",
        "Polygon",
    ]
    assert exploded_state is not None
    assert exploded_state.row_count == 4
    assert exploded_state.attributes.device_table is not None
    assert isinstance(exploded_state.provenance, NativeGeometryProvenance)
    assert exploded_state.provenance.operation == "mixed_geometry_parts"
    assert cp.asnumpy(exploded_state.provenance.source_rows).tolist() == [0, 1, 1, 2]
    assert cp.asnumpy(exploded_state.provenance.part_family_tags).tolist() == [
        FAMILY_TAGS[GeometryFamily.POINT],
        FAMILY_TAGS[GeometryFamily.LINESTRING],
        FAMILY_TAGS[GeometryFamily.LINESTRING],
        FAMILY_TAGS[GeometryFamily.POLYGON],
    ]
    assert producer_events == []

    reset_d2h_transfer_count()
    clear_materialization_events()
    with assert_zero_d2h_transfers():
        length = exploded_state.geometry_length_expression()
        lineal_rowset = exploded_state.geometry_family_rowset(
            GeometryFamily.LINESTRING,
        )
        arrays = exploded_state.attributes.numeric_column_arrays(("value",))
        rowset = length.greater_than(1.5)

    assert get_materialization_events(clear=True) == []
    assert lineal_rowset is not None
    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [10, 20, 20, 30]
    assert cp.asnumpy(length.values).tolist() == [0.0, 2.0, 3.0, 4.0]
    assert cp.asnumpy(lineal_rowset.positions).tolist() == [1, 2]
    assert cp.asnumpy(rowset.positions).tolist() == [1, 2, 3]
    reset_d2h_transfer_count()


def test_geodataframe_explode_geometrycollection_parts_attach_private_native_state() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for public GeometryCollection native ingress")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    gdf = GeoDataFrame(
        {
            "value": [10, 20],
            "geometry": GeoSeries(
                [
                    GeometryCollection(
                        [
                            Point(0, 0),
                            LineString([(0, 0), (0, 2)]),
                            box(0, 0, 1, 1),
                        ]
                    ),
                    Point(3, 0),
                ],
                name="geometry",
            ),
        }
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    exploded = gdf.explode(ignore_index=True)
    producer_events = get_d2h_transfer_events(clear=True)
    exploded_state = get_native_state(exploded)

    assert exploded["value"].tolist() == [10, 10, 10, 20]
    assert exploded.geometry.geom_type.tolist() == [
        "Point",
        "LineString",
        "Polygon",
        "Point",
    ]
    assert exploded_state is not None
    assert exploded_state.row_count == 4
    assert exploded_state.geometry.owned.residency is Residency.DEVICE
    assert isinstance(exploded_state.provenance, NativeGeometryProvenance)
    assert exploded_state.provenance.operation == "geometrycollection_parts"
    assert cp.asnumpy(exploded_state.provenance.source_rows).tolist() == [0, 0, 0, 1]
    assert cp.asnumpy(exploded_state.provenance.part_family_tags).tolist() == [
        FAMILY_TAGS[GeometryFamily.POINT],
        FAMILY_TAGS[GeometryFamily.LINESTRING],
        FAMILY_TAGS[GeometryFamily.POLYGON],
        FAMILY_TAGS[GeometryFamily.POINT],
    ]
    assert producer_events == []

    reset_d2h_transfer_count()
    clear_materialization_events()
    with assert_zero_d2h_transfers():
        length = exploded_state.geometry_length_expression()
        lineal_rowset = exploded_state.geometry_family_rowset(
            GeometryFamily.LINESTRING,
        )
        rowset = length.greater_than(1.5)

    assert get_materialization_events(clear=True) == []
    assert lineal_rowset is not None
    assert cp.asnumpy(length.values).tolist() == [0.0, 2.0, 4.0, 0.0]
    assert cp.asnumpy(lineal_rowset.positions).tolist() == [1]
    assert cp.asnumpy(rowset.positions).tolist() == [1, 2]
    reset_d2h_transfer_count()


def test_geodataframe_private_area_expression_feeds_native_row_filter() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for private area expression composition")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    geometries = [box(0, 0, 1, 1), box(0, 0, 3, 3), box(0, 0, 5, 5)]
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries(
                geometries,
                index=pd.Index(["a", "b", "c"], name="parcel"),
                name="geometry",
            ),
        },
        index=pd.Index(["a", "b", "c"], name="parcel"),
    )
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test private geodataframe area expression",
    )
    attribute_arrow = pa.table({"value": pa.array([1, 2, 3], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                index_override=gdf.index,
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    attach_native_state(gdf, state)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        expression = gdf._native_geometry_area_expression()
        rowset = expression.greater_than(4.0)
        taken_state = state.take(rowset, preserve_index=False)

    assert expression.is_device
    assert rowset.is_device
    assert cp.asnumpy(rowset.positions).tolist() == [1, 2]
    assert taken_state.row_count == 2
    assert taken_state.attributes.device_table is not None
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_public_geom_type_isin_mask_feeds_native_boolean_filter(monkeypatch) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native geom_type mask rowset")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.api.geo_base import _native_boolean_rowset_from_public_mask
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    geometries = [
        box(0, 0, 1, 1),
        Point(2, 2),
        box(0, 0, 3, 3),
    ]
    index = pd.Index(["a", "b", "c"], name="parcel")
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries(geometries, index=index, name="geometry"),
        },
        index=index,
    )
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public geom_type native rowset",
    )
    attribute_arrow = pa.table({"value": pa.array([1, 2, 3], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                index_override=gdf.index,
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    attach_native_state(gdf, state)

    geometry = gdf.geometry
    assert get_native_state(geometry) is not None
    clear_materialization_events()
    reset_d2h_transfer_count()
    with assert_zero_d2h_transfers():
        mask = geometry.geom_type.isin({"Polygon", "MultiPolygon"})
    assert get_materialization_events(clear=True) == []
    rowset = _native_boolean_rowset_from_public_mask(mask, state)
    assert rowset is not None
    assert rowset.is_device
    assert cp.asnumpy(rowset.positions).tolist() == [0, 2]
    assert rowset.geometry_family_domain == (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
    assert rowset.trusted_all_valid_rows is True

    clear_materialization_events()
    counts = geometry.geom_type.value_counts()
    assert counts.to_dict() == {"Polygon": 2, "Point": 1}
    materialization_operations = {
        event.operation for event in get_materialization_events(clear=True)
    }
    assert "geoseries_geom_type" in materialization_operations

    clear_materialization_events()
    reset_d2h_transfer_count()
    all_mask = geometry.geom_type.isin({"Point", "Polygon", "MultiPolygon"})
    assert all_mask.all()
    reduction_events = get_materialization_events(clear=True)
    assert reduction_events == []

    clear_materialization_events()
    all_filtered = gdf[all_mask]
    assert all_filtered.index.tolist() == ["a", "b", "c"]
    assert get_native_state(all_filtered) is not None
    assert get_materialization_events(clear=True) == []

    clear_materialization_events()

    original_getitem = pd.DataFrame.__getitem__

    def _fail_pandas_boolean_filter(self, key):
        if isinstance(key, pd.Series) and (
            pd.api.types.is_bool_dtype(key.dtype)
            or getattr(key.dtype, "name", None) == "vibespatial_bool"
        ):
            raise AssertionError("native boolean filter should run before pandas")
        return original_getitem(self, key)

    monkeypatch.setattr(pd.DataFrame, "__getitem__", _fail_pandas_boolean_filter)
    filtered = gdf[mask]
    filtered_state = get_native_state(filtered)
    materialization_operations = {
        event.operation for event in get_materialization_events(clear=True)
    }

    assert filtered.index.tolist() == ["a", "c"]
    assert filtered["value"].tolist() == [1, 3]
    assert filtered_state is not None
    assert filtered_state.row_count == 2
    assert filtered_state.attributes.device_table is not None
    assert filtered_state.geometry.owned.device_state.trusted_all_valid is True
    assert filtered_state.geometry.owned.device_state.trusted_polygonal_only is True
    assert "row_positions_to_host" not in materialization_operations
    assert "geoseries_geom_type_isin" not in materialization_operations
    assert filtered.geometry.geom_type.tolist() == ["Polygon", "Polygon"]


def test_public_geom_type_isin_filter_preserves_loader_attributes_until_access(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native geom_type mask rowset")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    geometries = [
        box(0, 0, 1, 1),
        Point(2, 2),
        box(0, 0, 3, 3),
    ]
    source = GeoDataFrame(
        {
            "value": [10, 20, 30],
            "geometry": GeoSeries(geometries, name="geometry"),
        },
    )
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public geom_type loader-backed rowset",
    )
    calls = 0

    def _load() -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return pd.DataFrame({"value": [10, 20, 30]}, index=pd.RangeIndex(3))

    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable.from_loader(
                _load,
                index_override=source.index,
                columns=("value",),
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    attach_native_state(source, state)

    original_getitem = pd.DataFrame.__getitem__

    def _fail_pandas_boolean_filter(self, key):
        if isinstance(key, pd.Series) and (
            pd.api.types.is_bool_dtype(key.dtype)
            or getattr(key.dtype, "name", None) == "vibespatial_bool"
        ):
            raise AssertionError("native boolean filter should run before pandas")
        return original_getitem(self, key)

    monkeypatch.setattr(pd.DataFrame, "__getitem__", _fail_pandas_boolean_filter)

    mask = source.geometry.geom_type.isin({"Polygon", "MultiPolygon"})
    clear_materialization_events()
    reset_d2h_transfer_count()
    with assert_zero_d2h_transfers():
        filtered = source[mask].copy()

    filtered_state = get_native_state(filtered)
    assert filtered_state is not None
    assert filtered_state.row_count == 2
    assert filtered_state.attributes.loader is not None
    assert calls == 0
    assert cp.asnumpy(filtered_state.index_plan.device_labels).tolist() == [0, 2]
    assert get_materialization_events(clear=True) == []
    assert get_d2h_transfer_events(clear=True) == []

    assert filtered["value"].tolist() == [10, 30]
    assert calls == 1


def test_public_geom_type_isin_filter_uses_rowset_for_host_label_take_index(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native geom_type mask rowset")
    cp = pytest.importorskip("cupy")
    from vibespatial.api._native_public_arrays import native_public_index_from_plan
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    geometries = [
        box(0, 0, 1, 1),
        Point(2, 2),
        box(0, 0, 3, 3),
    ]
    base_index = pd.Index(["a", "b", "c"], name="parcel")
    index_plan = NativeIndexPlan.from_index(base_index).take(
        cp.asarray([0, 2, 1], dtype=cp.int64),
        preserve_index=True,
        unique=True,
    )
    public_index = native_public_index_from_plan(index_plan)
    source = GeoDataFrame(
        {
            "value": [10, 30, 20],
            "geometry": GeoSeries(geometries, index=public_index, name="geometry"),
        },
        index=public_index,
    )
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public geom_type host-label-take rowset",
    )
    calls = 0

    def _load() -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return pd.DataFrame({"value": [10, 30, 20]}, index=public_index)

    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable.from_loader(
                _load,
                index_override=public_index,
                columns=("value",),
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
            index_plan=index_plan,
        )
    )
    attach_native_state(source, state)

    original_getitem = pd.DataFrame.__getitem__

    def _fail_pandas_boolean_filter(self, key):
        if isinstance(key, pd.Series) and (
            pd.api.types.is_bool_dtype(key.dtype)
            or getattr(key.dtype, "name", None) == "vibespatial_bool"
        ):
            raise AssertionError("native boolean filter should run before pandas")
        return original_getitem(self, key)

    monkeypatch.setattr(pd.DataFrame, "__getitem__", _fail_pandas_boolean_filter)

    mask = source.geometry.geom_type.isin({"Polygon", "MultiPolygon"})
    clear_materialization_events()
    reset_d2h_transfer_count()
    with assert_zero_d2h_transfers():
        filtered = source[mask].copy()

    filtered_state = get_native_state(filtered)
    assert filtered_state is not None
    assert filtered_state.row_count == 2
    assert filtered_state.attributes.loader is not None
    assert calls == 0
    assert cp.asnumpy(filtered_state.index_plan.take_positions).tolist() == [0, 1]
    assert get_materialization_events(clear=True) == []
    assert get_d2h_transfer_events(clear=True) == []

    assert filtered.index.tolist() == ["a", "b"]
    assert filtered["value"].tolist() == [10, 20]
    assert calls == 1


def test_public_geom_type_identity_filter_reuses_public_columns_and_device_state() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native geom_type identity rowset")
    pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")

    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    geometries = [box(0, 0, 1, 1), box(1, 1, 2, 2), box(2, 2, 3, 3)]
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public geom_type identity rowset",
    )
    index = pd.Index(["a", "b", "c"], name="parcel")
    source = GeoDataFrame(
        {
            "value": [10, 20, 30],
            "region": pd.array(["north", "south", "east"], dtype="string"),
            "geometry": GeoSeries(
                DeviceGeometryArray._from_owned(owned),
                index=index,
                name="geometry",
            ),
        },
        index=index,
    )
    attribute_arrow = pa.table(
        {
            "value": pa.array([10, 20, 30], type=pa.int64()),
            "region": pa.array(["north", "south", "east"], type=pa.string()),
        }
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                index_override=source.index,
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "region", "geometry"),
        )
    )
    attach_native_state(source, state)
    mask = source.geometry.geom_type.isin({"Polygon", "MultiPolygon"})

    clear_materialization_events()
    reset_d2h_transfer_count()
    with assert_zero_d2h_transfers():
        filtered = source[mask].copy()
    events = get_materialization_events(clear=True)
    filtered_state = get_native_state(filtered)

    assert filtered_state is not None
    assert filtered_state.row_count == 3
    assert filtered_state.attributes.device_table is not None
    assert filtered["value"].tolist() == [10, 20, 30]
    assert filtered["region"].tolist() == ["north", "south", "east"]
    assert events == []
    assert get_d2h_transfer_events(clear=True) == []


def test_public_dwithin_scalar_mask_feeds_native_boolean_filter(monkeypatch) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native dwithin mask rowset")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.api.geo_base import _native_boolean_rowset_from_public_mask

    geometries = [Point(0, 0), Point(1, 1), Point(5, 5)]
    index = pd.Index(["a", "b", "c"], name="building")
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public dwithin native rowset",
    )
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries(
                DeviceGeometryArray._from_owned(owned),
                index=index,
                name="geometry",
            ),
        },
        index=index,
    )
    attribute_arrow = pa.table({"value": pa.array([1, 2, 3], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                index_override=gdf.index,
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    attach_native_state(gdf, state)

    clear_dispatch_events()
    mask = gdf.geometry.dwithin(Point(0, 0), distance=1.5)
    rowset = _native_boolean_rowset_from_public_mask(mask, state)
    dispatch_events = get_dispatch_events(clear=True)

    assert mask.tolist() == [True, True, False]
    assert rowset is not None
    assert rowset.is_device
    assert cp.asnumpy(rowset.positions).tolist() == [0, 1]
    assert any(
        event.surface == "DeviceGeometryArray._dwithin_scalar"
        and event.implementation == "dwithin_scalar_gpu"
        for event in dispatch_events
    )

    clear_materialization_events()
    original_getitem = pd.DataFrame.__getitem__

    def _fail_pandas_boolean_filter(self, key):
        if isinstance(key, pd.Series) and pd.api.types.is_bool_dtype(key.dtype):
            raise AssertionError("native dwithin filter should run before pandas")
        return original_getitem(self, key)

    monkeypatch.setattr(pd.DataFrame, "__getitem__", _fail_pandas_boolean_filter)
    filtered = gdf[mask]
    filtered_state = get_native_state(filtered)
    materialization_operations = {
        event.operation for event in get_materialization_events(clear=True)
    }

    assert filtered.index.tolist() == ["a", "b"]
    assert filtered["value"].tolist() == [1, 2]
    assert filtered_state is not None
    assert filtered_state.row_count == 2
    assert filtered_state.attributes.device_table is not None
    assert "row_positions_to_host" not in materialization_operations


def test_geodataframe_assign_native_expression_column_feeds_known_consumers() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for public native expression assignment")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    geometries = [box(0, 0, 1, 1), box(0, 0, 3, 3), box(0, 0, 5, 5)]
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries(geometries, name="geometry"),
        }
    )
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public native expression assignment",
    )
    attribute_arrow = pa.table({"value": pa.array([1, 2, 3], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    attach_native_state(gdf, state)

    area_expression = gdf._native_geometry_area_expression()
    clear_materialization_events()
    reset_d2h_transfer_count()
    assigned = gdf.assign(area=area_expression)
    events = get_materialization_events(clear=True)
    d2h_events = get_d2h_transfer_events(clear=True)
    assigned_state = get_native_state(assigned)

    assert events == []
    assert d2h_events == []
    assert assigned_state is not None
    assert assigned_state.column_order == ("value", "geometry", "area")
    assert assigned_state.attributes.device_table is not None

    assert assigned["area"].tolist() == [1.0, 9.0, 25.0]
    export_events = get_materialization_events(clear=True)
    export_d2h_events = get_d2h_transfer_events(clear=True)
    assert any(
        event.boundary is MaterializationBoundary.USER_EXPORT
        and event.operation == "native_expression_to_public_column"
        for event in export_events
    )
    assert any(
        event.reason.endswith("::native_expression_to_public_column") for event in export_d2h_events
    )

    reset_d2h_transfer_count()
    clear_materialization_events()
    with assert_zero_d2h_transfers():
        area_column = assigned._native_attribute_expression("area")
        assert area_column is not None
        rowset = area_column.greater_than(4.0)
        filtered_state = assigned_state.take(rowset, preserve_index=False)

    assert area_column.is_device
    assert rowset.is_device
    assert cp.asnumpy(rowset.positions).tolist() == [1, 2]
    assert filtered_state.row_count == 2
    assert filtered_state.attributes.device_table is not None
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_geodataframe_assign_native_expression_series_without_frame_state() -> None:
    from vibespatial.api._native_expression import NativeExpression
    from vibespatial.api._native_public_arrays import NativeNumericExpressionArray

    gdf = GeoDataFrame(
        {"geometry": GeoSeries([Point(0, 0), Point(1, 1)], name="geometry")},
        index=pd.Index(["a", "b"]),
    )
    values = pd.Series(
        NativeNumericExpressionArray(
            NativeExpression(
                operation="test.detached_assignment",
                values=np.asarray([2, 1], dtype=np.int32),
                source_token="detached-source",
                source_row_count=2,
                dtype="int32",
                precision="source",
            )
        ),
        index=pd.Index(["b", "a"]),
    )

    gdf["code"] = values

    assert gdf["code"].tolist() == [1, 2]
    assert gdf["code"].dtype == np.dtype("int32")


def test_public_area_assignment_promotes_arrow_attributes_to_device() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for public area native sidecar assignment")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.api.geo_base import _native_expression_from_public_series

    geometries = [box(0, 0, 1, 1), box(0, 0, 3, 3), box(0, 0, 5, 5)]
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries(geometries, name="geometry"),
        }
    )
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public area assignment sidecar",
    )
    attribute_arrow = pa.table({"value": pa.array([1, 2, 3], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(arrow_table=attribute_arrow),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    attach_native_state(gdf, state)

    area = gdf.geometry.area
    area_expression = _native_expression_from_public_series(area)
    assert area.tolist() == [1.0, 9.0, 25.0]
    assert area_expression is not None
    assert area_expression.is_device

    gdf["area"] = area
    assigned_state = get_native_state(gdf)
    assert assigned_state is not None
    assert assigned_state.column_order == ("value", "geometry", "area")
    assert assigned_state.attributes.device_table is not None

    area_column = gdf._native_attribute_expression("area")
    assert area_column is not None
    assert area_column.is_device
    assert cp.asnumpy(area_column.values).tolist() == [1.0, 9.0, 25.0]


def test_public_area_setitem_series_sidecar_does_not_rematerialize_expression() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for public area native sidecar assignment")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.api.geo_base import _native_expression_from_public_series
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    geometries = [box(0, 0, 1, 1), box(0, 0, 3, 3), box(0, 0, 5, 5)]
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries(geometries, name="geometry"),
        }
    )
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public area sidecar setitem no rematerialize",
    )
    attribute_arrow = pa.table({"value": pa.array([1, 2, 3], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(arrow_table=attribute_arrow),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    attach_native_state(gdf, state)

    area = gdf.geometry.area
    assert _native_expression_from_public_series(area) is not None
    reset_d2h_transfer_count()
    clear_materialization_events()
    gdf["area"] = area
    d2h_reasons = {event.reason for event in get_d2h_transfer_events(clear=True)}
    materialization_operations = {
        event.operation for event in get_materialization_events(clear=True)
    }

    assigned_state = get_native_state(gdf)
    area_column = gdf._native_attribute_expression("area")
    assert assigned_state is not None
    assert assigned_state.column_order == ("value", "geometry", "area")
    assert assigned_state.attributes.device_table is not None
    assert area_column is not None
    assert area_column.is_device
    assert cp.asnumpy(area_column.values).tolist() == [1.0, 9.0, 25.0]
    assert "native_expression_to_public_column" not in materialization_operations
    assert not any(
        reason.endswith("::native_expression_to_public_column") for reason in d2h_reasons
    )


def test_public_area_series_stays_lazy_until_public_values() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for public area native sidecar")
    pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.api.geo_base import _native_expression_from_public_series
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    geometries = [box(0, 0, 1, 1), box(0, 0, 3, 3), box(0, 0, 5, 5)]
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries(geometries, name="geometry"),
        }
    )
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public area lazy sidecar",
    )
    attribute_arrow = pa.table({"value": pa.array([1, 2, 3], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(arrow_table=attribute_arrow),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    attach_native_state(gdf, state)

    clear_materialization_events()
    reset_d2h_transfer_count()
    area = gdf.geometry.area
    assert _native_expression_from_public_series(area) is not None
    assert get_materialization_events(clear=True) == []
    assert get_d2h_transfer_events(clear=True) == []

    assert area.tolist() == [1.0, 9.0, 25.0]
    materialization_operations = [
        event.operation for event in get_materialization_events(clear=True)
    ]
    d2h_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    assert materialization_operations == ["geoseries_area"]
    assert d2h_reasons == [
        "vibespatial.api.GeoSeries.area::geoseries_area",
    ]


def test_public_area_series_arithmetic_filter_preserves_device_rowflow() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for public area native sidecar arithmetic")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.api.geo_base import (
        _native_boolean_rowset_from_public_mask,
        _native_expression_from_public_series,
    )
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    geometries = [box(0, 0, 1, 1), box(0, 0, 3, 3), box(0, 0, 5, 5)]
    gdf = GeoDataFrame(
        {
            "parcel_area": [1.0, 3.0, 5.0],
            "geometry": GeoSeries(geometries, name="geometry"),
        }
    )
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public area arithmetic sidecar",
    )
    attribute_arrow = pa.table({"parcel_area": pa.array([1.0, 3.0, 5.0], type=pa.float64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("parcel_area", "geometry"),
        )
    )
    attach_native_state(gdf, state)

    reset_d2h_transfer_count()
    clear_materialization_events()
    with assert_zero_d2h_transfers():
        area = gdf.geometry.area
        ratio = area / gdf["parcel_area"]
        ratio_expression = _native_expression_from_public_series(ratio)
        assert ratio_expression is not None
        assert ratio_expression.is_device
        gdf["area_ratio"] = ratio
        assigned_state = get_native_state(gdf)
        assert assigned_state is not None
        mask = gdf["area_ratio"] >= 3.0
        rowset = _native_boolean_rowset_from_public_mask(mask, assigned_state)
        filtered = gdf[mask]

    filtered_state = get_native_state(filtered)
    ratio_column = gdf._native_attribute_expression("area_ratio")

    assert cp.asnumpy(ratio_expression.values).tolist() == [1.0, 3.0, 5.0]
    assert ratio_column is not None
    assert ratio_column.is_device
    assert rowset is not None
    assert rowset.is_device
    assert cp.asnumpy(rowset.positions).tolist() == [1, 2]
    assert filtered_state is not None
    assert filtered_state.row_count == 2
    assert filtered_state.attributes.device_table is not None
    assert get_materialization_events(clear=True) == []

    assert filtered["area_ratio"].tolist() == [3.0, 5.0]
    export_operations = [event.operation for event in get_materialization_events(clear=True)]
    assert export_operations == ["native_attribute_column_to_public_series"]
    reset_d2h_transfer_count()


def test_geodataframe_copy_preserves_lazy_native_columns_without_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native lazy copy")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    import pylibcudf as plc

    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    geometries = [box(0, 0, 1, 1), box(0, 0, 3, 3), box(0, 0, 5, 5)]
    gdf = GeoDataFrame(
        {
            "parcel_area": [1.0, 3.0, 5.0],
            "geometry": GeoSeries(geometries, name="geometry"),
        }
    )
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native lazy copy",
    )
    attribute_arrow = pa.table({"parcel_area": pa.array([1.0, 3.0, 5.0], type=pa.float64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("parcel_area", "geometry"),
        )
    )
    attach_native_state(gdf, state)
    area = gdf.geometry.area
    gdf["area_ratio"] = area / gdf["parcel_area"]
    filtered = gdf[gdf["area_ratio"] >= 3.0]
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        copied = filtered.copy()
        copied_state = get_native_state(copied)
        area_ratio = copied_state.attributes.numeric_column_arrays(("area_ratio",))

    assert copied_state is not None
    assert copied_state.index_plan.kind == "device-labels"
    assert copied_state.geometry.owned is not None
    assert copied_state.geometry.owned.residency is Residency.DEVICE
    assert area_ratio is not None
    assert cp.asnumpy(area_ratio["area_ratio"]).tolist() == [3.0, 5.0]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_geodataframe_column_projection_preserves_lazy_native_columns_without_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native lazy projection")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    import pylibcudf as plc

    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    geometries = [box(0, 0, 1, 1), box(0, 0, 3, 3), box(0, 0, 5, 5)]
    gdf = GeoDataFrame(
        {
            "parcel_id": [10, 20, 30],
            "geometry": GeoSeries(geometries, name="geometry"),
        }
    )
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native lazy projection",
    )
    attribute_arrow = pa.table({"parcel_id": pa.array([10, 20, 30], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("parcel_id", "geometry"),
        )
    )
    attach_native_state(gdf, state)
    gdf["parcel_area"] = gdf.geometry.area
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        projected = gdf[["parcel_id", "parcel_area", "geometry"]]
        projected_state = get_native_state(projected)
        parcel_values = projected_state.attributes.numeric_column_arrays(("parcel_id",))
        area_values = projected_state.attributes.numeric_column_arrays(("parcel_area",))

    assert projected_state is not None
    assert projected_state.column_order == ("parcel_id", "parcel_area", "geometry")
    assert projected_state.geometry.owned is not None
    assert projected_state.geometry.owned.residency is Residency.DEVICE
    assert parcel_values is not None
    assert area_values is not None
    assert cp.asnumpy(parcel_values["parcel_id"]).tolist() == [10, 20, 30]
    assert cp.asnumpy(area_values["parcel_area"]).tolist() == [1.0, 9.0, 25.0]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_native_backed_public_area_reuses_single_expression(monkeypatch) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for public area native expression export")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.api.geo_base import _native_expression_from_public_series
    from vibespatial.constructive import measurement as measurement_module

    geometries = [box(0, 0, 1, 1), box(0, 0, 3, 3), box(0, 0, 5, 5)]
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries(geometries, name="geometry"),
        }
    )
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test public area expression reuse",
    )
    attribute_arrow = pa.table({"value": pa.array([1, 2, 3], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(arrow_table=attribute_arrow),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    attach_native_state(gdf, state)

    calls = {"device_area": 0}
    real_device_area = measurement_module._area_gpu_device_fp64

    def _count_device_area(*args, **kwargs):
        calls["device_area"] += 1
        return real_device_area(*args, **kwargs)

    def _fail_public_area_owned(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("native-backed public area should reuse NativeExpression")

    monkeypatch.setattr(measurement_module, "_area_gpu_device_fp64", _count_device_area)
    monkeypatch.setattr(measurement_module, "area_owned", _fail_public_area_owned)

    area = gdf.geometry.area
    area_expression = _native_expression_from_public_series(area)

    assert area.tolist() == [1.0, 9.0, 25.0]
    assert area_expression is not None
    assert area_expression.is_device
    assert cp.asnumpy(area_expression.values).tolist() == [1.0, 9.0, 25.0]
    assert calls["device_area"] == 1


def test_geodataframe_setitem_native_expression_column_preserves_state() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native expression setitem")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")

    geometries = [box(0, 0, 1, 1), box(0, 0, 2, 2)]
    gdf = GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": GeoSeries(geometries, name="geometry"),
        }
    )
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native expression setitem",
    )
    attribute_arrow = pa.table({"value": pa.array([1, 2], type=pa.int64())})
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    attach_native_state(gdf, state)

    gdf["perimeter"] = gdf._native_geometry_length_expression()
    updated_state = get_native_state(gdf)
    perimeter_column = gdf._native_attribute_expression("perimeter")

    assert gdf["perimeter"].tolist() == [4.0, 8.0]
    assert updated_state is not None
    assert updated_state.column_order == ("value", "geometry", "perimeter")
    assert updated_state.attributes.device_table is not None
    assert perimeter_column is not None
    assert perimeter_column.is_device
    assert cp.asnumpy(perimeter_column.values).tolist() == [4.0, 8.0]


def test_geodataframe_native_series_arithmetic_filter_preserves_device_rowflow() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native expression public Series arithmetic")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.api.geo_base import (
        _native_boolean_rowset_from_public_mask,
        _native_expression_from_public_series,
    )
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    geometries = [box(0, 0, 1, 1), box(0, 0, 2, 2), box(0, 0, 3, 3)]
    gdf = GeoDataFrame(
        {
            "wet_area": [0.2, 2.0, 6.0],
            "parcel_area": [1.0, 4.0, 6.0],
            "geometry": GeoSeries(geometries, name="geometry"),
        }
    )
    owned = from_shapely_geometries(geometries).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native expression public arithmetic",
    )
    attribute_arrow = pa.table(
        {
            "wet_area": pa.array([0.2, 2.0, 6.0], type=pa.float64()),
            "parcel_area": pa.array([1.0, 4.0, 6.0], type=pa.float64()),
        }
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(attribute_arrow),
                column_override=tuple(attribute_arrow.column_names),
                schema_override=attribute_arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("wet_area", "parcel_area", "geometry"),
        )
    )
    attach_native_state(gdf, state)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        wet_area = gdf["wet_area"]
        parcel_area = gdf["parcel_area"]
        ratio = wet_area / parcel_area
        ratio_expression = _native_expression_from_public_series(ratio)
        assert ratio_expression is not None
        assert ratio_expression.is_device
        gdf["flood_ratio"] = ratio
        assigned_state = get_native_state(gdf)
        assert assigned_state is not None
        mask = gdf["flood_ratio"] >= 0.5
        rowset = _native_boolean_rowset_from_public_mask(mask, assigned_state)
        filtered = gdf[mask]

    filtered_state = get_native_state(filtered)
    ratio_column = gdf._native_attribute_expression("flood_ratio")

    assert cp.asnumpy(ratio_expression.values).tolist() == [0.2, 0.5, 1.0]
    assert ratio_column is not None
    assert ratio_column.is_device
    assert rowset is not None
    assert rowset.is_device
    assert cp.asnumpy(rowset.positions).tolist() == [1, 2]
    assert filtered_state is not None
    assert filtered_state.row_count == 2
    assert filtered_state.attributes.device_table is not None
    assert get_materialization_events(clear=True) == []

    assert filtered["flood_ratio"].tolist() == [0.5, 1.0]
    export_operations = [event.operation for event in get_materialization_events(clear=True)]
    assert export_operations == ["native_attribute_column_to_public_series"]
    reset_d2h_transfer_count()


def test_geodataframe_iloc_slice_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()
    clear_materialization_events()

    selected = gdf.iloc[:1]
    selected_state = get_native_state(selected)

    assert selected["value"].tolist() == [1]
    assert selected.index.tolist() == [0]
    assert selected_state is not None
    assert selected_state.row_count == 1
    assert get_materialization_events(clear=True) == []


def test_native_attribute_full_slice_copies_without_allocating_row_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibespatial.api._native_public_arrays as native_arrays

    table = NativeAttributeTable(
        dataframe=pd.DataFrame({"value": [1, 2, 3]}),
    )
    values = native_arrays.NativeAttributeColumnArray(table, "value")

    def _fail_arange(*_args, **_kwargs):
        raise AssertionError("a full extension-array slice must not allocate positions")

    monkeypatch.setattr(native_arrays.np, "arange", _fail_arange)
    copied = values[:]

    assert type(copied).__name__ == "NativeAttributeColumnArray"
    assert len(copied) == len(values)
    assert copied is not values


def test_native_iloc_monotonic_positions_skip_global_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibespatial.api.geodataframe as geodataframe_module

    def _fail_unique(*_args, **_kwargs):
        raise AssertionError("strictly increasing positions already prove uniqueness")

    monkeypatch.setattr(geodataframe_module.np, "unique", _fail_unique)
    positions, unique = geodataframe_module._native_iloc_row_positions(
        np.asarray([1, 3, 7], dtype=np.int64),
        10,
    )

    assert positions.tolist() == [1, 3, 7]
    assert unique is True


def test_geodataframe_iloc_take_preserves_projected_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    selected = gdf.iloc[[1, 0], [1, 2]]
    selected_state = get_native_state(selected)

    assert selected.columns.tolist() == ["name", "geometry"]
    assert selected.index.tolist() == [1, 0]
    assert selected_state is not None
    assert selected_state.row_count == 2
    assert selected_state.column_order == ("name", "geometry")
    assert selected_state.to_native_tabular_result().attributes.to_pandas()["name"].tolist() == [
        "b",
        "a",
    ]


def test_geodataframe_take_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    selected = gdf.take([1, 0])
    selected_state = get_native_state(selected)

    assert selected["value"].tolist() == [2, 1]
    assert selected.index.tolist() == [1, 0]
    assert selected_state is not None
    assert selected_state.row_count == 2
    assert selected_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        2,
        1,
    ]


def test_geodataframe_take_axis_columns_preserves_projected_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    selected = gdf.take([1, 2], axis=1)
    selected_state = get_native_state(selected)

    assert selected.columns.tolist() == ["name", "geometry"]
    assert selected_state is not None
    assert selected_state.column_order == ("name", "geometry")
    assert selected_state.row_count == 2


def test_geodataframe_head_tail_preserve_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    head = gdf.head(1)
    tail = gdf.tail(1)

    assert get_native_state(head) is not None
    assert get_native_state(tail) is not None
    assert head["value"].tolist() == [1]
    assert tail["value"].tolist() == [2]


def test_geodataframe_boolean_series_unique_index_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)"],
                name="geometry",
            ),
        },
        index=pd.Index(["a", "b", "c"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    selected = gdf[pd.Series([True, False, True], index=gdf.index)]
    selected_state = get_native_state(selected)

    assert selected.index.tolist() == ["a", "c"]
    assert selected["value"].tolist() == [1, 3]
    assert selected_state is not None
    assert selected_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        3,
    ]


def test_geodataframe_boolean_series_unique_host_index_is_strict_native_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)"],
                name="geometry",
            ),
        },
        index=pd.Index(["a", "b", "c"], name="site"),
    )
    _attach_owned_native_tabular_state(gdf, attribute_storage="arrow")
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_materialization_events()

    selected = gdf[pd.Series([True, False, True], index=gdf.index)]

    assert selected.index.tolist() == ["a", "c"]
    assert get_native_state(selected) is not None
    assert get_materialization_events(clear=True) == []


def test_geodataframe_boolean_filter_preserves_host_then_device_index_plans() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device-label native filter")
    pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")

    owned = from_shapely_geometries(
        [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        residency=Residency.DEVICE,
    )
    gdf = GeoDataFrame(
        {"value": [0, 1, 2, 3]},
        geometry=GeoSeries(DeviceGeometryArray._from_owned(owned), name="geometry"),
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(
                    pa.table({"value": pa.array([0, 1, 2, 3], type=pa.int64())})
                ),
                index_override=gdf.index,
                column_override=("value",),
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=gdf.crs),
            geometry_name="geometry",
            column_order=("value", "geometry"),
        )
    )
    attach_native_state(gdf, state)

    label_backed_rows = gdf[pd.Series([True, True, True, False], index=gdf.index)]
    label_backed_rows_state = get_native_state(label_backed_rows)
    assert label_backed_rows_state is not None
    assert label_backed_rows_state.index_plan.kind == "range"

    clear_materialization_events()
    selected = label_backed_rows[
        label_backed_rows.geometry.dwithin(Point(1, 1), distance=0.1)
    ]
    selected_state = get_native_state(selected)

    assert selected_state is not None
    assert selected_state.index_plan.kind == "device-labels"
    assert [
        event
        for event in get_materialization_events(clear=True)
        if event.operation == "index_plan_to_host"
    ] == []

    assert selected.index.tolist() == [1]
    index_events = [
        event
        for event in get_materialization_events(clear=True)
        if event.operation == "index_plan_to_host"
    ]
    assert len(index_events) == 1
    assert selected["value"].tolist() == [1]


def test_geodataframe_boolean_series_duplicate_index_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (1 1)", "POINT (2 2)"],
                name="geometry",
            ),
        },
        index=pd.Index(["same", "same"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    selected = gdf[pd.Series([True, False], index=gdf.index)]

    assert selected.index.tolist() == ["same"]
    assert selected["value"].tolist() == [1]
    selected_state = get_native_state(selected)
    assert selected_state is not None
    assert selected_state.index_plan.has_duplicates is False
    assert selected_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [1]


def test_geodataframe_boolean_series_multiindex_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)"],
                name="geometry",
            ),
        },
        index=pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("a", 3)],
            names=["site", "rank"],
        ),
    )
    _attach_native_tabular_state(gdf)

    selected = gdf[pd.Series([True, False, True], index=gdf.index)]
    selected_state = get_native_state(selected)

    assert selected.index.tolist() == [("a", 1), ("a", 3)]
    assert selected["value"].tolist() == [1, 3]
    assert selected_state is not None
    assert selected_state.index_plan.nlevels == 2
    assert selected_state.index_plan.to_public_index().tolist() == [
        ("a", 1),
        ("a", 3),
    ]
    assert selected_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        3,
    ]


def test_geodataframe_position_selectors_delegating_to_take_preserve_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    sampled = gdf.sample(n=1, random_state=0)
    largest = gdf.nlargest(1, "value")

    assert sampled.index.tolist() == [1]
    assert get_native_state(sampled) is not None
    assert largest.index.tolist() == [1]
    assert get_native_state(largest) is not None


def test_geodataframe_drop_row_preserves_unique_index_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)"],
                name="geometry",
            ),
        },
        index=pd.Index(["a", "b", "c"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    selected = gdf.drop(index=["b"])
    selected_state = get_native_state(selected)

    assert selected.index.tolist() == ["a", "c"]
    assert selected["value"].tolist() == [1, 3]
    assert selected_state is not None
    assert selected_state.row_count == 2
    assert selected_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        3,
    ]


def test_geodataframe_drop_column_preserves_projected_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    selected = gdf.drop(columns=["value"])
    selected_state = get_native_state(selected)

    assert selected.columns.tolist() == ["name", "geometry"]
    assert selected_state is not None
    assert selected_state.column_order == ("name", "geometry")
    assert selected_state.row_count == 2


def test_geodataframe_drop_geometry_drops_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    selected = gdf.drop(columns=["geometry"])

    assert get_native_state(selected) is None


def test_geodataframe_drop_duplicate_index_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)"],
                name="geometry",
            ),
        },
        index=pd.Index(["same", "same", "other"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    selected = gdf.drop(index=["other"])

    assert selected.index.tolist() == ["same", "same"]
    selected_state = get_native_state(selected)
    assert selected_state is not None
    assert selected_state.index_plan.has_duplicates is True
    assert selected_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        2,
    ]


def test_geodataframe_drop_multiindex_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)"],
                name="geometry",
            ),
        },
        index=pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("a", 3)],
            names=["site", "rank"],
        ),
    )
    _attach_native_tabular_state(gdf)

    selected = gdf.drop(index=[("b", 2)])
    selected_state = get_native_state(selected)

    assert selected.index.tolist() == [("a", 1), ("a", 3)]
    assert selected["value"].tolist() == [1, 3]
    assert selected_state is not None
    assert selected_state.index_plan.nlevels == 2
    assert selected_state.index_plan.to_public_index().tolist() == [
        ("a", 1),
        ("a", 3),
    ]
    assert selected_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        3,
    ]


def test_geodataframe_drop_inplace_clears_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    result = gdf.drop(index=[0], inplace=True)

    assert result is None
    assert gdf.index.tolist() == [1]
    assert get_native_state(gdf) is None


def test_geodataframe_drop_duplicates_ignore_index_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 1, 2],
            "name": ["a", "b", "c"],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)"],
                name="geometry",
            ),
        },
        index=pd.Index(["same", "same", "other"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    selected = gdf.drop_duplicates(
        subset=["value"],
        keep="last",
        ignore_index=True,
    )
    selected_state = get_native_state(selected)

    assert selected.index.tolist() == [0, 1]
    assert selected["name"].tolist() == ["b", "c"]
    assert selected_state is not None
    assert selected_state.index_plan.to_public_index().tolist() == [0, 1]
    assert selected_state.to_native_tabular_result().attributes.to_pandas()["name"].tolist() == [
        "b",
        "c",
    ]


def test_geodataframe_drop_duplicates_ignore_index_device_state_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native drop_duplicates")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    gdf = GeoDataFrame(
        {
            "value": [1, 1, 2],
            "name": ["a", "b", "c"],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)"],
                name="geometry",
            ),
        }
    )
    state = _attach_owned_native_tabular_state(gdf, attribute_storage="arrow")
    state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native drop_duplicates device geometry",
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        selected = gdf.drop_duplicates(subset=["value"], ignore_index=True)

    selected_state = get_native_state(selected)
    assert selected.index.tolist() == [0, 1]
    assert selected_state is not None
    assert selected_state.index_plan.to_public_index().tolist() == [0, 1]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_geodataframe_drop_duplicates_lazy_native_columns_preserves_device_rowflow() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native lazy drop_duplicates")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    import pylibcudf as plc

    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        residency=Residency.DEVICE,
    )
    arrow = pa.table(
        {
            "parcel_id": pa.array([1, 1, 2, 2], type=pa.int64()),
            "flood_ratio": pa.array([0.7, 0.3, 0.9, 0.1], type=pa.float64()),
        }
    )
    gdf = GeoDataFrame(
        {
            "parcel_id": [1, 1, 2, 2],
            "flood_ratio": [0.7, 0.3, 0.9, 0.1],
            "geometry": GeoSeries(DeviceGeometryArray._from_owned(owned), name="geometry"),
        }
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(arrow),
                index_override=pd.RangeIndex(4),
                column_override=tuple(arrow.column_names),
                schema_override=arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("parcel_id", "flood_ratio", "geometry"),
        )
    )
    attach_native_state(gdf, state)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        deduped = gdf.drop_duplicates("parcel_id")
        deduped_state = get_native_state(deduped)
        parcel_values = deduped_state.attributes.numeric_column_arrays(("parcel_id",))
        ratio_values = deduped_state.attributes.numeric_column_arrays(("flood_ratio",))
        last_deduped = gdf.drop_duplicates("parcel_id", keep="last")
        last_state = get_native_state(last_deduped)
        last_parcel_values = last_state.attributes.numeric_column_arrays(("parcel_id",))
        last_ratio_values = last_state.attributes.numeric_column_arrays(("flood_ratio",))
        no_duplicates = gdf.drop_duplicates("parcel_id", keep=False)
        no_duplicates_state = get_native_state(no_duplicates)

    assert deduped_state is not None
    assert deduped_state.index_plan.kind == "device-labels"
    assert deduped_state.geometry.owned is not None
    assert deduped_state.geometry.owned.residency is Residency.DEVICE
    assert parcel_values is not None
    assert ratio_values is not None
    assert cp.asnumpy(parcel_values["parcel_id"]).tolist() == [1, 2]
    assert cp.asnumpy(ratio_values["flood_ratio"]).tolist() == [0.7, 0.9]
    assert last_state is not None
    assert last_parcel_values is not None
    assert last_ratio_values is not None
    assert cp.asnumpy(last_parcel_values["parcel_id"]).tolist() == [1, 2]
    assert cp.asnumpy(last_ratio_values["flood_ratio"]).tolist() == [0.3, 0.1]
    assert no_duplicates_state is not None
    assert no_duplicates_state.row_count == 0
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


@pytest.mark.parametrize(
    "key_array",
    [
        pytest.param(
            pa.array([1, None, 2, None, 1, 3], type=pa.int64()),
            id="nullable-int64",
        ),
        pytest.param(
            pa.array(["x", None, "y", None, "x", "z"], type=pa.string()),
            id="string",
        ),
        pytest.param(
            pa.array(
                [
                    pd.Timestamp("2026-01-01"),
                    None,
                    pd.Timestamp("2026-01-02"),
                    None,
                    pd.Timestamp("2026-01-01"),
                    pd.Timestamp("2026-01-03"),
                ],
                type=pa.timestamp("ns"),
            ),
            id="datetime",
        ),
        pytest.param(
            pa.array(
                pd.Categorical(
                    ["x", None, "y", None, "x", "z"],
                    categories=["x", "y", "z"],
                )
            ),
            id="categorical",
        ),
        pytest.param(
            pa.array([1.0, np.nan, 2.0, np.nan, 1.0, 3.0], type=pa.float64()),
            id="float-nan",
        ),
        pytest.param(
            pa.array([1.0, None, 2.0, np.nan, 1.0, 3.0], type=pa.float64()),
            id="float-null-nan",
        ),
    ],
)
@pytest.mark.parametrize(
    ("keep", "expected_positions"),
    [
        pytest.param("first", [0, 1, 2, 5], id="first"),
        pytest.param("last", [2, 3, 4, 5], id="last"),
        pytest.param(False, [2, 5], id="none"),
    ],
)
def test_geodataframe_drop_duplicates_device_dtype_breadth_stays_native_until_export(
    key_array: pa.Array,
    keep: str | bool,
    expected_positions: list[int],
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native dtype deduplication")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    arrow = pa.table(
        {
            "key": key_array,
            "source_position": pa.array(range(len(key_array)), type=pa.int64()),
        }
    )
    owned = from_shapely_geometries(
        [Point(position, position) for position in range(len(key_array))],
        residency=Residency.DEVICE,
    )
    public_attributes = arrow.to_pandas()
    gdf = GeoDataFrame(
        {
            "key": public_attributes["key"],
            "source_position": public_attributes["source_position"],
            "geometry": GeoSeries(DeviceGeometryArray._from_owned(owned), name="geometry"),
        }
    )
    attach_native_state(
        gdf,
        NativeFrameState.from_native_tabular_result(
            NativeTabularResult(
                attributes=NativeAttributeTable(
                    device_table=plc.Table.from_arrow(arrow),
                    index_override=pd.RangeIndex(len(key_array)),
                    column_override=tuple(arrow.column_names),
                    schema_override=arrow.schema,
                ),
                geometry=GeometryNativeResult.from_owned(owned, crs=None),
                geometry_name="geometry",
                column_order=("key", "source_position", "geometry"),
            )
        ),
    )
    reset_d2h_transfer_count()
    clear_materialization_events()
    clear_dispatch_events()

    with assert_zero_d2h_transfers():
        deduped = gdf.drop_duplicates("key", keep=keep)
        deduped_state = get_native_state(deduped)
        assert deduped_state is not None
        assert deduped_state.attributes.is_device_backed
        assert deduped_state.attributes.row_positions is not None
        result_positions = cp.asarray(deduped_state.attributes.row_positions)

    assert cp.asnumpy(result_positions).tolist() == expected_positions
    assert get_materialization_events(clear=True) == []
    dispatch_events = get_dispatch_events(clear=True)
    assert [event.implementation for event in dispatch_events] == [
        "native_device_distinct_rowset"
    ]

    exported = deduped_state.attributes.to_arrow(index=False)
    assert exported["source_position"].to_pylist() == expected_positions
    assert exported.schema.field("key").type == key_array.type
    export_events = get_materialization_events(clear=True)
    assert [event.operation for event in export_events] == ["device_attributes_to_arrow"]
    assert all(event.d2h_transfer for event in export_events)
    reset_d2h_transfer_count()


def test_geodataframe_drop_duplicates_device_admission_decline_skips_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device distinct admission")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import DeviceMemoryAdmission, get_cuda_runtime

    arrow = pa.table(
        {
            "key": pa.array(["x", "x", "y"], type=pa.string()),
            "source_position": pa.array([0, 1, 2], type=pa.int64()),
        }
    )
    owned = from_shapely_geometries(
        [Point(0, 0), Point(1, 1), Point(2, 2)],
        residency=Residency.DEVICE,
    )
    gdf = GeoDataFrame(
        {
            "key": ["x", "x", "y"],
            "source_position": [0, 1, 2],
            "geometry": GeoSeries(DeviceGeometryArray._from_owned(owned), name="geometry"),
        }
    )
    attach_native_state(
        gdf,
        NativeFrameState.from_native_tabular_result(
            NativeTabularResult(
                attributes=NativeAttributeTable(
                    device_table=plc.Table.from_arrow(arrow),
                    index_override=pd.RangeIndex(3),
                    column_override=tuple(arrow.column_names),
                    schema_override=arrow.schema,
                ),
                geometry=GeometryNativeResult.from_owned(owned, crs=None),
                geometry_name="geometry",
                column_order=("key", "source_position", "geometry"),
            )
        ),
    )
    runtime = get_cuda_runtime()
    admission_calls = []

    def _decline_distinct(*, stage, required_bytes, requested_units=0):
        admission_calls.append((stage, required_bytes, requested_units))
        return DeviceMemoryAdmission(
            stage=stage,
            required_bytes=required_bytes,
            remaining_bytes=0,
            budget_bytes=0,
            admitted=False,
            requested_units=requested_units,
            admitted_units=0,
            bytes_per_unit=required_bytes,
        )

    monkeypatch.setattr(runtime, "admit_device_memory", _decline_distinct)

    def _fail_distinct(*_args, **_kwargs):
        raise AssertionError("declined tabular distinct must not launch")

    monkeypatch.setattr(plc.stream_compaction, "distinct_indices", _fail_distinct)

    from vibespatial.api import geodataframe as geodataframe_module

    rowset = geodataframe_module._native_drop_duplicates_rowset(gdf, "key", "first")

    assert rowset is None
    assert len(admission_calls) == 1
    stage, required_bytes, requested_units = admission_calls[0]
    assert stage == "tabular-distinct"
    assert required_bytes >= 1 << 20
    assert requested_units == 3


def test_geodataframe_identity_native_drop_duplicates_rebuilds_without_lazy_column_export() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native lazy identity drop_duplicates")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    import pylibcudf as plc

    from vibespatial.api._native_public_arrays import NativeNumericExpressionArray
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [Point(0, 0), Point(1, 1), Point(2, 2)],
        residency=Residency.DEVICE,
    )
    arrow = pa.table(
        {
            "parcel_id": pa.array([1, 2, 3], type=pa.int64()),
            "flood_ratio": pa.array([0.7, 0.3, 0.9], type=pa.float64()),
        }
    )
    gdf = GeoDataFrame(
        {
            "parcel_id": [1, 2, 3],
            "flood_ratio": [0.7, 0.3, 0.9],
            "geometry": GeoSeries(DeviceGeometryArray._from_owned(owned), name="geometry"),
        }
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(arrow),
                index_override=pd.RangeIndex(3),
                column_override=tuple(arrow.column_names),
                schema_override=arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("parcel_id", "flood_ratio", "geometry"),
        )
    )
    attach_native_state(gdf, state)
    projected = gdf[["parcel_id", "flood_ratio", "geometry"]]
    assert isinstance(projected["parcel_id"].array, NativeNumericExpressionArray)

    reset_d2h_transfer_count()
    clear_materialization_events()
    with assert_zero_d2h_transfers():
        deduped = projected.drop_duplicates("parcel_id")
        deduped_state = get_native_state(deduped)
        parcel_values = deduped_state.attributes.numeric_column_arrays(("parcel_id",))
        ratio_values = deduped_state.attributes.numeric_column_arrays(("flood_ratio",))

    assert deduped_state is not None
    assert deduped_state.row_count == 3
    assert parcel_values is not None
    assert ratio_values is not None
    assert cp.asnumpy(parcel_values["parcel_id"]).tolist() == [1, 2, 3]
    assert cp.asnumpy(ratio_values["flood_ratio"]).tolist() == [0.7, 0.3, 0.9]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_geodataframe_set_crs_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    projected = gdf.set_crs("EPSG:4326", allow_override=True)
    projected_state = get_native_state(projected)

    assert projected.crs == "EPSG:4326"
    assert projected_state is not None
    assert projected_state.geometry.crs == "EPSG:4326"
    assert projected_state.column_order == tuple(projected.columns)
    assert projected_state.row_count == len(projected)


def test_geodataframe_set_crs_device_state_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native set_crs")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    gdf = GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    state = _attach_owned_native_tabular_state(gdf, attribute_storage="arrow")
    state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native set_crs device geometry",
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        projected = gdf.set_crs("EPSG:4326", allow_override=True)

    projected_state = get_native_state(projected)
    assert projected_state is not None
    assert projected_state.geometry.crs == "EPSG:4326"
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_geodataframe_set_geometry_current_column_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    selected = gdf.set_geometry("geometry")
    selected_state = get_native_state(selected)

    assert selected.geometry.name == "geometry"
    assert selected_state is not None
    assert selected_state.geometry_name == "geometry"
    assert selected_state.column_order == tuple(selected.columns)
    assert selected_state.row_count == len(selected)


def test_geodataframe_set_geometry_secondary_column_promotes_private_native_state() -> None:
    gdf = _two_geometry_geodataframe()
    _attach_two_geometry_native_tabular_state(gdf)

    selected = gdf.set_geometry("other_geometry")
    selected_state = get_native_state(selected)

    assert selected.geometry.name == "other_geometry"
    assert [str(value) for value in selected.geometry] == [
        "POINT (10 10)",
        "POINT (20 20)",
    ]
    assert selected_state is not None
    assert selected_state.geometry_name == "other_geometry"
    assert selected_state.column_order == ("value", "geometry", "other_geometry")
    assert [column.name for column in selected_state.secondary_geometry] == ["geometry"]


def test_geodataframe_set_geometry_secondary_device_state_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native set_geometry")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    gdf = _two_geometry_geodataframe()
    state = _attach_two_geometry_native_tabular_state(
        gdf,
        owned=True,
        attribute_storage="arrow",
    )
    state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native set_geometry primary device geometry",
    )
    state.secondary_geometry[0].geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native set_geometry secondary device geometry",
    )
    promoted_owned = state.secondary_geometry[0].geometry.owned
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        selected = gdf.set_geometry("other_geometry")

    selected_state = get_native_state(selected)
    assert selected_state is not None
    assert selected_state.geometry_name == "other_geometry"
    assert selected_state.geometry.owned is promoted_owned
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def _native_backed_geoseries_from_arrow(index=None) -> GeoSeries:
    source = GeoSeries(
        [Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
        name="geometry",
    )
    result = GeoSeries.from_arrow(source.to_arrow(), index=index)
    assert get_native_state(result) is not None
    return result


def test_geoseries_set_crs_preserves_private_native_state() -> None:
    series = _native_backed_geoseries_from_arrow()

    projected = series.set_crs("EPSG:3857", allow_override=True)
    projected_state = get_native_state(projected)

    assert projected.crs == "EPSG:3857"
    assert projected_state is not None
    assert projected_state.geometry.crs == "EPSG:3857"
    assert projected_state.column_order == ("geometry",)
    assert projected_state.row_count == len(projected)


def test_geoseries_copy_preserves_private_native_state() -> None:
    series = _native_backed_geoseries_from_arrow()

    copied = series.copy()
    copied_state = get_native_state(copied)

    assert copied is not series
    assert copied_state is not None
    assert copied_state.row_count == len(copied)
    assert copied_state.column_order == ("geometry",)


def test_geoseries_take_preserves_private_native_state() -> None:
    series = _native_backed_geoseries_from_arrow(index=pd.Index(["a", "b", "c"], name="site"))

    selected = series.take([2, 0])
    selected_state = get_native_state(selected)

    assert selected.index.tolist() == ["c", "a"]
    assert [str(value) for value in selected] == ["POINT (2 2)", "POINT (0 0)"]
    assert selected_state is not None
    assert selected_state.index_plan.to_public_index().tolist() == ["c", "a"]
    assert selected_state.row_count == len(selected)


def test_geoseries_head_tail_preserve_private_native_state() -> None:
    series = _native_backed_geoseries_from_arrow(index=pd.Index(["a", "b", "c"], name="site"))

    head = series.head(2)
    tail = series.tail(2)

    assert head.index.tolist() == ["a", "b"]
    assert tail.index.tolist() == ["b", "c"]
    assert get_native_state(head) is not None
    assert get_native_state(tail) is not None


def test_geoseries_drop_reindex_sample_preserve_private_native_state() -> None:
    series = _native_backed_geoseries_from_arrow(index=pd.Index(["a", "b", "c"], name="site"))

    dropped = series.drop(index=["b"])
    reindexed = series.reindex(["c", "a", "c"])
    sampled = series.sample(n=1, random_state=0)

    assert dropped.index.tolist() == ["a", "c"]
    assert reindexed.index.tolist() == ["c", "a", "c"]
    assert sampled.index.tolist() == ["c"]
    assert get_native_state(dropped) is not None
    reindexed_state = get_native_state(reindexed)
    assert reindexed_state is not None
    assert reindexed_state.index_plan.has_duplicates is True
    assert get_native_state(sampled) is not None


def test_geoseries_getitem_slice_preserves_private_native_state() -> None:
    series = _native_backed_geoseries_from_arrow(index=pd.Index(["a", "b", "c"], name="site"))

    selected = series[1:]
    selected_state = get_native_state(selected)

    assert selected.index.tolist() == ["b", "c"]
    assert selected_state is not None
    assert selected_state.row_count == 2
    assert selected_state.index_plan.to_public_index().tolist() == ["b", "c"]


def test_geoseries_getitem_boolean_series_preserves_private_native_state() -> None:
    series = _native_backed_geoseries_from_arrow(index=pd.Index(["a", "b", "c"], name="site"))

    selected = series[pd.Series([True, False, True], index=series.index)]
    selected_state = get_native_state(selected)

    assert selected.index.tolist() == ["a", "c"]
    assert selected_state is not None
    assert selected_state.row_count == 2
    assert selected_state.index_plan.to_public_index().tolist() == ["a", "c"]


def test_geoseries_getitem_boolean_series_label_aligned_preserves_private_native_state() -> None:
    series = _native_backed_geoseries_from_arrow(index=pd.Index(["a", "b", "c"], name="site"))

    selected = series[pd.Series([True, False, True], index=pd.Index(["c", "b", "a"]))]
    selected_state = get_native_state(selected)

    assert selected.index.tolist() == ["a", "c"]
    assert selected_state is not None
    assert selected_state.index_plan.to_public_index().tolist() == ["a", "c"]


def test_geoseries_loc_iloc_preserve_private_native_state() -> None:
    series = _native_backed_geoseries_from_arrow(index=pd.Index(["a", "b", "c"], name="site"))

    iloc_selected = series.iloc[[2, 0]]
    loc_selected = series.loc[["c", "a", "c"]]
    loc_boolean = series.loc[pd.Series([True, False, True], index=pd.Index(["c", "b", "a"]))]

    iloc_state = get_native_state(iloc_selected)
    loc_state = get_native_state(loc_selected)
    loc_boolean_state = get_native_state(loc_boolean)

    assert iloc_selected.index.tolist() == ["c", "a"]
    assert iloc_state is not None
    assert iloc_state.index_plan.to_public_index().tolist() == ["c", "a"]
    assert loc_selected.index.tolist() == ["c", "a", "c"]
    assert loc_state is not None
    assert loc_state.index_plan.has_duplicates is True
    assert loc_state.index_plan.to_public_index().tolist() == ["c", "a", "c"]
    assert loc_boolean.index.tolist() == ["a", "c"]
    assert loc_boolean_state is not None
    assert loc_boolean_state.index_plan.to_public_index().tolist() == ["a", "c"]


def test_geoseries_indexer_writes_clear_private_native_state() -> None:
    series = _native_backed_geoseries_from_arrow(index=pd.Index(["a", "b", "c"], name="site"))

    series.iloc[0] = Point(10, 10)

    assert get_native_state(series) is None


def test_geoseries_sort_index_unique_preserves_private_native_state() -> None:
    series = _native_backed_geoseries_from_arrow(index=pd.Index(["b", "a", "c"], name="site"))

    selected = series.sort_index()
    selected_state = get_native_state(selected)

    assert selected.index.tolist() == ["a", "b", "c"]
    assert [str(value) for value in selected] == [
        "POINT (1 1)",
        "POINT (0 0)",
        "POINT (2 2)",
    ]
    assert selected_state is not None
    assert selected_state.index_plan.to_public_index().tolist() == ["a", "b", "c"]


def test_geoseries_sort_index_duplicate_preserves_private_native_state() -> None:
    series = _native_backed_geoseries_from_arrow(index=pd.Index(["b", "a", "a"], name="site"))

    selected = series.sort_index(kind="mergesort")
    selected_state = get_native_state(selected)

    assert selected.index.tolist() == ["a", "a", "b"]
    assert [str(value) for value in selected] == [
        "POINT (1 1)",
        "POINT (2 2)",
        "POINT (0 0)",
    ]
    assert selected_state is not None
    assert selected_state.index_plan.to_public_index().tolist() == ["a", "a", "b"]


def test_geoseries_sort_index_ignore_index_preserves_private_native_state() -> None:
    series = _native_backed_geoseries_from_arrow(index=pd.Index(["b", "a", "c"], name="site"))

    selected = series.sort_index(ignore_index=True)
    selected_state = get_native_state(selected)

    assert selected.index.equals(pd.RangeIndex(3))
    assert [str(value) for value in selected] == [
        "POINT (1 1)",
        "POINT (0 0)",
        "POINT (2 2)",
    ]
    assert selected_state is not None
    assert selected_state.index_plan.kind == "range"


def test_geoseries_metadata_relabels_preserve_private_native_state() -> None:
    series = _native_backed_geoseries_from_arrow(index=pd.Index(["a", "b", "c"], name="site"))

    renamed = series.rename("geom")
    mapped = renamed.rename(index={"a": "x", "b": "y", "c": "z"})
    axis_named = mapped.rename_axis("station")
    axis_set = axis_named.set_axis(pd.Index(["u", "v", "w"], name="stop"))

    renamed_state = get_native_state(renamed)
    mapped_state = get_native_state(mapped)
    axis_named_state = get_native_state(axis_named)
    axis_set_state = get_native_state(axis_set)

    assert renamed.name == "geom"
    assert renamed_state is not None
    assert renamed_state.geometry_name == "geom"
    assert renamed_state.column_order == ("geom",)
    assert mapped.index.tolist() == ["x", "y", "z"]
    assert mapped_state is not None
    assert mapped_state.index_plan.to_public_index().tolist() == ["x", "y", "z"]
    assert axis_named.index.name == "station"
    assert axis_named_state is not None
    assert axis_named_state.index_plan.to_public_index().name == "station"
    assert axis_set.index.tolist() == ["u", "v", "w"]
    assert axis_set.index.name == "stop"
    assert axis_set_state is not None
    assert axis_set_state.index_plan.to_public_index().tolist() == ["u", "v", "w"]


def test_geoseries_composition_device_state_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native GeoSeries composition")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    series = _native_backed_geoseries_from_arrow()
    state = get_native_state(series)
    state.geometry.owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="unit test native geoseries composition device geometry",
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        selected = series.iloc[[1, 2]]
        selected = selected.loc[pd.Series([True, True], index=selected.index)]
        projected = selected.set_crs("EPSG:3857", allow_override=True)
        relabeled = projected.rename("geom").rename_axis("site")

    relabeled_state = get_native_state(relabeled)
    assert relabeled_state is not None
    assert relabeled_state.geometry.crs == "EPSG:3857"
    assert relabeled_state.geometry_name == "geom"
    assert relabeled_state.row_count == 2
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_native_attribute_table_rename_columns_preserves_device_payload() -> None:
    table = NativeAttributeTable(
        device_table=_FakeDeviceTable(),
        index_override=pd.RangeIndex(2),
        column_override=("value", "name"),
        schema_override=pa.schema(
            [
                pa.field("value", pa.int64()),
                pa.field("name", pa.string()),
            ]
        ),
    )

    renamed = table.rename_columns({"value": "score"})

    assert renamed.device_table is table.device_table
    assert tuple(renamed.columns) == ("score", "name")
    assert renamed.schema_override.names == ["score", "name"]


def test_native_attribute_table_rename_columns_preserves_multiindex_labels() -> None:
    columns = pd.MultiIndex.from_tuples(
        [("metrics", "value"), ("labels", "name")],
        names=["kind", "field"],
    )
    table = NativeAttributeTable(dataframe=pd.DataFrame([[1, "a"]], columns=columns))

    renamed = table.rename_columns({("metrics", "value"): ("metrics", "score")})

    assert isinstance(renamed.columns, pd.MultiIndex)
    assert renamed.columns.names == ["kind", "field"]
    assert renamed.columns.tolist() == [("metrics", "score"), ("labels", "name")]


def test_native_attribute_table_assign_columns_preserves_arrow_payload() -> None:
    table = NativeAttributeTable(
        arrow_table=pa.table({"value": [1, 2]}),
        index_override=pd.RangeIndex(2),
        column_override=("value",),
    )

    assigned = table.assign_columns(
        {"score": pd.Series([10, 20])},
        columns=("value", "score"),
    )

    assert assigned is not None
    assert assigned.arrow_table is not None
    assert tuple(assigned.columns) == ("value", "score")
    assert assigned.to_pandas()["score"].tolist() == [10, 20]


def test_native_attribute_table_assign_columns_partitions_device_payload() -> None:
    table = NativeAttributeTable(
        device_table=_FakeDeviceTable(),
        index_override=pd.RangeIndex(2),
        column_override=("value", "name"),
        schema_override=pa.schema(
            [
                pa.field("value", pa.int64()),
                pa.field("name", pa.string()),
            ]
        ),
    )

    assigned = table.assign_columns(
        {"score": pd.Series([object(), object()], dtype=object)},
        columns=("value", "name", "score"),
    )

    assert assigned is not None
    assert assigned.parts is not None
    assert tuple(assigned.columns) == ("value", "name", "score")
    assert assigned.parts[0].device_table is not None
    assert assigned.parts[1].dataframe is not None


def test_native_attribute_table_assign_expression_columns_preserves_device_payload() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device expression attribute assignment")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    source = pa.table({"value": pa.array([1, 2], type=pa.int64())})
    table = NativeAttributeTable(
        device_table=plc.Table.from_arrow(source),
        index_override=pd.RangeIndex(2),
        column_override=("value",),
        schema_override=source.schema,
    )
    expression = NativeExpression(
        operation="geometry.area",
        values=cp.asarray([1.5, 2.5], dtype=cp.float64),
        source_row_count=2,
        dtype="float64",
        precision="fp64",
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        assigned = table.assign_columns(
            {"area": expression},
            columns=("value", "area"),
        )
        assert assigned is not None
        arrays = assigned.numeric_column_arrays(["value", "area"])

    assert assigned.device_table is not None
    assert tuple(assigned.columns) == ("value", "area")
    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [1, 2]
    assert cp.asnumpy(arrays["area"]).tolist() == [1.5, 2.5]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_native_attribute_loader_with_expression_materializes_values() -> None:
    table = NativeAttributeTable.from_loader(
        lambda: pd.DataFrame({"value": [1, 2]}),
        index_override=pd.RangeIndex(2),
        columns=("value",),
    )
    expression = NativeExpression(
        operation="geometry.area",
        values=np.asarray([1.5, 2.5], dtype=np.float64),
        source_row_count=2,
        dtype="float64",
        precision="fp64",
    )

    appended = table.with_column("area", expression)
    frame = appended.to_pandas(copy=False)

    assert tuple(appended.columns) == ("value", "area")
    assert frame["area"].tolist() == [1.5, 2.5]
    assert appended.to_arrow(index=False).column("area").to_pylist() == [1.5, 2.5]


def test_native_attribute_mixed_host_device_assignment_preserves_expression_payload() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for mixed native attribute assignment")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    table = NativeAttributeTable(
        dataframe=pd.DataFrame(
            {
                "label": ["a", "b"],
                "value": [1, 2],
            }
        )
    )
    expression = NativeExpression(
        operation="geometry.area",
        values=cp.asarray([1.5, 2.5], dtype=cp.float64),
        source_row_count=2,
        dtype="float64",
        precision="fp64",
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        assigned = table.assign_columns(
            {"area": expression},
            columns=("label", "value", "area"),
        )
        arrays = assigned.numeric_column_arrays(["area"])

    assert assigned is not None
    assert assigned.parts is not None
    assert tuple(assigned.columns) == ("label", "value", "area")
    assert arrays is not None
    assert cp.asnumpy(arrays["area"]).tolist() == [1.5, 2.5]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_native_attribute_mixed_host_assignment_preserves_device_part() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for mixed native attribute assignment")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    table = NativeAttributeTable(dataframe=pd.DataFrame({"label": ["a", "b"]}))
    expression = NativeExpression(
        operation="geometry.area",
        values=cp.asarray([1.5, 2.5], dtype=cp.float64),
        source_row_count=2,
        dtype="float64",
        precision="fp64",
    )
    mixed = table.assign_columns(
        {"area": expression},
        columns=("label", "area"),
    )
    assert mixed is not None
    assert mixed.parts is not None
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        assigned = mixed.assign_columns(
            {"parcel_group": pd.Series(["0", "1"])},
            columns=("label", "area", "parcel_group"),
        )
        arrays = assigned.numeric_column_arrays(["area"])

    assert assigned is not None
    assert assigned.parts is not None
    assert tuple(assigned.columns) == ("label", "area", "parcel_group")
    host_part = next(part for part in assigned.parts if "parcel_group" in tuple(part.columns))
    assert host_part.dataframe is not None
    assert host_part.dataframe["parcel_group"].tolist() == ["0", "1"]
    assert arrays is not None
    assert cp.asnumpy(arrays["area"]).tolist() == [1.5, 2.5]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_native_attribute_table_assign_numeric_columns_preserves_device_payload() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device attribute assignment")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    source = pa.table({"value": pa.array([1, 2], type=pa.int64())})
    table = NativeAttributeTable(
        device_table=plc.Table.from_arrow(source),
        index_override=pd.RangeIndex(2),
        column_override=("value",),
        schema_override=source.schema,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        assigned = table.assign_columns(
            {
                "score": pd.Series([10, 20]),
                "flag": pd.Series([True, False]),
            },
            columns=("value", "score", "flag"),
        )
        assert assigned is not None
        arrays = assigned.numeric_column_arrays(("value", "score", "flag"))

    assert assigned.device_table is not None
    assert tuple(assigned.columns) == ("value", "score", "flag")
    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [1, 2]
    assert cp.asnumpy(arrays["score"]).tolist() == [10, 20]
    assert cp.asnumpy(arrays["flag"]).tolist() == [True, False]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_native_attribute_table_with_column_preserves_device_payload_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device attribute append")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    source = pa.table({"value": pa.array([1, 2], type=pa.int64())})
    table = NativeAttributeTable(
        device_table=plc.Table.from_arrow(source),
        index_override=pd.RangeIndex(2),
        column_override=("value",),
        schema_override=source.schema,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        appended = table.with_column("score", pd.Series([10, 20]))
        arrays = appended.numeric_column_arrays(("value", "score"))

    assert appended.device_table is not None
    assert tuple(appended.columns) == ("value", "score")
    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [1, 2]
    assert cp.asnumpy(arrays["score"]).tolist() == [10, 20]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_native_attribute_table_with_column_unsupported_device_payload_exports_observably() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device attribute append fallback")
    plc = pytest.importorskip("pylibcudf")

    source = pa.table({"value": pa.array([1, 2], type=pa.int64())})
    table = NativeAttributeTable(
        device_table=plc.Table.from_arrow(source),
        index_override=pd.RangeIndex(2),
        column_override=("value",),
        schema_override=source.schema,
    )
    clear_materialization_events()

    labels = [object(), object()]
    appended = table.with_column("label", labels)

    assert appended.parts is not None
    exported_labels = appended.to_pandas()["label"].tolist()
    assert exported_labels[0] is labels[0]
    assert exported_labels[1] is labels[1]
    events = get_materialization_events(clear=True)
    assert any(
        event.boundary is MaterializationBoundary.USER_EXPORT and event.d2h_transfer
        for event in events
    )


def test_geodataframe_rename_attribute_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    renamed = gdf.rename(columns={"value": "score"})
    renamed_state = get_native_state(renamed)

    assert renamed.columns.tolist() == ["score", "name", "geometry"]
    assert renamed_state is not None
    assert renamed_state.column_order == ("score", "name", "geometry")
    assert renamed_state.geometry_name == "geometry"
    assert renamed_state.to_native_tabular_result().attributes.to_pandas()["score"].tolist() == [
        1,
        2,
    ]


def test_geodataframe_rename_index_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()
    clear_materialization_events()

    renamed = gdf.rename(index={0: "left", 1: "right"})
    renamed_state = get_native_state(renamed)

    assert renamed.index.tolist() == ["left", "right"]
    assert renamed_state is not None
    assert renamed_state.column_order == ("value", "name", "geometry")
    assert renamed_state.index_plan.to_public_index().tolist() == ["left", "right"]
    assert get_materialization_events(clear=True) == []


def test_geodataframe_rename_geometry_column_without_active_update_drops_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    renamed = gdf.rename(columns={"geometry": "geom"})

    assert renamed.columns.tolist() == ["value", "name", "geom"]
    assert renamed._geometry_column_name == "geometry"
    assert get_native_state(renamed) is None


def test_geodataframe_rename_geometry_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    renamed = gdf.rename_geometry("geom")
    renamed_state = get_native_state(renamed)

    assert renamed.columns.tolist() == ["value", "name", "geom"]
    assert renamed.geometry.name == "geom"
    assert renamed_state is not None
    assert renamed_state.column_order == ("value", "name", "geom")
    assert renamed_state.geometry_name == "geom"


def test_geodataframe_rename_duplicate_columns_drops_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    renamed = gdf.rename(columns={"value": "name"})

    assert renamed.columns.tolist() == ["name", "name", "geometry"]
    assert get_native_state(renamed) is None


def test_geodataframe_rename_inplace_clears_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    result = gdf.rename(columns={"value": "score"}, inplace=True)

    assert result is None
    assert gdf.columns.tolist() == ["score", "name", "geometry"]
    assert get_native_state(gdf) is None


def test_geodataframe_rename_axis_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    renamed = gdf.rename_axis("site")
    renamed_state = get_native_state(renamed)

    assert renamed.index.name == "site"
    assert renamed_state is not None
    assert renamed_state.index_plan.name == "site"
    assert renamed_state.column_order == ("value", "name", "geometry")


def test_geodataframe_rename_axis_columns_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    renamed = gdf.rename_axis("field", axis=1)
    renamed_state = get_native_state(renamed)

    assert renamed.columns.name == "field"
    assert renamed_state is not None
    assert renamed_state.column_order == ("value", "name", "geometry")
    assert renamed_state.index_plan.to_public_index().equals(renamed.index)


def test_geodataframe_set_axis_columns_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    relabeled = gdf.set_axis(["score", "name", "geometry"], axis=1)
    relabeled_state = get_native_state(relabeled)

    assert relabeled.columns.tolist() == ["score", "name", "geometry"]
    assert relabeled_state is not None
    assert relabeled_state.column_order == ("score", "name", "geometry")
    assert relabeled_state.to_native_tabular_result().attributes.to_pandas()["score"].tolist() == [
        1,
        2,
    ]


def test_geodataframe_set_axis_index_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    relabeled = gdf.set_axis(pd.Index(["x", "y"], name="site"), axis=0)
    relabeled_state = get_native_state(relabeled)

    assert relabeled.index.tolist() == ["x", "y"]
    assert relabeled.index.name == "site"
    assert relabeled_state is not None
    assert relabeled_state.index_plan.kind == "host-labels"
    assert relabeled_state.index_plan.name == "site"


def test_geodataframe_reset_index_drop_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [2, 1],
            "geometry": GeoSeries.from_wkt(
                ["POINT (2 2)", "POINT (1 1)"],
                name="geometry",
            ),
        },
        index=pd.Index(["b", "a"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    reset = gdf.reset_index(drop=True)
    reset_state = get_native_state(reset)

    assert reset.index.equals(pd.RangeIndex(2))
    assert reset_state is not None
    assert reset_state.index_plan.kind == "range"
    assert reset_state.to_native_tabular_result().attributes.index.equals(pd.RangeIndex(2))


def test_geodataframe_reset_index_without_drop_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [2, 1],
            "geometry": GeoSeries.from_wkt(
                ["POINT (2 2)", "POINT (1 1)"],
                name="geometry",
            ),
        },
        index=pd.Index(["b", "a"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    reset = gdf.reset_index()

    assert reset.columns.tolist() == ["site", "value", "geometry"]
    reset_state = get_native_state(reset)
    assert reset_state is not None
    assert reset_state.index_plan.kind == "range"
    assert reset_state.column_order == ("site", "value", "geometry")
    assert reset_state.to_native_tabular_result().attributes.to_pandas()["site"].tolist() == [
        "b",
        "a",
    ]


def test_geodataframe_reset_multiindex_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [2, 1],
            "geometry": GeoSeries.from_wkt(
                ["POINT (2 2)", "POINT (1 1)"],
                name="geometry",
            ),
        },
        index=pd.MultiIndex.from_tuples(
            [("b", 2), ("a", 1)],
            names=["site", "rank"],
        ),
    )
    _attach_native_tabular_state(gdf)

    reset = gdf.reset_index()
    reset_state = get_native_state(reset)

    assert reset.columns.tolist() == ["site", "rank", "value", "geometry"]
    assert reset_state is not None
    attrs = reset_state.to_native_tabular_result().attributes.to_pandas()
    assert reset_state.index_plan.kind == "range"
    assert reset_state.column_order == ("site", "rank", "value", "geometry")
    assert attrs["site"].tolist() == ["b", "a"]
    assert attrs["rank"].tolist() == [2, 1]


def test_geodataframe_reset_multiindex_level_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [2, 1],
            "geometry": GeoSeries.from_wkt(
                ["POINT (2 2)", "POINT (1 1)"],
                name="geometry",
            ),
        },
        index=pd.MultiIndex.from_tuples(
            [("b", 2), ("a", 1)],
            names=["site", "rank"],
        ),
    )
    _attach_native_tabular_state(gdf)

    reset = gdf.reset_index(level="rank")
    reset_state = get_native_state(reset)

    assert reset.index.tolist() == ["b", "a"]
    assert reset.index.name == "site"
    assert reset.columns.tolist() == ["rank", "value", "geometry"]
    assert reset_state is not None
    assert reset_state.index_plan.name == "site"
    assert reset_state.column_order == ("rank", "value", "geometry")
    attrs = reset_state.to_native_tabular_result().attributes.to_pandas()
    assert attrs.index.tolist() == ["b", "a"]
    assert attrs["rank"].tolist() == [2, 1]
    assert attrs["value"].tolist() == [2, 1]


def test_geodataframe_reset_multiindex_level_drop_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [2, 1],
            "geometry": GeoSeries.from_wkt(
                ["POINT (2 2)", "POINT (1 1)"],
                name="geometry",
            ),
        },
        index=pd.MultiIndex.from_tuples(
            [("b", 2), ("a", 1)],
            names=["site", "rank"],
        ),
    )
    _attach_native_tabular_state(gdf)

    reset = gdf.reset_index(level="rank", drop=True)
    reset_state = get_native_state(reset)

    assert reset.index.tolist() == ["b", "a"]
    assert reset.index.name == "site"
    assert reset.columns.tolist() == ["value", "geometry"]
    assert reset_state is not None
    assert reset_state.index_plan.name == "site"
    assert reset_state.column_order == ("value", "geometry")
    attrs = reset_state.to_native_tabular_result().attributes.to_pandas()
    assert attrs.index.tolist() == ["b", "a"]
    assert attrs["value"].tolist() == [2, 1]


def test_geodataframe_device_attribute_reset_index_preserves_private_state_without_runtime_d2h() -> (
    None
):
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device reset-index attribute")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    gdf = GeoDataFrame(
        {
            "value": [2, 1],
            "flag": [True, False],
            "geometry": GeoSeries.from_wkt(
                ["POINT (2 2)", "POINT (1 1)"],
                name="geometry",
            ),
        },
        index=pd.RangeIndex(10, 12, name="site"),
    )
    _attach_owned_native_tabular_state(gdf, attribute_storage="device")
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        reset = gdf.reset_index()
        reset_state = get_native_state(reset)
        assert reset_state is not None
        arrays = reset_state.attributes.numeric_column_arrays(("site", "value", "flag"))

    assert reset_state.attributes.device_table is not None
    assert reset_state.column_order == ("site", "value", "flag", "geometry")
    assert arrays is not None
    assert cp.asnumpy(arrays["site"]).tolist() == [10, 11]
    assert cp.asnumpy(arrays["value"]).tolist() == [2, 1]
    assert cp.asnumpy(arrays["flag"]).tolist() == [True, False]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_geodataframe_reset_index_inplace_clears_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    result = gdf.reset_index(drop=True, inplace=True)

    assert result is None
    assert gdf.index.equals(pd.RangeIndex(2))
    assert get_native_state(gdf) is None


def test_geodataframe_set_index_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    indexed = gdf.set_index("name")
    indexed_state = get_native_state(indexed)

    assert indexed.index.tolist() == ["a", "b"]
    assert indexed.index.name == "name"
    assert indexed.columns.tolist() == ["value", "geometry"]
    assert indexed_state is not None
    assert indexed_state.index_plan.kind == "host-labels"
    assert indexed_state.index_plan.name == "name"
    assert indexed_state.column_order == ("value", "geometry")
    assert indexed_state.to_native_tabular_result().attributes.index.tolist() == [
        "a",
        "b",
    ]


def test_geodataframe_set_index_drop_false_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    indexed = gdf.set_index("name", drop=False)
    indexed_state = get_native_state(indexed)

    assert indexed.index.tolist() == ["a", "b"]
    assert indexed.columns.tolist() == ["value", "name", "geometry"]
    assert indexed_state is not None
    assert indexed_state.column_order == ("value", "name", "geometry")


def test_geodataframe_set_index_duplicate_labels_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "zone": ["same", "same"],
            "value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (1 1)", "POINT (2 2)"],
                name="geometry",
            ),
        }
    )
    _attach_native_tabular_state(gdf)

    indexed = gdf.set_index("zone")
    indexed_state = get_native_state(indexed)

    assert indexed.index.tolist() == ["same", "same"]
    assert indexed_state is not None
    assert indexed_state.index_plan.has_duplicates is True
    assert indexed_state.column_order == ("value", "geometry")
    assert indexed_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        2,
    ]


def test_geodataframe_set_index_multiindex_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    indexed = gdf.set_index(["name", "value"])
    indexed_state = get_native_state(indexed)

    assert indexed.index.nlevels == 2
    assert indexed.index.tolist() == [("a", 1), ("b", 2)]
    assert indexed_state is not None
    assert indexed_state.index_plan.nlevels == 2
    assert indexed_state.column_order == ("geometry",)


def test_geodataframe_set_index_append_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    indexed = gdf.set_index("name", append=True)
    indexed_state = get_native_state(indexed)

    assert indexed.index.nlevels == 2
    assert indexed.index.tolist() == [(0, "a"), (1, "b")]
    assert indexed_state is not None
    assert indexed_state.index_plan.nlevels == 2
    assert indexed_state.column_order == ("value", "geometry")


def test_geodataframe_set_index_geometry_drops_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    indexed = gdf.set_index("geometry", drop=False)

    assert "geometry" in indexed.columns
    assert get_native_state(indexed) is None


def test_geodataframe_set_index_inplace_clears_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    result = gdf.set_index("name", inplace=True)

    assert result is None
    assert gdf.index.tolist() == ["a", "b"]
    assert get_native_state(gdf) is None


def test_geodataframe_reindex_rows_preserves_unique_index_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)"],
                name="geometry",
            ),
        },
        index=pd.Index(["a", "b", "c"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    reindexed = gdf.reindex(["c", "a"])
    reindexed_state = get_native_state(reindexed)

    assert reindexed.index.tolist() == ["c", "a"]
    assert reindexed["value"].tolist() == [3, 1]
    assert reindexed_state is not None
    assert reindexed_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        3,
        1,
    ]


def test_geodataframe_reindex_columns_preserves_projected_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    reindexed = gdf.reindex(columns=["geometry", "value"])
    reindexed_state = get_native_state(reindexed)

    assert reindexed.columns.tolist() == ["geometry", "value"]
    assert reindexed_state is not None
    assert reindexed_state.column_order == ("geometry", "value")


def test_geodataframe_reindex_rows_and_columns_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "name": ["a", "b", "c"],
            "geometry": GeoSeries.from_wkt(
                ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)"],
                name="geometry",
            ),
        },
        index=pd.Index(["a", "b", "c"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    reindexed = gdf.reindex(index=["c", "a"], columns=["geometry", "value"])
    reindexed_state = get_native_state(reindexed)

    assert reindexed.index.tolist() == ["c", "a"]
    assert reindexed.columns.tolist() == ["geometry", "value"]
    assert reindexed_state is not None
    assert reindexed_state.column_order == ("geometry", "value")
    assert reindexed_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        3,
        1,
    ]


def test_geodataframe_reindex_missing_row_drops_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    reindexed = gdf.reindex([1, 99])

    assert reindexed.index.tolist() == [1, 99]
    assert get_native_state(reindexed) is None


def test_geodataframe_reindex_missing_column_drops_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    reindexed = gdf.reindex(columns=["value", "geometry", "missing"])

    assert reindexed.columns.tolist() == ["value", "geometry", "missing"]
    assert get_native_state(reindexed) is None


def test_geodataframe_reindex_duplicate_target_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (1 1)", "POINT (2 2)"],
                name="geometry",
            ),
        },
        index=pd.Index(["a", "b"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    reindexed = gdf.reindex(["b", "b"])
    reindexed_state = get_native_state(reindexed)

    assert reindexed.index.tolist() == ["b", "b"]
    assert reindexed["value"].tolist() == [2, 2]
    assert reindexed_state is not None
    assert reindexed_state.index_plan.has_duplicates is True
    assert reindexed_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        2,
        2,
    ]


def test_geodataframe_reindex_like_preserves_exact_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "name": ["a", "b", "c"],
            "geometry": GeoSeries.from_wkt(
                ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)"],
                name="geometry",
            ),
        },
        index=pd.Index(["a", "b", "c"], name="site"),
    )
    _attach_native_tabular_state(gdf)
    other = GeoDataFrame(
        {
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
            "value": [0, 0],
        },
        index=pd.Index(["c", "a"], name="site"),
    )

    reindexed = gdf.reindex_like(other)
    reindexed_state = get_native_state(reindexed)

    assert reindexed.index.tolist() == ["c", "a"]
    assert reindexed.columns.tolist() == ["geometry", "value"]
    assert reindexed_state is not None
    assert reindexed_state.column_order == ("geometry", "value")


def test_geodataframe_filter_columns_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    filtered = gdf.filter(items=["geometry", "value"])
    filtered_state = get_native_state(filtered)

    assert filtered.columns.tolist() == ["geometry", "value"]
    assert filtered_state is not None
    assert filtered_state.column_order == ("geometry", "value")


def test_geodataframe_filter_rows_preserves_unique_index_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)"],
                name="geometry",
            ),
        },
        index=pd.Index(["a", "b", "c"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    filtered = gdf.filter(regex="b|c", axis=0)
    filtered_state = get_native_state(filtered)

    assert filtered.index.tolist() == ["b", "c"]
    assert filtered_state is not None
    assert filtered_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        2,
        3,
    ]


def test_geodataframe_filter_without_geometry_drops_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    filtered = gdf.filter(items=["value", "name"])

    assert filtered.columns.tolist() == ["value", "name"]
    assert get_native_state(filtered) is None


def test_geodataframe_select_dtypes_geometry_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    selected = gdf.select_dtypes(include=["geometry"])
    selected_state = get_native_state(selected)

    assert selected.columns.tolist() == ["geometry"]
    assert selected_state is not None
    assert selected_state.column_order == ("geometry",)
    assert selected_state.row_count == len(gdf)


def test_geodataframe_select_dtypes_without_geometry_drops_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    selected = gdf.select_dtypes(include=["number"])

    assert selected.columns.tolist() == ["value"]
    assert get_native_state(selected) is None


def test_geodataframe_assign_new_attribute_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    assigned = gdf.assign(score=[10, 20])
    assigned_state = get_native_state(assigned)

    assert assigned.columns.tolist() == ["value", "name", "geometry", "score"]
    assert assigned_state is not None
    assert assigned_state.column_order == ("value", "name", "geometry", "score")
    assert assigned_state.to_native_tabular_result().attributes.to_pandas()["score"].tolist() == [
        10,
        20,
    ]


def test_geodataframe_assign_replace_attribute_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    assigned = gdf.assign(value=[10, 20])
    assigned_state = get_native_state(assigned)

    assert assigned["value"].tolist() == [10, 20]
    assert assigned_state is not None
    assert assigned_state.column_order == ("value", "name", "geometry")
    assert assigned_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        10,
        20,
    ]


def test_geodataframe_assign_callable_attribute_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    assigned = gdf.assign(score=lambda frame: frame["value"] + 10)
    assigned_state = get_native_state(assigned)

    assert assigned["score"].tolist() == [11, 12]
    assert assigned_state is not None
    assert assigned_state.to_native_tabular_result().attributes.to_pandas()["score"].tolist() == [
        11,
        12,
    ]


def test_geodataframe_assign_geometry_drops_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    assigned = gdf.assign(
        geometry=GeoSeries.from_wkt(
            ["POINT (2 2)", "POINT (3 3)"],
            name="geometry",
        )
    )

    assert assigned.geometry.to_wkt().tolist() == ["POINT (2 2)", "POINT (3 3)"]
    assert get_native_state(assigned) is None


def test_geodataframe_assign_native_geometry_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()
    owned = from_shapely_geometries([Point(2, 2), Point(3, 3)])
    native_geometry = GeoSeries(GeometryArray.from_owned(owned), name="geometry")

    assigned = gdf.assign(geometry=native_geometry)
    assigned_state = get_native_state(assigned)

    assert assigned.geometry.to_wkt().tolist() == ["POINT (2 2)", "POINT (3 3)"]
    assert assigned_state is not None
    assert assigned_state.geometry.owned is owned
    assert assigned_state.column_order == ("value", "name", "geometry")


def test_geodataframe_assign_partitions_device_attribute_state() -> None:
    gdf, _state = _native_backed_geodataframe()
    device_attributes = NativeAttributeTable(
        device_table=_FakeDeviceTable(),
        index_override=gdf.index,
        column_override=("value", "name"),
        schema_override=pa.schema(
            [
                pa.field("value", pa.int64()),
                pa.field("name", pa.string()),
            ]
        ),
    )
    result = NativeTabularResult(
        attributes=device_attributes,
        geometry=GeometryNativeResult.from_geoseries(gdf.geometry),
        geometry_name="geometry",
        column_order=("value", "name", "geometry"),
    )
    attach_native_state(gdf, NativeFrameState.from_native_tabular_result(result))

    assigned_values = [object(), object()]
    assigned = gdf.assign(score=assigned_values)

    assigned_state = get_native_state(assigned)
    assert assigned_state is not None
    assert assigned_state.attributes.parts is not None
    assert assigned["score"].tolist() == assigned_values


@pytest.mark.parametrize("operation", ["assign", "setitem", "insert"])
def test_geodataframe_device_attribute_numeric_assignment_preserves_private_state_without_runtime_d2h(
    operation: str,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device attribute assignment")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    gdf = GeoDataFrame(
        {
            "value": [1, 2],
            "flag": [True, False],
            "geometry": GeoSeries.from_wkt(
                ["POINT (1 1)", "POINT (2 2)"],
                name="geometry",
            ),
        }
    )
    _attach_owned_native_tabular_state(gdf, attribute_storage="device")
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        if operation == "assign":
            target = gdf.assign(score=[10, 20])
            expected_order = ("value", "flag", "geometry", "score")
        elif operation == "setitem":
            gdf["score"] = [10, 20]
            target = gdf
            expected_order = ("value", "flag", "geometry", "score")
        else:
            gdf.insert(1, "score", [10, 20])
            target = gdf
            expected_order = ("value", "score", "flag", "geometry")
        state = get_native_state(target)
        assert state is not None
        arrays = state.attributes.numeric_column_arrays(("value", "flag", "score"))

    assert state.attributes.device_table is not None
    assert state.column_order == expected_order
    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [1, 2]
    assert cp.asnumpy(arrays["flag"]).tolist() == [True, False]
    assert cp.asnumpy(arrays["score"]).tolist() == [10, 20]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_attribute_cast_accepts_native_numeric_expression_dtype() -> None:
    from vibespatial.api._native_public_arrays import (
        NativeAttributeColumnArray,
        NativeNumericExpressionArray,
        NativeNumericExpressionDtype,
    )

    source = NativeAttributeColumnArray(
        NativeAttributeTable(dataframe=pd.DataFrame({"value": [1, 2]})),
        "value",
    )

    cast = source.astype(NativeNumericExpressionDtype(np.float64))

    assert isinstance(cast, NativeNumericExpressionArray)
    assert cast.dtype == np.dtype("float64")
    assert cast.to_numpy().tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    ("values", "dtype", "expected"),
    [
        ([1, None, 3], pd.Int64Dtype(), [1, pd.NA, 3]),
        ([True, None, False], pd.BooleanDtype(), [True, pd.NA, False]),
        (["a", None, "c"], pd.StringDtype(), ["a", pd.NA, "c"]),
    ],
)
def test_attribute_cast_accepts_pandas_extension_dtype(
    values: list[object],
    dtype: object,
    expected: list[object],
) -> None:
    from vibespatial.api._native_public_arrays import NativeAttributeColumnArray

    source = NativeAttributeColumnArray(
        NativeAttributeTable(dataframe=pd.DataFrame({"value": values})),
        "value",
    )

    cast = source.astype(dtype)

    assert cast.dtype == dtype
    assert cast.equals(pd.array(expected, dtype=dtype))


@pytest.mark.parametrize("use_native_dtype", [False, True])
def test_device_decimal_attribute_cast_preserves_lineage_without_runtime_d2h(
    use_native_dtype: bool,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device attribute cast")
    pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.api._native_public_arrays import (
        NativeAttributeColumnArray,
        NativeNumericExpressionArray,
        NativeNumericExpressionDtype,
    )
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        pylibcudf_column_from_arrow,
        reset_d2h_transfer_count,
    )

    values = pa.array(
        [Decimal("1.25"), Decimal("2.50")],
        type=pa.decimal128(8, 2),
    )
    table = NativeAttributeTable(
        device_table=pytest.importorskip("pylibcudf").Table(
            [pylibcudf_column_from_arrow(values)]
        ),
        column_override=("distance",),
        schema_override=pa.schema([pa.field("distance", values.type)]),
    )
    source = NativeAttributeColumnArray(
        table,
        "distance",
        source_token="frame-lineage",
    )
    reset_d2h_transfer_count()

    dtype = (
        NativeNumericExpressionDtype(np.float64)
        if use_native_dtype
        else np.float64
    )

    with assert_zero_d2h_transfers():
        cast = source.astype(dtype)

    assert isinstance(cast, NativeNumericExpressionArray)
    assert cast.expression.source_token == "frame-lineage"
    reset_d2h_transfer_count()


def test_geodataframe_sort_values_preserves_unique_index_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [2, 1],
            "geometry": GeoSeries.from_wkt(
                ["POINT (2 2)", "POINT (1 1)"],
                name="geometry",
            ),
        },
        index=pd.Index(["b", "a"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    sorted_gdf = gdf.sort_values("value")
    sorted_state = get_native_state(sorted_gdf)

    assert sorted_gdf.index.tolist() == ["a", "b"]
    assert sorted_gdf["value"].tolist() == [1, 2]
    assert sorted_state is not None
    assert sorted_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        2,
    ]


def test_geodataframe_sort_values_ignore_index_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    sorted_gdf = gdf.sort_values("value", ascending=False, ignore_index=True)
    sorted_state = get_native_state(sorted_gdf)

    assert sorted_gdf.index.equals(pd.RangeIndex(2))
    assert sorted_gdf["value"].tolist() == [2, 1]
    assert sorted_state is not None
    assert sorted_state.index_plan.kind == "range"
    assert sorted_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        2,
        1,
    ]


def test_geodataframe_sort_index_ignore_index_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [2, 1],
            "geometry": GeoSeries.from_wkt(
                ["POINT (2 2)", "POINT (1 1)"],
                name="geometry",
            ),
        },
        index=pd.Index(["b", "a"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    sorted_gdf = gdf.sort_index(ignore_index=True)
    sorted_state = get_native_state(sorted_gdf)

    assert sorted_gdf.index.equals(pd.RangeIndex(2))
    assert sorted_gdf["value"].tolist() == [1, 2]
    assert sorted_state is not None
    assert sorted_state.index_plan.kind == "range"
    assert sorted_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        2,
    ]


def test_geodataframe_sort_index_duplicate_index_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [3, 2, 1],
            "geometry": GeoSeries.from_wkt(
                ["POINT (3 3)", "POINT (2 2)", "POINT (1 1)"],
                name="geometry",
            ),
        },
        index=pd.Index(["b", "a", "a"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    sorted_gdf = gdf.sort_index(kind="mergesort")
    sorted_state = get_native_state(sorted_gdf)

    assert sorted_gdf.index.tolist() == ["a", "a", "b"]
    assert sorted_gdf.columns.equals(gdf.columns)
    assert sorted_gdf["value"].tolist() == [2, 1, 3]
    assert sorted_state is not None
    assert sorted_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        2,
        1,
        3,
    ]


def test_geodataframe_sort_values_ignore_index_preserves_device_state_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native sort preservation")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    import pylibcudf as plc

    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [Point(2, 2), Point(0, 0), Point(1, 1)],
        residency=Residency.DEVICE,
    )
    arrow = pa.table({"score": pa.array([2.0, 0.0, 1.0], type=pa.float64())})
    gdf = GeoDataFrame(
        {
            "score": [2.0, 0.0, 1.0],
            "geometry": GeoSeries(
                GeometryArray.from_owned(owned, crs=None),
                name="geometry",
            ),
        }
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(arrow),
                index_override=pd.RangeIndex(3),
                column_override=tuple(arrow.column_names),
                schema_override=arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("score", "geometry"),
        )
    )
    attach_native_state(gdf, state)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        sorted_gdf = gdf.sort_values("score", ignore_index=True)
        sorted_state = get_native_state(sorted_gdf)
        arrays = sorted_state.attributes.numeric_column_arrays(("score",))

    assert sorted_state is not None
    assert sorted_state.index_plan.kind == "range"
    assert sorted_state.geometry.owned is not None
    assert sorted_state.geometry.owned.residency is Residency.DEVICE
    assert arrays is not None
    assert cp.asnumpy(arrays["score"]).tolist() == [0.0, 1.0, 2.0]
    assert get_materialization_events(clear=True) == []

    assert sorted_gdf.index.equals(pd.RangeIndex(3))
    assert sorted_gdf["score"].tolist() == [0.0, 1.0, 2.0]
    export_operations = [event.operation for event in get_materialization_events(clear=True)]
    assert export_operations == ["native_attribute_column_to_public_series"]
    reset_d2h_transfer_count()


def test_geodataframe_sort_values_lazy_native_columns_preserves_device_rowflow() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native lazy sort preservation")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    import pylibcudf as plc

    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        residency=Residency.DEVICE,
    )
    arrow = pa.table(
        {
            "parcel_id": pa.array([2, 1, 1, 2], type=pa.int64()),
            "flood_ratio": pa.array([0.1, 0.3, 0.7, 0.9], type=pa.float64()),
        }
    )
    gdf = GeoDataFrame(
        {
            "parcel_id": [2, 1, 1, 2],
            "flood_ratio": [0.1, 0.3, 0.7, 0.9],
            "geometry": GeoSeries(DeviceGeometryArray._from_owned(owned), name="geometry"),
        }
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(arrow),
                index_override=pd.RangeIndex(4),
                column_override=tuple(arrow.column_names),
                schema_override=arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("parcel_id", "flood_ratio", "geometry"),
        )
    )
    attach_native_state(gdf, state)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        sorted_gdf = gdf.sort_values(
            ["parcel_id", "flood_ratio"],
            ascending=[True, False],
        )
        sorted_state = get_native_state(sorted_gdf)
        parcel_values = sorted_state.attributes.numeric_column_arrays(("parcel_id",))
        ratio_values = sorted_state.attributes.numeric_column_arrays(("flood_ratio",))

    assert sorted_state is not None
    assert sorted_state.index_plan.kind == "device-labels"
    assert sorted_state.geometry.owned is not None
    assert sorted_state.geometry.owned.residency is Residency.DEVICE
    assert parcel_values is not None
    assert ratio_values is not None
    assert cp.asnumpy(parcel_values["parcel_id"]).tolist() == [1, 1, 2, 2]
    assert cp.asnumpy(ratio_values["flood_ratio"]).tolist() == [0.7, 0.3, 0.9, 0.1]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_geodataframe_sort_values_decimal_device_columns_preserves_device_rowflow() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native decimal sort preservation")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")

    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        residency=Residency.DEVICE,
    )
    arrow = pa.table(
        {
            "tripkey": pa.array([4, 3, 2, 1], type=pa.int64()),
            "tip": pa.array(
                [Decimal("1.00"), Decimal("3.50"), Decimal("3.50"), Decimal("2.00")],
                type=pa.decimal128(8, 2),
            ),
        }
    )
    gdf = GeoDataFrame(
        {
            "tripkey": [4, 3, 2, 1],
            "tip": [Decimal("1.00"), Decimal("3.50"), Decimal("3.50"), Decimal("2.00")],
            "geometry": GeoSeries(DeviceGeometryArray._from_owned(owned), name="geometry"),
        }
    )
    attach_native_state(
        gdf,
        NativeFrameState.from_native_tabular_result(
            NativeTabularResult(
                attributes=NativeAttributeTable(
                    device_table=plc.Table.from_arrow(arrow),
                    index_override=pd.RangeIndex(4),
                    column_override=tuple(arrow.column_names),
                    schema_override=arrow.schema,
                ),
                geometry=GeometryNativeResult.from_owned(owned, crs=None),
                geometry_name="geometry",
                column_order=("tripkey", "tip", "geometry"),
            )
        ),
    )
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        sorted_gdf = gdf.sort_values(
            ["tip", "tripkey"],
            ascending=[False, True],
        )
        sorted_state = get_native_state(sorted_gdf)
        positions = cp.asarray(sorted_state.attributes.row_positions)

    assert sorted_state is not None
    assert cp.asnumpy(positions).tolist() == [2, 1, 3, 0]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_geodataframe_sort_dedupe_composes_native_attribute_row_view() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native row-indirected attributes")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    import pylibcudf as plc

    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    geoms = [
        Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
        MultiPolygon(
            [
                Polygon([(10, 0), (11, 0), (11, 1), (10, 0)]),
                Polygon([(12, 0), (13, 0), (13, 1), (12, 0)]),
            ]
        ),
        Polygon(
            [(20, 0), (24, 0), (24, 4), (20, 0)],
            [[(21, 1), (22, 1), (22, 2), (21, 1)]],
        ),
        MultiPolygon(
            [
                Polygon(
                    [(30, 0), (34, 0), (34, 4), (30, 0)],
                    [[(31, 1), (32, 1), (32, 2), (31, 1)]],
                )
            ]
        ),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    arrow = pa.table(
        {
            "parcel_id": pa.array([2, 1, 1, 2], type=pa.int64()),
            "flood_ratio": pa.array([0.1, 0.3, 0.7, 0.9], type=pa.float64()),
            "zone_id": pa.array([5, 6, 7, 8], type=pa.int64()),
        }
    )
    gdf = GeoDataFrame(
        {
            "parcel_id": [2, 1, 1, 2],
            "flood_ratio": [0.1, 0.3, 0.7, 0.9],
            "zone_id": [5, 6, 7, 8],
            "geometry": GeoSeries(
                DeviceGeometryArray._from_owned(owned),
                name="geometry",
            ),
        }
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(arrow),
                index_override=pd.RangeIndex(4),
                column_override=tuple(arrow.column_names),
                schema_override=arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("parcel_id", "flood_ratio", "zone_id", "geometry"),
        )
    )
    attach_native_state(gdf, state)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        sorted_gdf = gdf.sort_values(
            ["parcel_id", "flood_ratio"],
            ascending=[True, False],
        )
        sorted_state = get_native_state(sorted_gdf)
        deduped = sorted_gdf.drop_duplicates("parcel_id")
        deduped_state = get_native_state(deduped)
        parcel_values = deduped_state.attributes.numeric_column_arrays(("parcel_id",))
        ratio_values = deduped_state.attributes.numeric_column_arrays(("flood_ratio",))

    assert sorted_state is not None
    assert sorted_state.attributes.row_positions is not None
    assert sorted_state.geometry.owned is not None
    assert sorted_state.geometry.owned.is_indexed_view
    assert deduped_state is not None
    assert deduped_state.attributes.row_positions is not None
    assert cp.asnumpy(deduped_state.attributes.row_positions).tolist() == [2, 3]
    assert deduped_state.geometry.owned is not None
    assert deduped_state.geometry.owned.is_indexed_view
    assert parcel_values is not None
    assert ratio_values is not None
    assert cp.asnumpy(parcel_values["parcel_id"]).tolist() == [1, 2]
    assert cp.asnumpy(ratio_values["flood_ratio"]).tolist() == [0.7, 0.9]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_geodataframe_sort_dedupe_accepts_composed_device_attribute_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for composed device attribute sorting")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    geodataframe_module = importlib.import_module("vibespatial.api.geodataframe")
    owned = from_shapely_geometries(
        [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        residency=Residency.DEVICE,
    )
    index = pd.RangeIndex(4)

    def _device_part(name: str, values) -> NativeAttributeTable:
        arrow = pa.table({name: values})
        return NativeAttributeTable(
            device_table=plc.Table.from_arrow(arrow),
            index_override=index,
            column_override=(name,),
            schema_override=arrow.schema,
        )

    attributes = NativeAttributeTable(
        parts=(
            _device_part("left_key", pa.array([2, 1, 1, 2], type=pa.int64())),
            _device_part("right_key", pa.array([3, 4, 2, 1], type=pa.int64())),
            _device_part("label", pa.array(["a", "b", None, "d"])),
        ),
        index_override=index,
        column_override=("left_key", "right_key", "label"),
    )
    gdf = GeoDataFrame(
        {
            "left_key": [2, 1, 1, 2],
            "right_key": [3, 4, 2, 1],
            "label": ["a", "b", None, "d"],
            "geometry": GeoSeries(DeviceGeometryArray._from_owned(owned), name="geometry"),
        }
    )
    attach_native_state(
        gdf,
        NativeFrameState.from_native_tabular_result(
            NativeTabularResult(
                attributes=attributes,
                geometry=GeometryNativeResult.from_owned(owned, crs=None),
                geometry_name="geometry",
                column_order=("left_key", "right_key", "label", "geometry"),
            )
        ),
    )
    monkeypatch.setattr(
        geodataframe_module,
        "_cupy_native_sort_positions",
        lambda *_args, **_kwargs: None,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()
    clear_dispatch_events()

    with assert_zero_d2h_transfers():
        sorted_gdf = gdf.sort_values(
            ["left_key", "right_key"],
            kind="stable",
        )
        sorted_state = get_native_state(sorted_gdf)
        deduped = sorted_gdf.drop_duplicates("left_key", keep="first")
        deduped_state = get_native_state(deduped)
        key_values = deduped_state.attributes.numeric_column_arrays(
            ("left_key", "right_key")
        )

    assert sorted_state is not None
    assert sorted_state.attributes.parts is not None
    assert deduped_state is not None
    assert deduped_state.attributes.parts is not None
    assert key_values is not None
    assert cp.asnumpy(key_values["left_key"]).tolist() == [1, 2]
    assert cp.asnumpy(key_values["right_key"]).tolist() == [2, 1]
    assert get_materialization_events(clear=True) == []
    assert [event.operation for event in get_dispatch_events(clear=True)] == [
        "sort_values",
        "drop_duplicates",
    ]
    reset_d2h_transfer_count()


def test_geodataframe_small_native_sort_uses_cupy_rowset_without_pylibcudf_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native lazy sort preservation")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    import pylibcudf as plc
    import pylibcudf.sorting as sorting

    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    def _fail_sort(*_args, **_kwargs):  # pragma: no cover - failure path
        raise AssertionError("small native sort should use the CuPy rowset path")

    monkeypatch.setattr(sorting, "stable_sorted_order", _fail_sort)
    monkeypatch.setattr(sorting, "sorted_order", _fail_sort)

    owned = from_shapely_geometries(
        [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        residency=Residency.DEVICE,
    )
    arrow = pa.table(
        {
            "parcel_id": pa.array([2, 1, 1, 2], type=pa.int64()),
            "flood_ratio": pa.array([0.1, 0.3, 0.7, 0.9], type=pa.float64()),
        }
    )
    gdf = GeoDataFrame(
        {
            "parcel_id": [2, 1, 1, 2],
            "flood_ratio": [0.1, 0.3, 0.7, 0.9],
            "geometry": GeoSeries(DeviceGeometryArray._from_owned(owned), name="geometry"),
        }
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=NativeAttributeTable(
                device_table=plc.Table.from_arrow(arrow),
                index_override=pd.RangeIndex(4),
                column_override=tuple(arrow.column_names),
                schema_override=arrow.schema,
            ),
            geometry=GeometryNativeResult.from_owned(owned, crs=None),
            geometry_name="geometry",
            column_order=("parcel_id", "flood_ratio", "geometry"),
        )
    )
    attach_native_state(gdf, state)
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        sorted_gdf = gdf.sort_values(
            ["parcel_id", "flood_ratio"],
            ascending=[True, False],
        )
        sorted_state = get_native_state(sorted_gdf)
        parcel_values = sorted_state.attributes.numeric_column_arrays(("parcel_id",))
        ratio_values = sorted_state.attributes.numeric_column_arrays(("flood_ratio",))

    assert sorted_state is not None
    assert parcel_values is not None
    assert ratio_values is not None
    assert cp.asnumpy(parcel_values["parcel_id"]).tolist() == [1, 1, 2, 2]
    assert cp.asnumpy(ratio_values["flood_ratio"]).tolist() == [0.7, 0.3, 0.9, 0.1]
    assert get_materialization_events(clear=True) == []
    assert sorted_gdf["parcel_id"].tolist() == [1, 1, 2, 2]
    reset_d2h_transfer_count()

@pytest.mark.parametrize(
    ("ascending", "na_position", "expected_positions"),
    [
        pytest.param(True, "last", [2, 0, 1, 3], id="ascending-nulls-last"),
        pytest.param(True, "first", [1, 3, 2, 0], id="ascending-nulls-first"),
        pytest.param(False, "last", [0, 2, 1, 3], id="descending-nulls-last"),
        pytest.param(False, "first", [1, 3, 0, 2], id="descending-nulls-first"),
    ],
)
def test_geodataframe_pylibcudf_sort_normalizes_float_nulls_and_nans(
    monkeypatch: pytest.MonkeyPatch,
    ascending: bool,
    na_position: str,
    expected_positions: list[int],
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for pylibcudf missing-value sort")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")
    from vibespatial.api import geodataframe as geodataframe_module
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    monkeypatch.setattr(geodataframe_module, "_NATIVE_SORT_CUPY_MAX_ROWS", 0)
    arrow = pa.table(
        {
            "key": pa.array([1.0, np.nan, 0.0, None], type=pa.float64()),
            "source_position": pa.array(range(4), type=pa.int64()),
        }
    )
    owned = from_shapely_geometries(
        [Point(position, position) for position in range(4)],
        residency=Residency.DEVICE,
    )
    gdf = GeoDataFrame(
        {
            "key": arrow["key"].to_pandas(),
            "source_position": range(4),
            "geometry": GeoSeries(DeviceGeometryArray._from_owned(owned), name="geometry"),
        }
    )
    attach_native_state(
        gdf,
        NativeFrameState.from_native_tabular_result(
            NativeTabularResult(
                attributes=NativeAttributeTable(
                    device_table=plc.Table.from_arrow(arrow),
                    index_override=pd.RangeIndex(4),
                    column_override=tuple(arrow.column_names),
                    schema_override=arrow.schema,
                ),
                geometry=GeometryNativeResult.from_owned(owned, crs=None),
                geometry_name="geometry",
                column_order=("key", "source_position", "geometry"),
            )
        ),
    )
    reset_d2h_transfer_count()
    clear_materialization_events()
    clear_fallback_events()

    with assert_zero_d2h_transfers():
        sorted_gdf = gdf.sort_values(
            "key",
            ascending=ascending,
            na_position=na_position,
            kind="stable",
        )
        sorted_state = get_native_state(sorted_gdf)
        assert sorted_state is not None
        result_positions = cp.asarray(sorted_state.attributes.row_positions)

    assert cp.asnumpy(result_positions).tolist() == expected_positions
    assert get_materialization_events(clear=True) == []
    assert get_fallback_events(clear=True) == []
    reset_d2h_transfer_count()


def test_geodataframe_native_tabular_declines_record_fallback_events() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device tabular fallback observability")
    plc = pytest.importorskip("pylibcudf")
    arrow = pa.table(
        {
            "value": pa.array([1, 2], type=pa.int64()),
            "name": pa.array(["a", "b"], type=pa.string()),
        }
    )
    owned = from_shapely_geometries(
        [Point(0, 0), Point(1, 1)],
        residency=Residency.DEVICE,
    )
    gdf = GeoDataFrame(
        {
            "value": [1, 2],
            "name": ["a", "b"],
            "geometry": GeoSeries(DeviceGeometryArray._from_owned(owned), name="geometry"),
        }
    )
    attach_native_state(
        gdf,
        NativeFrameState.from_native_tabular_result(
            NativeTabularResult(
                attributes=NativeAttributeTable(
                    device_table=plc.Table.from_arrow(arrow),
                    index_override=pd.RangeIndex(2),
                    column_override=tuple(arrow.column_names),
                    schema_override=arrow.schema,
                ),
                geometry=GeometryNativeResult.from_owned(owned, crs=None),
                geometry_name="geometry",
                column_order=("value", "name", "geometry"),
            )
        ),
    )
    clear_fallback_events()

    sorted_gdf = gdf.sort_values("value", key=lambda values: -values)
    sort_events = get_fallback_events(clear=True)
    deduped = gdf.drop_duplicates("geometry")
    dedup_events = get_fallback_events(clear=True)

    assert sorted_gdf["value"].tolist() == [2, 1]
    assert deduped["name"].tolist() == ["a", "b"]
    assert [event.surface for event in sort_events] == [
        "geopandas.geodataframe.sort_values"
    ]
    assert [event.surface for event in dedup_events] == [
        "geopandas.geodataframe.drop_duplicates"
    ]


def test_geodataframe_sort_values_duplicate_index_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [2, 1],
            "geometry": GeoSeries.from_wkt(
                ["POINT (2 2)", "POINT (1 1)"],
                name="geometry",
            ),
        },
        index=pd.Index(["same", "same"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    sorted_gdf = gdf.sort_values("value")
    sorted_state = get_native_state(sorted_gdf)

    assert sorted_gdf["value"].tolist() == [1, 2]
    assert sorted_gdf.index.tolist() == ["same", "same"]
    assert sorted_gdf.columns.equals(gdf.columns)
    assert sorted_state is not None
    assert sorted_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        2,
    ]


def test_geodataframe_sort_index_axis_columns_preserves_projected_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    sorted_columns = gdf.sort_index(axis=1)
    sorted_state = get_native_state(sorted_columns)

    assert sorted_columns.columns.tolist() == ["geometry", "name", "value"]
    assert sorted_state is not None
    assert sorted_state.column_order == ("geometry", "name", "value")


def test_geodataframe_sort_values_inplace_clears_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    gdf.sort_values("value", ascending=False, inplace=True)

    assert gdf["value"].tolist() == [2, 1]
    assert get_native_state(gdf) is None


def test_geodataframe_loc_read_preserves_unique_index_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    selected = gdf.loc[[0]]
    selected_state = get_native_state(selected)

    assert selected["value"].tolist() == [1]
    assert selected_state is not None
    assert selected_state.row_count == 1
    assert selected_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [1]


def test_geodataframe_loc_column_projection_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    selected = gdf.loc[[1, 0], ["geometry", "value"]]
    selected_state = get_native_state(selected)

    assert selected.index.tolist() == [1, 0]
    assert selected.columns.tolist() == ["geometry", "value"]
    assert selected_state is not None
    assert selected_state.column_order == ("geometry", "value")
    assert selected_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        2,
        1,
    ]


def test_geodataframe_loc_duplicate_source_index_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        },
        index=pd.Index(["same", "same"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    selected = gdf.loc[["same"]]
    selected_state = get_native_state(selected)

    assert selected["value"].tolist() == [1, 2]
    assert selected_state is not None
    assert selected_state.row_count == 2
    assert selected_state.index_plan.has_duplicates is True
    assert selected_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        2,
    ]


def test_geodataframe_loc_duplicate_label_repeat_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)"],
                name="geometry",
            ),
        },
        index=pd.Index(["same", "same", "other"], name="site"),
    )
    _attach_native_tabular_state(gdf)

    selected = gdf.loc[["same", "other", "same"], ["geometry", "value"]]
    selected_state = get_native_state(selected)

    assert selected.index.tolist() == ["same", "same", "other", "same", "same"]
    assert selected["value"].tolist() == [1, 2, 3, 1, 2]
    assert selected_state is not None
    assert selected_state.column_order == ("geometry", "value")
    assert selected_state.index_plan.has_duplicates is True
    assert selected_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        2,
        3,
        1,
        2,
    ]


def test_geodataframe_loc_multiindex_preserves_private_native_state() -> None:
    gdf = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)"],
                name="geometry",
            ),
        },
        index=pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("a", 3)],
            names=["site", "rank"],
        ),
    )
    _attach_native_tabular_state(gdf)

    selected = gdf.loc[[("a", 1), ("a", 3)], ["value", "geometry"]]
    selected_state = get_native_state(selected)

    assert selected.index.tolist() == [("a", 1), ("a", 3)]
    assert selected["value"].tolist() == [1, 3]
    assert selected_state is not None
    assert selected_state.index_plan.nlevels == 2
    assert selected_state.index_plan.to_public_index().tolist() == [
        ("a", 1),
        ("a", 3),
    ]
    assert selected_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        3,
    ]


def test_geodataframe_query_expression_drops_result_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    queried = gdf.query("value > 1")

    assert queried["value"].tolist() == [2]
    assert get_native_state(gdf) is not None
    assert get_native_state(queried) is None


def test_geodataframe_query_expression_inplace_clears_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    result = gdf.query("value > 1", inplace=True)

    assert result is None
    assert gdf["value"].tolist() == [2]
    assert get_native_state(gdf) is None


@pytest.mark.gpu
def test_geodataframe_query_integer_expression_keeps_device_state() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native query expression")
    cp = pytest.importorskip("cupy")
    gdf = _device_native_expression_geodataframe()

    clear_dispatch_events()
    clear_fallback_events()
    queried = gdf.query("value > 1")
    state = get_native_state(queried)
    events = get_dispatch_events(clear=True)

    assert queried.index.tolist() == [101, 102]
    assert state is not None
    assert state.attributes.is_device_backed
    arrays = state.attributes.numeric_column_arrays(("value",))
    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [2, 3]
    assert get_fallback_events(clear=True) == []
    assert any(
        event.surface == "geopandas.geodataframe.query"
        and event.implementation == "native_device_scalar_expression"
        for event in events
    )


@pytest.mark.gpu
def test_geodataframe_query_and_eval_preserve_public_and_state_attrs() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native expression metadata")
    gdf = _device_native_expression_geodataframe()
    gdf.attrs["source"] = "kept"
    gdf.columns.name = "fields"

    queried = gdf.query("value > 1")
    evaluated = gdf.eval("score = value * 2")

    assert queried.attrs == {"source": "kept"}
    assert evaluated.attrs == {"source": "kept"}
    assert queried.columns.name == "fields"
    assert evaluated.columns.name == "fields"
    assert get_native_state(queried).attrs == queried.attrs
    assert get_native_state(evaluated).attrs == evaluated.attrs


@pytest.mark.gpu
def test_geodataframe_query_composes_over_device_row_view() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native query composition")
    cp = pytest.importorskip("cupy")
    gdf = _device_native_expression_geodataframe([1, 2, 3, 4])

    queried = gdf.query("value > 1").query("value < 4")
    state = get_native_state(queried)

    assert queried.index.tolist() == [101, 102]
    assert state is not None
    arrays = state.attributes.numeric_column_arrays(("value",))
    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [2, 3]


@pytest.mark.gpu
def test_geodataframe_eval_composes_over_sorted_device_row_view() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native eval composition")
    cp = pytest.importorskip("cupy")
    gdf = _device_native_expression_geodataframe([3, 1, 2])

    evaluated = gdf.sort_values("value").eval("score = value * 2")
    state = get_native_state(evaluated)

    assert evaluated.index.tolist() == [101, 102, 100]
    assert state is not None
    arrays = state.attributes.numeric_column_arrays(("value", "score"))
    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [1, 2, 3]
    assert cp.asnumpy(arrays["score"]).tolist() == [2, 4, 6]


@pytest.mark.gpu
def test_geodataframe_query_preserves_exact_int64_scalar_comparison() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native int64 query expression")
    gdf = _device_native_expression_geodataframe([2**53, 2**53 + 1])

    queried = gdf.query(f"value > {2**53}")

    assert queried.index.tolist() == [101]
    assert queried["value"].tolist() == [2**53 + 1]
    assert get_native_state(queried) is not None


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("expr", "expected_index"),
    [
        pytest.param("value > 99", [], id="empty"),
        pytest.param("value >= 1", [100, 101, 102], id="identity"),
    ],
)
def test_geodataframe_query_empty_and_identity_device_rowsets(
    expr: str,
    expected_index: list[int],
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native query rowset edge cases")
    gdf = _device_native_expression_geodataframe()

    queried = gdf.query(expr)

    assert queried.index.tolist() == expected_index
    assert get_native_state(queried) is not None


def test_geodataframe_eval_expression_drops_result_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    evaluated = gdf.eval("score = value + 10")

    assert evaluated["score"].tolist() == [11, 12]
    assert get_native_state(gdf) is not None
    assert get_native_state(evaluated) is None


def test_geodataframe_eval_expression_inplace_clears_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    result = gdf.eval("score = value + 10", inplace=True)

    assert result is None
    assert gdf["score"].tolist() == [11, 12]
    assert get_native_state(gdf) is None


@pytest.mark.gpu
def test_geodataframe_eval_assignment_keeps_device_expression_state() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native eval expression")
    cp = pytest.importorskip("cupy")
    gdf = _device_native_expression_geodataframe()

    clear_dispatch_events()
    clear_fallback_events()
    evaluated = gdf.eval("score = value * 2 + other")
    state = get_native_state(evaluated)
    events = get_dispatch_events(clear=True)

    assert evaluated.index.equals(gdf.index)
    assert state is not None
    assert state.attributes.is_device_backed
    arrays = state.attributes.numeric_column_arrays(("score",))
    assert arrays is not None
    assert cp.asnumpy(arrays["score"]).tolist() == [12, 15, 18]
    assert cp.asarray(arrays["score"]).dtype == cp.dtype("int64")
    assert get_fallback_events(clear=True) == []
    assert any(
        event.surface == "geopandas.geodataframe.eval"
        and event.implementation == "native_device_scalar_expression"
        for event in events
    )


@pytest.mark.gpu
def test_geodataframe_eval_unproven_division_falls_back_observably() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native eval fallback")
    gdf = _device_native_expression_geodataframe()

    clear_fallback_events()
    evaluated = gdf.eval("score = value / 2")
    events = get_fallback_events(clear=True)

    assert evaluated["score"].tolist() == [0.5, 1.0, 1.5]
    assert get_native_state(evaluated) is None
    assert len(events) == 1
    assert events[0].surface == "geopandas.geodataframe.eval"


@pytest.mark.gpu
def test_geodataframe_eval_unproven_division_is_strict_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for strict native eval fallback")
    gdf = _device_native_expression_geodataframe()
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    with pytest.raises(StrictNativeFallbackError, match=r"geodataframe\.eval"):
        gdf.eval("score = value / 2")

    assert get_native_state(gdf) is not None


@pytest.mark.gpu
def test_geodataframe_invalid_query_keeps_pandas_error_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for strict native invalid query")
    from pandas.errors import UndefinedVariableError

    gdf = _device_native_expression_geodataframe()
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_fallback_events()

    with pytest.raises(UndefinedVariableError):
        gdf.query("missing_column > 1")

    assert get_fallback_events(clear=True) == []


@pytest.mark.gpu
def test_geodataframe_malformed_local_query_keeps_pandas_error_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for strict malformed query")
    from pandas.errors import UndefinedVariableError

    gdf = _device_native_expression_geodataframe()
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_fallback_events()

    with pytest.raises(UndefinedVariableError):
        gdf.query("value > @")

    assert get_fallback_events(clear=True) == []


@pytest.mark.gpu
def test_geodataframe_malformed_backtick_query_keeps_pandas_error_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for strict malformed backtick query")
    gdf = _device_native_expression_geodataframe()
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_fallback_events()

    with pytest.raises(SyntaxError):
        gdf.query("value > `other` +")

    assert get_fallback_events(clear=True) == []


@pytest.mark.gpu
@pytest.mark.parametrize(
    "expr",
    [
        pytest.param("`missing name` > 1", id="backtick"),
        pytest.param("value > @missing", id="local"),
    ],
)
def test_geodataframe_undefined_pandas_name_keeps_error_in_strict_mode(
    expr: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for strict pandas name validation")
    from pandas.errors import UndefinedVariableError

    gdf = _device_native_expression_geodataframe()
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_fallback_events()

    with pytest.raises(UndefinedVariableError):
        gdf.query(expr)

    assert get_fallback_events(clear=True) == []


@pytest.mark.gpu
def test_geodataframe_quoted_at_sign_remains_strict_observable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for quoted query fallback")
    gdf = _device_native_expression_geodataframe().rename(
        columns={"value": "value space"}
    )
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    with pytest.raises(
        StrictNativeFallbackError,
        match=r"geodataframe\.query",
    ):
        gdf.query('`value space` > 1 and "x@y.com" == "x@y.com"')


@pytest.mark.gpu
def test_geodataframe_comment_at_sign_remains_strict_observable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for commented query fallback")
    gdf = _device_native_expression_geodataframe().rename(
        columns={"value": "value space"}
    )
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    with pytest.raises(
        StrictNativeFallbackError,
        match=r"geodataframe\.query",
    ):
        gdf.query("`value space` > 1 # @missing")


@pytest.mark.gpu
def test_geodataframe_query_admission_declines_before_expression_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native query admission")
    from vibespatial.api import geodataframe as geodataframe_module
    from vibespatial.cuda._runtime import DeviceMemoryAdmission, get_cuda_runtime

    gdf = _device_native_expression_geodataframe()
    runtime = get_cuda_runtime()
    admissions = []

    def _decline(*, stage, required_bytes, requested_units=0):
        admissions.append((stage, required_bytes, requested_units))
        return DeviceMemoryAdmission(
            stage=stage,
            required_bytes=required_bytes,
            remaining_bytes=0,
            budget_bytes=0,
            admitted=False,
            requested_units=requested_units,
            admitted_units=0,
            bytes_per_unit=required_bytes,
        )

    monkeypatch.setattr(runtime, "admit_device_memory", _decline)
    monkeypatch.setattr(
        geodataframe_module,
        "_native_query_selection_node",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("declined query must not gather expression columns")
        ),
    )
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    with pytest.raises(StrictNativeFallbackError):
        gdf.query("value > 1")

    assert admissions
    assert admissions[0][0] == "tabular-query-expression"
    assert admissions[0][1] > len(gdf) * 112 + (1 << 20)


@pytest.mark.gpu
def test_geodataframe_query_unsupported_device_expression_falls_back_observably() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native query fallback")
    gdf = _device_native_expression_geodataframe()

    clear_fallback_events()
    queried = gdf.query("value > 1.5")
    events = get_fallback_events(clear=True)

    assert queried.index.tolist() == [101, 102]
    assert get_native_state(queried) is None
    assert len(events) == 1
    assert events[0].surface == "geopandas.geodataframe.query"


@pytest.mark.gpu
def test_geodataframe_query_unsupported_device_expression_is_strict_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for strict native query fallback")
    gdf = _device_native_expression_geodataframe()
    source_state = get_native_state(gdf)
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    with pytest.raises(StrictNativeFallbackError, match=r"geodataframe\.query"):
        gdf.query("value > 1.5")

    assert get_native_state(gdf) is source_state


@pytest.mark.gpu
def test_geodataframe_query_supported_function_falls_back_observably() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native query function fallback")
    gdf = _device_native_expression_geodataframe()

    clear_fallback_events()
    queried = gdf.query("abs(value) > 1")
    events = get_fallback_events(clear=True)

    assert queried.index.tolist() == [101, 102]
    assert get_native_state(queried) is None
    assert len(events) == 1
    assert events[0].surface == "geopandas.geodataframe.query"


@pytest.mark.gpu
def test_geodataframe_query_supported_function_is_strict_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for strict native query function fallback")
    gdf = _device_native_expression_geodataframe()
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    with pytest.raises(StrictNativeFallbackError, match=r"geodataframe\.query"):
        gdf.query("abs(value) > 1")


def test_geodataframe_merge_drops_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    merged = gdf.merge(
        pd.DataFrame({"value": [1, 2], "score": [10, 20]}),
        on="value",
    )

    assert merged["score"].tolist() == [10, 20]
    assert get_native_state(gdf) is not None
    assert get_native_state(merged) is None


@pytest.mark.gpu
def test_geodataframe_merge_unique_right_relation_keeps_device_state() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native equality merge")
    cp = pytest.importorskip("cupy")
    gdf = _device_native_expression_geodataframe([2, 1, 2, 3])
    right = pd.DataFrame(
        {
            "value": [2, 1],
            "score": [20, 10],
            "other": [200, 100],
        }
    )

    clear_dispatch_events()
    clear_fallback_events()
    merged = gdf.merge(
        right,
        on="value",
        suffixes=("_left", "_right"),
    )
    state = get_native_state(merged)
    events = get_dispatch_events(clear=True)

    assert merged.columns.tolist() == [
        "value",
        "other_left",
        "geometry",
        "score",
        "other_right",
    ]
    assert merged.index.equals(pd.RangeIndex(3))
    assert state is not None
    assert state.attributes.is_device_backed
    arrays = state.attributes.numeric_column_arrays(
        ("value", "score", "other_right"),
    )
    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [2, 1, 2]
    assert cp.asnumpy(arrays["score"]).tolist() == [20, 10, 20]
    assert cp.asnumpy(arrays["other_right"]).tolist() == [200, 100, 200]
    assert get_fallback_events(clear=True) == []
    assert any(
        event.surface == "geopandas.geodataframe.merge"
        and event.implementation == "pylibcudf_unique_right_inner_join"
        for event in events
    )


@pytest.mark.gpu
def test_geodataframe_query_then_merge_uses_logical_key_domain() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native query/merge composition")
    cp = pytest.importorskip("cupy")
    gdf = _device_native_expression_geodataframe([2, 1, 2, 3])
    right = pd.DataFrame({"value": [2, 3], "score": [20, 30]})

    merged = gdf.query("value > 1").merge(right, on="value")
    state = get_native_state(merged)

    assert state is not None
    arrays = state.attributes.numeric_column_arrays(("value", "score"))
    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [2, 2, 3]
    assert cp.asnumpy(arrays["score"]).tolist() == [20, 20, 30]


@pytest.mark.gpu
def test_geodataframe_query_then_merge_orders_row_view_across_streams() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for row-view stream ordering")
    cp = pytest.importorskip("cupy")
    gdf = _device_native_expression_geodataframe([2, 1, 2, 3])
    right = pd.DataFrame({"value": [2, 3], "score": [20, 30]})
    producer = cp.cuda.Stream(non_blocking=True)
    consumer = cp.cuda.Stream(non_blocking=True)

    with producer:
        queried = gdf.query("value > 1")
    queried_state = get_native_state(queried)
    assert queried_state.attributes.row_positions_readiness is not None

    with consumer:
        merged = queried.merge(right, on="value")
        arrays = get_native_state(merged).attributes.numeric_column_arrays(
            ("value", "score")
        )
    consumer.synchronize()

    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [2, 2, 3]
    assert cp.asnumpy(arrays["score"]).tolist() == [20, 20, 30]


@pytest.mark.gpu
def test_geodataframe_merge_admission_precedes_h2d_and_relation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native merge admission")
    from vibespatial.api import geodataframe as geodataframe_module
    from vibespatial.cuda._runtime import DeviceMemoryAdmission, get_cuda_runtime

    gdf = _device_native_expression_geodataframe()
    state = get_native_state(gdf)
    right = pd.DataFrame({"value": [1, 2, 3], "score": [10, 20, 30]})
    runtime = get_cuda_runtime()
    admissions = []

    def _decline(*, stage, required_bytes, requested_units=0):
        admissions.append((stage, required_bytes, requested_units))
        return DeviceMemoryAdmission(
            stage=stage,
            required_bytes=required_bytes,
            remaining_bytes=0,
            budget_bytes=0,
            admitted=False,
            requested_units=requested_units,
            admitted_units=0,
            bytes_per_unit=required_bytes,
        )

    monkeypatch.setattr(runtime, "admit_device_memory", _decline)
    monkeypatch.setattr(
        geodataframe_module,
        "_native_device_carrier_bytes",
        lambda _value: 4096,
    )
    monkeypatch.setattr(
        geodataframe_module,
        "_native_device_table_from_public_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("declined merge must not upload the right frame")
        ),
    )
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    with pytest.raises(StrictNativeFallbackError, match=r"geodataframe\.merge"):
        gdf.merge(right, on="value")

    left_bytes = geodataframe_module._native_device_column_storage_bytes(
        state.attributes,
        tuple(state.attributes.columns),
    )
    right_bytes = int(right.memory_usage(index=False, deep=True).sum())
    expected = (
        5 * max(left_bytes + right_bytes, len(gdf))
        + 2 * 4096
        + len(gdf) * 32
        + (1 << 20)
    )
    assert admissions == [
        ("tabular-unique-right-inner-merge", expected, len(gdf))
    ]


@pytest.mark.gpu
@pytest.mark.parametrize("operation", ["merge", "join"])
@pytest.mark.parametrize("metadata_equal", [True, False])
def test_geodataframe_native_join_metadata_matches_pandas(
    operation: str,
    metadata_equal: bool,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native join metadata")
    gdf = _device_native_expression_geodataframe()
    gdf.attrs = {"source": "shared"}
    gdf.columns.name = "fields"
    if operation == "merge":
        right = pd.DataFrame(
            {"value": [1, 2, 3], "score": [10, 20, 30]},
        )
    else:
        right = pd.DataFrame(
            {"score": [10, 20, 30]},
            index=gdf.index,
        )
    right.attrs = {"source": "shared" if metadata_equal else "other"}
    right.columns.name = "fields" if metadata_equal else "right_fields"

    result = gdf.merge(right, on="value") if operation == "merge" else gdf.join(right)
    state = get_native_state(result)

    expected_attrs = {"source": "shared"} if metadata_equal else {}
    assert result.attrs == expected_attrs
    assert result.columns.name == ("fields" if metadata_equal else None)
    assert state is not None
    assert state.attrs == expected_attrs


@pytest.mark.gpu
@pytest.mark.parametrize(
    "case",
    [
        "merge_missing_key",
        "merge_invalid_validate",
        "merge_empty_suffixes",
        "join_invalid_validate",
        "join_empty_suffixes",
    ],
)
def test_geodataframe_invalid_join_input_keeps_pandas_error_in_strict_mode(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for strict native join validation")
    gdf = _device_native_expression_geodataframe()
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_fallback_events()

    expected_error = KeyError if case == "merge_missing_key" else ValueError
    with pytest.raises(expected_error):
        if case == "merge_missing_key":
            gdf.merge(pd.DataFrame({"value": [1]}), on="missing")
        elif case == "merge_invalid_validate":
            gdf.merge(
                pd.DataFrame({"value": [1, 2, 3]}),
                on="value",
                validate="bogus",
            )
        elif case == "merge_empty_suffixes":
            gdf.merge(
                pd.DataFrame({"value": [1, 2, 3], "other": [4, 5, 6]}),
                on="value",
                suffixes=("", ""),
            )
        elif case == "join_invalid_validate":
            gdf.join(
                pd.DataFrame({"score": [1, 2, 3]}, index=gdf.index),
                validate="bogus",
            )
        else:
            gdf.join(pd.DataFrame({"other": [1, 2, 3]}, index=gdf.index))

    assert get_fallback_events(clear=True) == []


@pytest.mark.gpu
def test_geodataframe_array_key_merge_remains_observable_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for array-key merge fallback")
    gdf = _device_native_expression_geodataframe()
    right = pd.DataFrame({"score": [10, 20, 30]})
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    with pytest.raises(StrictNativeFallbackError, match=r"geodataframe\.merge"):
        gdf.merge(
            right,
            left_on=np.asarray([1, 2, 3]),
            right_on=np.asarray([1, 2, 3]),
        )


@pytest.mark.gpu
def test_geodataframe_ambiguous_merge_key_keeps_pandas_error_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for ambiguous merge key validation")
    gdf = _device_native_expression_geodataframe().rename_axis("value")
    right = pd.DataFrame({"value": [1, 2, 3], "score": [10, 20, 30]})
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_fallback_events()

    with pytest.raises(ValueError, match="both an index level and a column label"):
        gdf.merge(right, on="value")

    assert get_fallback_events(clear=True) == []


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("operation", "invalid_kind"),
    [
        pytest.param("merge", "missing", id="merge-missing"),
        pytest.param("merge", "ambiguous", id="merge-ambiguous"),
        pytest.param("join", "missing", id="join-missing"),
        pytest.param("join", "ambiguous", id="join-ambiguous"),
    ],
)
def test_geodataframe_list_key_errors_survive_strict_mode(
    operation: str,
    invalid_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for list-form key validation")
    gdf = _device_native_expression_geodataframe()
    if invalid_kind == "ambiguous":
        gdf = gdf.rename_axis("value")
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_fallback_events()

    expected_error = KeyError if invalid_kind == "missing" else ValueError
    key = "missing" if invalid_kind == "missing" else "value"
    with pytest.raises(expected_error):
        if operation == "merge":
            gdf.merge(
                pd.DataFrame({"value": [1, 2, 3], "score": [10, 20, 30]}),
                left_on=[key],
                right_on=["value"],
            )
        else:
            gdf.join(
                pd.DataFrame({"score": [10, 20, 30]}, index=gdf.index),
                on=[key],
            )

    assert get_fallback_events(clear=True) == []


@pytest.mark.gpu
@pytest.mark.parametrize("operation", ["merge", "join"])
def test_geodataframe_key_arity_error_survives_strict_mode(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for key arity validation")
    gdf = _device_native_expression_geodataframe()
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_fallback_events()

    with pytest.raises(ValueError):
        if operation == "merge":
            gdf.merge(
                pd.DataFrame(
                    {
                        "value": [1, 2, 3],
                        "other": [10, 11, 12],
                    }
                ),
                left_on=["value", "other"],
                right_on=["value"],
            )
        else:
            gdf.join(
                pd.DataFrame({"score": [10, 20, 30]}, index=gdf.index),
                on=["value", "other"],
            )

    assert get_fallback_events(clear=True) == []


@pytest.mark.gpu
@pytest.mark.parametrize(
    "on",
    [
        pytest.param(np.asarray([1, 2, 3]), id="ndarray"),
        pytest.param(pd.Series([1, 2, 3]), id="series"),
        pytest.param(pd.Index([1, 2, 3]), id="index"),
        pytest.param(pd.array([1, 2, 3], dtype="Int64"), id="extension-array"),
    ],
)
def test_geodataframe_array_join_key_remains_observable_in_strict_mode(
    on,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for array-valued join fallback")
    gdf = _device_native_expression_geodataframe()
    right = pd.DataFrame(
        {"score": [10, 20, 30]},
        index=pd.Index([1, 2, 3]),
    )
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    with pytest.raises(
        StrictNativeFallbackError,
        match=r"geodataframe\.join",
    ):
        gdf.join(right, on=on)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "on",
    [
        pytest.param([np.asarray([1, 2, 3])], id="list-ndarray"),
        pytest.param((pd.Series([1, 2, 3]),), id="tuple-series"),
        pytest.param([pd.array([1, 2, 3], dtype="Int64")], id="list-extension"),
    ],
)
def test_geodataframe_nested_array_join_key_remains_strict_observable(
    on,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for nested array join fallback")
    gdf = _device_native_expression_geodataframe()
    right = pd.DataFrame(
        {"score": [10, 20, 30]},
        index=pd.Index([1, 2, 3]),
    )
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    with pytest.raises(
        StrictNativeFallbackError,
        match=r"geodataframe\.join",
    ):
        gdf.join(right, on=on)


@pytest.mark.gpu
@pytest.mark.parametrize("operation", ["merge", "join"])
def test_geodataframe_suffix_collision_keeps_pandas_error_in_strict_mode(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for suffix collision validation")
    from pandas.errors import MergeError

    gdf = _device_native_expression_geodataframe().eval(
        "other_left = value + 1"
    )
    if operation == "merge":
        right = pd.DataFrame(
            {"value": [1, 2, 3], "other": [20, 21, 22]},
        )
    else:
        right = pd.DataFrame(
            {"other": [20, 21, 22]},
            index=gdf.index,
        )
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_fallback_events()

    with pytest.raises(MergeError):
        if operation == "merge":
            gdf.merge(right, on="value", suffixes=("_left", "_right"))
        else:
            gdf.join(right, lsuffix="_left", rsuffix="_right")

    assert get_fallback_events(clear=True) == []


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("values", "validate"),
    [
        pytest.param([1, 2, 3], "m:1", id="duplicate-right"),
        pytest.param([1, 1, 2], "1:1", id="duplicate-left"),
    ],
)
def test_geodataframe_merge_validate_failure_keeps_pandas_error_in_strict_mode(
    values: list[int],
    validate: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for merge cardinality validation")
    from pandas.errors import MergeError

    gdf = _device_native_expression_geodataframe(values)
    right_values = [1, 2, 2] if validate == "m:1" else [1, 2, 3]
    right = pd.DataFrame(
        {"value": right_values, "score": [10, 20, 30]},
    )
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_fallback_events()

    with pytest.raises(MergeError):
        gdf.merge(right, on="value", validate=validate)

    assert get_fallback_events(clear=True) == []


@pytest.mark.gpu
def test_geodataframe_join_validate_failure_keeps_pandas_error_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for join cardinality validation")
    from pandas.errors import MergeError

    gdf = _device_native_expression_geodataframe().set_axis(
        pd.Index([1, 1, 2]),
        axis=0,
    )
    right = pd.DataFrame({"score": [10, 20]}, index=pd.Index([1, 2]))
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    clear_fallback_events()

    with pytest.raises(MergeError):
        gdf.join(right, validate="1:1")

    assert get_fallback_events(clear=True) == []


@pytest.mark.gpu
def test_geodataframe_join_multiindex_columns_is_strict_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for MultiIndex join fallback")
    gdf = _device_native_expression_geodataframe()
    right = pd.DataFrame(
        [[10, 20, 30]],
        index=pd.MultiIndex.from_tuples([("metrics", "score")]),
        columns=gdf.index,
    ).T
    assert isinstance(right.columns, pd.MultiIndex)
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    with pytest.raises(StrictNativeFallbackError, match=r"geodataframe\.join"):
        gdf.join(right)


@pytest.mark.gpu
def test_geodataframe_merge_nullable_string_keys_match_pandas_order() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native nullable-string merge")
    plc = pytest.importorskip("pylibcudf")
    keys = ["b", None, "a", "b"]
    owned = from_shapely_geometries(
        [Point(position, position) for position in range(4)],
        residency=Residency.DEVICE,
    )
    arrow = pa.table(
        {
            "key": pa.array(keys, type=pa.string()),
            "value": pa.array([0, 1, 2, 3], type=pa.int64()),
        }
    )
    gdf = GeoDataFrame(
        {
            "key": keys,
            "value": [0, 1, 2, 3],
            "geometry": GeoSeries(
                DeviceGeometryArray._from_owned(owned),
                name="geometry",
            ),
        }
    )
    attach_native_state(
        gdf,
        NativeFrameState.from_native_tabular_result(
            NativeTabularResult(
                attributes=NativeAttributeTable(
                    device_table=plc.Table.from_arrow(arrow),
                    index_override=gdf.index,
                    column_override=tuple(arrow.column_names),
                    schema_override=arrow.schema,
                ),
                geometry=GeometryNativeResult.from_owned(owned, crs=None),
                geometry_name="geometry",
                column_order=("key", "value", "geometry"),
            )
        ),
    )
    right = pd.DataFrame(
        {
            "key": ["a", None, "b"],
            "label": ["A", "missing", "B"],
        }
    )

    merged = gdf.merge(right, on="key")

    merged_keys = merged["key"].tolist()
    assert merged_keys[0] == "b"
    assert pd.isna(merged_keys[1])
    assert merged_keys[2:] == ["a", "b"]
    assert merged["value"].tolist() == [0, 1, 2, 3]
    assert merged["label"].tolist() == ["B", "missing", "A", "B"]
    assert get_native_state(merged) is not None
    assert get_native_state(merged).attributes.is_device_backed


@pytest.mark.gpu
def test_geodataframe_merge_duplicate_right_falls_back_observably() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native equality merge fallback")
    gdf = _device_native_expression_geodataframe()
    right = pd.DataFrame({"value": [2, 2], "score": [20, 21]})

    clear_fallback_events()
    merged = gdf.merge(right, on="value")
    events = get_fallback_events(clear=True)

    assert merged["score"].tolist() == [20, 21]
    assert get_native_state(merged) is None
    assert len(events) == 1
    assert events[0].surface == "geopandas.geodataframe.merge"


@pytest.mark.gpu
def test_geodataframe_merge_duplicate_right_is_strict_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for strict native equality merge")
    gdf = _device_native_expression_geodataframe()
    right = pd.DataFrame({"value": [2, 2], "score": [20, 21]})
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    with pytest.raises(StrictNativeFallbackError, match=r"geodataframe\.merge"):
        gdf.merge(right, on="value")

    assert get_native_state(gdf) is not None


def test_geodataframe_join_drops_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    joined = gdf.join(pd.DataFrame({"score": [10, 20]}, index=gdf.index))

    assert joined["score"].tolist() == [10, 20]
    assert get_native_state(gdf) is not None
    assert get_native_state(joined) is None


@pytest.mark.gpu
def test_geodataframe_join_exact_unique_index_keeps_device_state() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native exact-index join")
    cp = pytest.importorskip("cupy")
    gdf = _device_native_expression_geodataframe()
    right = pd.DataFrame(
        {"score": [30, 20, 10]},
        index=gdf.index,
    )

    clear_dispatch_events()
    clear_fallback_events()
    joined = gdf.join(right)
    state = get_native_state(joined)
    events = get_dispatch_events(clear=True)

    assert joined.columns.tolist() == ["value", "other", "geometry", "score"]
    assert joined.index.equals(gdf.index)
    assert state is not None
    assert state.attributes.is_device_backed
    arrays = state.attributes.numeric_column_arrays(("score",))
    assert arrays is not None
    assert cp.asnumpy(arrays["score"]).tolist() == [30, 20, 10]
    assert get_fallback_events(clear=True) == []
    assert any(
        event.surface == "geopandas.geodataframe.join"
        and event.implementation == "native_exact_index_column_join"
        for event in events
    )


@pytest.mark.gpu
def test_geodataframe_join_device_index_declines_before_public_index_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native device-index join decline")
    gdf = _device_native_expression_geodataframe().query("value > 1")
    assert get_native_state(gdf).index_plan.kind == "host-labels-take"
    right = pd.DataFrame(
        {"score": [20, 30]},
        index=pd.Index([101, 102], name="source_id"),
    )
    monkeypatch.setattr(
        NativeIndexPlan,
        "to_public_index",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native join admission must not export device labels")
        ),
    )
    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

    with pytest.raises(StrictNativeFallbackError, match=r"geodataframe\.join"):
        gdf.join(right)


@pytest.mark.gpu
def test_geodataframe_merge_secondary_geometry_falls_back_observably() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for secondary geometry merge fallback")
    gdf = _two_geometry_geodataframe()
    _attach_two_geometry_native_tabular_state(
        gdf,
        owned=True,
        attribute_storage="device",
    )
    right = pd.DataFrame({"value": [1, 2], "score": [10, 20]})

    clear_fallback_events()
    merged = gdf.merge(right, on="value")
    events = get_fallback_events(clear=True)

    assert "other_geometry" in merged.columns
    assert get_native_state(merged) is None
    assert len(events) == 1
    assert events[0].surface == "geopandas.geodataframe.merge"


@pytest.mark.gpu
def test_geodataframe_merge_right_geometry_dtype_falls_back_observably() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for right geometry merge fallback")
    gdf = _device_native_expression_geodataframe()
    right = pd.DataFrame(
        {
            "value": [1, 2, 3],
            "right_geometry": GeoSeries.from_wkt(
                ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)"],
            ),
        }
    )

    clear_fallback_events()
    merged = gdf.merge(right, on="value")
    events = get_fallback_events(clear=True)

    assert "right_geometry" in merged.columns
    assert get_native_state(merged) is None
    assert len(events) == 1
    assert events[0].surface == "geopandas.geodataframe.merge"


def test_geodataframe_concat_preserves_exact_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    concatenated = pd.concat([gdf, gdf.iloc[:1]], ignore_index=True)
    state = get_native_state(concatenated)

    assert concatenated["value"].tolist() == [1, 2, 1]
    assert get_native_state(gdf) is not None
    assert state is not None
    assert state.row_count == 3
    assert state.column_order == ("value", "name", "geometry")
    assert state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        2,
        1,
    ]


def test_geodataframe_concat_duplicate_index_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    concatenated = pd.concat([gdf, gdf.iloc[:1]], ignore_index=False)
    state = get_native_state(concatenated)

    assert concatenated.index.tolist() == [0, 1, 0]
    assert get_native_state(gdf) is not None
    assert state is not None
    assert state.index_plan.has_duplicates is True
    assert state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        2,
        1,
    ]


def test_geodataframe_concat_preserves_unique_index_private_native_state() -> None:
    left = GeoDataFrame(
        {
            "value": [1, 2],
            "name": ["a", "b"],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        },
        index=pd.Index(["a", "b"], name="row"),
    )
    right = GeoDataFrame(
        {
            "value": [3],
            "name": ["c"],
            "geometry": GeoSeries.from_wkt(["POINT (2 2)"], name="geometry"),
        },
        index=pd.Index(["c"], name="row"),
    )
    _attach_native_tabular_state(left)
    _attach_native_tabular_state(right)

    concatenated = pd.concat([left, right], ignore_index=False)
    state = get_native_state(concatenated)

    assert concatenated.index.tolist() == ["a", "b", "c"]
    assert concatenated.index.name == "row"
    assert get_native_state(left) is not None
    assert get_native_state(right) is not None
    assert state is not None
    assert state.index_plan.kind == "host-labels"
    assert state.index_plan.to_public_index().tolist() == ["a", "b", "c"]
    assert state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        2,
        3,
    ]


def test_geodataframe_concat_multiindex_preserves_private_native_state() -> None:
    left = GeoDataFrame(
        {
            "value": [1, 2],
            "name": ["a", "b"],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        },
        index=pd.MultiIndex.from_tuples(
            [("left", 0), ("left", 1)],
            names=["side", "row"],
        ),
    )
    right = GeoDataFrame(
        {
            "value": [3],
            "name": ["c"],
            "geometry": GeoSeries.from_wkt(["POINT (2 2)"], name="geometry"),
        },
        index=pd.MultiIndex.from_tuples(
            [("right", 0)],
            names=["side", "row"],
        ),
    )
    _attach_native_tabular_state(left)
    _attach_native_tabular_state(right)

    concatenated = pd.concat([left, right], ignore_index=False)
    state = get_native_state(concatenated)

    assert concatenated.index.tolist() == [("left", 0), ("left", 1), ("right", 0)]
    assert state is not None
    assert state.index_plan.nlevels == 2
    assert state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        2,
        3,
    ]


def test_geodataframe_concat_preserves_device_private_native_state() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for native concat preservation")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    import pylibcudf as plc

    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    def build_frame(scores, points):
        owned = from_shapely_geometries(points, residency=Residency.DEVICE)
        arrow = pa.table({"score": pa.array(scores, type=pa.float64())})
        gdf = GeoDataFrame(
            {
                "score": list(scores),
                "geometry": GeoSeries(
                    GeometryArray.from_owned(owned, crs=None),
                    name="geometry",
                ),
            }
        )
        state = NativeFrameState.from_native_tabular_result(
            NativeTabularResult(
                attributes=NativeAttributeTable(
                    device_table=plc.Table.from_arrow(arrow),
                    index_override=pd.RangeIndex(len(scores)),
                    column_override=tuple(arrow.column_names),
                    schema_override=arrow.schema,
                ),
                geometry=GeometryNativeResult.from_owned(owned, crs=None),
                geometry_name="geometry",
                column_order=("score", "geometry"),
            )
        )
        attach_native_state(gdf, state)
        return gdf

    left = build_frame([1.5, 2.5], [Point(0, 0), Point(1, 1)])
    right = build_frame([3.5], [Point(2, 2)])
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        concatenated = pd.concat([left, right], ignore_index=True)
        state = get_native_state(concatenated)
        arrays = state.attributes.numeric_column_arrays(("score",))

    assert concatenated["score"].tolist() == [1.5, 2.5, 3.5]
    assert state is not None
    assert state.attributes.device_table is not None
    assert state.geometry.composition is not None
    assert state.geometry.composition.residency is Residency.DEVICE
    assert state.geometry.composition.trusted_singular_rows is True
    assert arrays is not None
    assert cp.asnumpy(arrays["score"]).tolist() == [1.5, 2.5, 3.5]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


@pytest.mark.parametrize(
    ("operation", "frame_factory", "mutator"),
    [
        (
            "apply",
            _native_backed_geodataframe_only,
            lambda frame: frame.apply(lambda column: column),
        ),
    ],
)
def test_geodataframe_broad_pandas_results_drop_private_native_state(
    operation,
    frame_factory,
    mutator,
) -> None:
    frame = frame_factory()

    result = mutator(frame)

    assert operation
    assert get_native_state(frame) is not None
    assert get_native_state(result) is None


def test_geodataframe_dropna_rows_preserves_private_native_state() -> None:
    frame = _native_backed_nullable_geodataframe()

    result = frame.dropna()
    state = get_native_state(result)

    assert result.index.tolist() == [0]
    assert result["value"].tolist() == [1.0]
    assert state is not None
    assert state.row_count == 1
    assert state.column_order == ("value", "name", "geometry")
    assert state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [1.0]


def test_geodataframe_dropna_ignore_index_preserves_private_native_state() -> None:
    index = pd.Index(["z", "y", "x"], name="site")
    frame = GeoDataFrame(
        {
            "value": [1.0, None, 3.0],
            "name": ["a", "b", "c"],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)"],
                name="geometry",
                index=index,
            ),
        },
        index=index,
    )
    _attach_native_tabular_state(frame)

    result = frame.dropna(ignore_index=True)
    state = get_native_state(result)

    assert result.index.equals(pd.RangeIndex(2))
    assert result["value"].tolist() == [1.0, 3.0]
    assert state is not None
    assert state.index_plan.kind == "range"
    assert state.column_order == ("value", "name", "geometry")
    assert state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1.0,
        3.0,
    ]


def test_geodataframe_dropna_columns_preserves_projected_private_native_state() -> None:
    frame = _native_backed_nullable_geodataframe()

    result = frame.dropna(axis="columns")
    state = get_native_state(result)

    assert tuple(result.columns) == ("name", "geometry")
    assert state is not None
    assert state.row_count == 2
    assert state.column_order == ("name", "geometry")
    assert state.to_native_tabular_result().attributes.to_pandas()["name"].tolist() == [
        "a",
        "b",
    ]


def test_geodataframe_fillna_attribute_only_preserves_private_native_state() -> None:
    frame = _native_backed_nullable_geodataframe()

    result = frame.fillna({"value": 2.0})
    state = get_native_state(result)

    assert result["value"].tolist() == [1.0, 2.0]
    assert state is not None
    assert state.row_count == 2
    assert state.column_order == ("value", "name", "geometry")
    assert state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1.0,
        2.0,
    ]


def test_geodataframe_fillna_owned_geometry_preserves_without_geometry_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = from_shapely_geometries([Point(0, 0), Point(1, 1)])
    frame = GeoDataFrame(
        {
            "value": [1.0, None],
            "geometry": GeoSeries(GeometryArray.from_owned(owned), name="geometry"),
        },
    )
    _attach_native_tabular_state(frame)

    def fail_owned_to_shapely(*args, **kwargs):
        raise AssertionError("owned geometry should not materialize")

    monkeypatch.setattr(
        "vibespatial.api.geometry_array.owned_to_shapely",
        fail_owned_to_shapely,
    )

    result = frame.fillna({"value": 2.0})
    state = get_native_state(result)

    assert state is not None
    assert state.geometry.owned is owned
    assert getattr(result.geometry.values, "_owned", None) is owned
    assert state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1.0,
        2.0,
    ]


def test_geodataframe_replace_attribute_only_preserves_private_native_state() -> None:
    frame = _native_backed_geodataframe_only()

    result = frame.replace({"value": {1: 9}})
    state = get_native_state(result)

    assert result["value"].tolist() == [9, 2]
    assert state is not None
    assert state.row_count == 2
    assert state.column_order == ("value", "name", "geometry")
    assert state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        9,
        2,
    ]


def test_geodataframe_replace_geometry_change_drops_private_native_state() -> None:
    frame = _native_backed_geodataframe_only()

    result = frame.replace({"geometry": {frame.geometry.iloc[0]: frame.geometry.iloc[1]}})

    assert result.geometry.iloc[0].equals(frame.geometry.iloc[1])
    assert get_native_state(result) is None


def test_geodataframe_astype_attribute_only_preserves_private_native_state() -> None:
    frame = _native_backed_geodataframe_only()

    result = frame.astype({"value": "float64"})
    state = get_native_state(result)

    assert result["value"].tolist() == [1.0, 2.0]
    assert str(result["value"].dtype) == "float64"
    assert state is not None
    assert state.row_count == 2
    assert state.column_order == ("value", "name", "geometry")
    assert state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1.0,
        2.0,
    ]


@pytest.mark.parametrize(
    ("operation", "mutator"),
    [
        (
            "where",
            lambda frame: frame.where(
                pd.DataFrame(
                    {
                        "value": [False, True],
                        "name": [True, True],
                        "geometry": [True, True],
                    },
                ),
                GeoDataFrame(
                    {
                        "value": [9, 9],
                        "name": ["x", "x"],
                        "geometry": GeoSeries.from_wkt(
                            ["POINT (9 9)", "POINT (9 9)"],
                            name="geometry",
                        ),
                    },
                ),
            ),
        ),
        (
            "mask",
            lambda frame: frame.mask(
                pd.DataFrame(
                    {
                        "value": [True, False],
                        "name": [False, False],
                        "geometry": [False, False],
                    },
                ),
                GeoDataFrame(
                    {
                        "value": [9, 9],
                        "name": ["x", "x"],
                        "geometry": GeoSeries.from_wkt(
                            ["POINT (9 9)", "POINT (9 9)"],
                            name="geometry",
                        ),
                    },
                ),
            ),
        ),
    ],
)
def test_geodataframe_where_mask_attribute_only_preserve_private_native_state(
    operation,
    mutator,
) -> None:
    frame = _native_backed_geodataframe_only()

    result = mutator(frame)
    state = get_native_state(result)

    assert operation
    assert result["value"].tolist() == [9, 2]
    assert state is not None
    assert state.row_count == 2
    assert state.column_order == ("value", "name", "geometry")
    assert state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        9,
        2,
    ]


@pytest.mark.parametrize(
    ("operation", "mutator"),
    [
        (
            "where",
            lambda frame: frame.where(
                pd.DataFrame(
                    {
                        "value": [True, True],
                        "name": [True, True],
                        "geometry": [False, True],
                    },
                ),
                GeoDataFrame(
                    {
                        "value": [9, 9],
                        "name": ["x", "x"],
                        "geometry": GeoSeries.from_wkt(
                            ["POINT (9 9)", "POINT (9 9)"],
                            name="geometry",
                        ),
                    },
                ),
            ),
        ),
        (
            "mask",
            lambda frame: frame.mask(
                pd.DataFrame(
                    {
                        "value": [False, False],
                        "name": [False, False],
                        "geometry": [True, False],
                    },
                ),
                GeoDataFrame(
                    {
                        "value": [9, 9],
                        "name": ["x", "x"],
                        "geometry": GeoSeries.from_wkt(
                            ["POINT (9 9)", "POINT (9 9)"],
                            name="geometry",
                        ),
                    },
                ),
            ),
        ),
    ],
)
def test_geodataframe_where_mask_geometry_change_drop_private_native_state(
    operation,
    mutator,
) -> None:
    frame = _native_backed_geodataframe_only()

    result = mutator(frame)

    assert operation
    assert result.geometry.iloc[0].equals(Point(9, 9))
    assert get_native_state(result) is None


@pytest.mark.parametrize(
    ("operation", "frame_factory", "mutator"),
    [
        (
            "where",
            _native_backed_geodataframe_only,
            lambda frame: frame.where(
                pd.DataFrame(
                    {
                        "value": [False, True],
                        "name": [True, True],
                        "geometry": [True, True],
                    },
                ),
                GeoDataFrame(
                    {
                        "value": [9, 9],
                        "name": ["x", "x"],
                        "geometry": GeoSeries.from_wkt(
                            ["POINT (9 9)", "POINT (9 9)"],
                            name="geometry",
                        ),
                    },
                ),
                inplace=True,
            ),
        ),
        (
            "mask",
            _native_backed_geodataframe_only,
            lambda frame: frame.mask(
                pd.DataFrame(
                    {
                        "value": [True, False],
                        "name": [False, False],
                        "geometry": [False, False],
                    },
                ),
                GeoDataFrame(
                    {
                        "value": [9, 9],
                        "name": ["x", "x"],
                        "geometry": GeoSeries.from_wkt(
                            ["POINT (9 9)", "POINT (9 9)"],
                            name="geometry",
                        ),
                    },
                ),
                inplace=True,
            ),
        ),
        (
            "replace",
            _native_backed_geodataframe_only,
            lambda frame: frame.replace({"value": {2: 9}}, inplace=True),
        ),
        (
            "rename_axis",
            _native_backed_geodataframe_only,
            lambda frame: frame.rename_axis("site", inplace=True),
        ),
        (
            "dropna",
            _native_backed_nullable_geodataframe,
            lambda frame: frame.dropna(inplace=True),
        ),
        (
            "fillna",
            _native_backed_nullable_geodataframe,
            lambda frame: frame.fillna({"value": 2.0}, inplace=True),
        ),
        (
            "update",
            _native_backed_geodataframe_only,
            lambda frame: frame.update(pd.DataFrame({"value": [9, 10]})),
        ),
    ],
)
def test_geodataframe_broad_pandas_inplace_mutators_clear_private_native_state(
    operation,
    frame_factory,
    mutator,
) -> None:
    frame = frame_factory()

    mutator(frame)

    assert operation
    assert get_native_state(frame) is None


def test_geodataframe_setitem_new_attribute_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    gdf["new_value"] = [3, 4]

    setitem_state = get_native_state(gdf)

    assert setitem_state is not None
    assert setitem_state.column_order == ("value", "name", "geometry", "new_value")
    assert setitem_state.to_native_tabular_result().attributes.to_pandas()[
        "new_value"
    ].tolist() == [3, 4]


def test_geodataframe_setitem_replace_attribute_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    gdf["value"] = [10, 20]
    setitem_state = get_native_state(gdf)

    assert gdf["value"].tolist() == [10, 20]
    assert setitem_state is not None
    assert setitem_state.column_order == ("value", "name", "geometry")
    assert setitem_state.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        10,
        20,
    ]


def test_geodataframe_setitem_geometry_clears_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    gdf["geometry"] = GeoSeries.from_wkt(
        ["POINT (2 2)", "POINT (3 3)"],
        name="geometry",
    )

    assert gdf.geometry.to_wkt().tolist() == ["POINT (2 2)", "POINT (3 3)"]
    assert get_native_state(gdf) is None


def test_geodataframe_setitem_native_geometry_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()
    owned = from_shapely_geometries([Point(2, 2), Point(3, 3)])

    gdf["geometry"] = GeoSeries(GeometryArray.from_owned(owned), name="geometry")
    setitem_state = get_native_state(gdf)

    assert gdf.geometry.to_wkt().tolist() == ["POINT (2 2)", "POINT (3 3)"]
    assert setitem_state is not None
    assert setitem_state.geometry.owned is owned
    assert setitem_state.column_order == ("value", "name", "geometry")


def test_geodataframe_setitem_multiple_attribute_columns_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    gdf[["value", "name"]] = [[10, "x"], [20, "y"]]
    setitem_state = get_native_state(gdf)

    assert gdf["value"].tolist() == [10, 20]
    assert gdf["name"].tolist() == ["x", "y"]
    assert setitem_state is not None
    assert setitem_state.column_order == ("value", "name", "geometry")
    assert setitem_state.to_native_tabular_result().attributes.to_pandas()["name"].tolist() == [
        "x",
        "y",
    ]


def test_geodataframe_setitem_multiple_columns_with_geometry_clears_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    gdf[["value", "geometry"]] = [
        [10, Point(10, 10)],
        [20, Point(20, 20)],
    ]

    assert gdf["value"].tolist() == [10, 20]
    assert get_native_state(gdf) is None


def test_geodataframe_setitem_multiple_device_attribute_columns_preserves_without_runtime_d2h() -> (
    None
):
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device attribute assignment")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    gdf = GeoDataFrame(
        {
            "value": [1, 2],
            "flag": [True, False],
            "geometry": GeoSeries.from_wkt(
                ["POINT (1 1)", "POINT (2 2)"],
                name="geometry",
            ),
        }
    )
    _attach_owned_native_tabular_state(gdf, attribute_storage="device")
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        gdf[["value", "score"]] = [[10, 30], [20, 40]]
        state = get_native_state(gdf)
        assert state is not None
        arrays = state.attributes.numeric_column_arrays(("value", "flag", "score"))

    assert state.attributes.device_table is not None
    assert state.column_order == ("value", "flag", "geometry", "score")
    assert arrays is not None
    assert cp.asnumpy(arrays["value"]).tolist() == [10, 20]
    assert cp.asnumpy(arrays["flag"]).tolist() == [True, False]
    assert cp.asnumpy(arrays["score"]).tolist() == [30, 40]
    assert get_materialization_events(clear=True) == []
    reset_d2h_transfer_count()


def test_geodataframe_setitem_partitions_device_attribute_state() -> None:
    gdf, _state = _native_backed_geodataframe()
    device_attributes = NativeAttributeTable(
        device_table=_FakeDeviceTable(),
        index_override=gdf.index,
        column_override=("value", "name"),
        schema_override=pa.schema(
            [
                pa.field("value", pa.int64()),
                pa.field("name", pa.string()),
            ]
        ),
    )
    result = NativeTabularResult(
        attributes=device_attributes,
        geometry=GeometryNativeResult.from_geoseries(gdf.geometry),
        geometry_name="geometry",
        column_order=("value", "name", "geometry"),
    )
    attach_native_state(gdf, NativeFrameState.from_native_tabular_result(result))

    assigned_values = [object(), object()]
    gdf["score"] = assigned_values

    assigned_state = get_native_state(gdf)
    assert assigned_state is not None
    assert assigned_state.attributes.parts is not None
    assert gdf["score"].tolist() == assigned_values


def test_geodataframe_insert_attribute_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    result = gdf.insert(1, "score", [10, 20])
    insert_state = get_native_state(gdf)

    assert result is None
    assert gdf.columns.tolist() == ["value", "score", "name", "geometry"]
    assert insert_state is not None
    assert insert_state.column_order == ("value", "score", "name", "geometry")
    assert insert_state.to_native_tabular_result().attributes.to_pandas()["score"].tolist() == [
        10,
        20,
    ]


def test_geodataframe_insert_duplicate_column_declines_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    gdf.insert(1, "value", [10, 20], allow_duplicates=True)

    assert gdf.columns.tolist() == ["value", "value", "name", "geometry"]
    assert get_native_state(gdf) is None


def test_geodataframe_insert_partitions_device_attribute_state() -> None:
    gdf, _state = _native_backed_geodataframe()
    device_attributes = NativeAttributeTable(
        device_table=_FakeDeviceTable(),
        index_override=gdf.index,
        column_override=("value", "name"),
        schema_override=pa.schema(
            [
                pa.field("value", pa.int64()),
                pa.field("name", pa.string()),
            ]
        ),
    )
    result = NativeTabularResult(
        attributes=device_attributes,
        geometry=GeometryNativeResult.from_geoseries(gdf.geometry),
        geometry_name="geometry",
        column_order=("value", "name", "geometry"),
    )
    attach_native_state(gdf, NativeFrameState.from_native_tabular_result(result))

    assigned_values = [object(), object()]
    gdf.insert(1, "score", assigned_values)

    assigned_state = get_native_state(gdf)
    assert assigned_state is not None
    assert assigned_state.attributes.parts is not None
    assert gdf["score"].tolist() == assigned_values


def test_geodataframe_pop_attribute_preserves_projected_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    popped = gdf.pop("name")
    pop_state = get_native_state(gdf)

    assert popped.tolist() == ["a", "b"]
    assert gdf.columns.tolist() == ["value", "geometry"]
    assert pop_state is not None
    assert pop_state.column_order == ("value", "geometry")


def test_geodataframe_pop_geometry_clears_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    popped = gdf.pop("geometry")

    assert getattr(popped, "dtype", None).name == "geometry"
    assert "geometry" not in gdf.columns
    assert get_native_state(gdf) is None


def test_geodataframe_delitem_attribute_preserves_projected_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    del gdf["name"]
    del_state = get_native_state(gdf)

    assert gdf.columns.tolist() == ["value", "geometry"]
    assert del_state is not None
    assert del_state.column_order == ("value", "geometry")


def test_geodataframe_delitem_geometry_clears_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    del gdf["geometry"]

    assert "geometry" not in gdf.columns
    assert get_native_state(gdf) is None


@pytest.mark.parametrize(
    ("indexer_name", "key"),
    [
        ("loc", (0, "value")),
        ("iloc", (0, 0)),
        ("at", (0, "value")),
        ("iat", (0, 0)),
    ],
)
def test_geodataframe_indexer_assignment_clears_private_native_state(
    indexer_name: str,
    key,
) -> None:
    gdf, _state = _native_backed_geodataframe()

    getattr(gdf, indexer_name)[key] = 99

    assert get_native_state(gdf) is None
    assert gdf["value"].tolist()[0] == 99


def test_native_state_registry_rejects_public_index_mutation() -> None:
    gdf, _state = _native_backed_geodataframe()

    gdf.index = pd.Index(["a", "b"], name="zone")

    assert get_native_state(gdf) is None


def test_sjoin_bridge_declines_after_source_index_mutation() -> None:
    left, _left_state = _native_backed_geodataframe()
    right = GeoDataFrame(
        {
            "zone": ["a"],
            "geometry": GeoSeries.from_wkt(
                ["POLYGON ((-1 -1, 2 -1, 2 2, -1 2, -1 -1))"],
                name="geometry",
            ),
        }
    )
    right_result = NativeTabularResult(
        attributes=NativeAttributeTable(dataframe=right.drop(columns=["geometry"])),
        geometry=GeometryNativeResult.from_geoseries(right.geometry),
        geometry_name="geometry",
        column_order=("zone", "geometry"),
    )
    attach_native_state(right, NativeFrameState.from_native_tabular_result(right_result))
    left.index = pd.Index(["stale-a", "stale-b"], name="changed")

    export_result, _implementation, _execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
    )

    assert export_result.left_semijoin_native_frame(order="first") is None
    assert export_result.right_semijoin_native_frame(order="first") is not None


def test_sjoin_bridge_row_positions_are_not_duplicate_label_loc_semantics() -> None:
    left = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (5 5)", "POINT (20 20)"],
                name="geometry",
            ),
        }
    )
    left.index = pd.Index(["dup", "dup", "other"], name="site_id")
    left_result = NativeTabularResult(
        attributes=NativeAttributeTable(dataframe=left.drop(columns=["geometry"])),
        geometry=GeometryNativeResult.from_geoseries(left.geometry),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state(left, NativeFrameState.from_native_tabular_result(left_result))
    right = GeoDataFrame(
        {
            "zone": ["a"],
            "geometry": GeoSeries.from_wkt(
                ["POLYGON ((-1 -1, 1 -1, 1 1, -1 1, -1 -1))"],
                name="geometry",
            ),
        }
    )

    export_result, _implementation, _execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
    )
    matched = export_result.left_semijoin_native_frame(
        order="first",
        preserve_index=True,
    )
    public_joined = export_result.to_geodataframe()
    public_loc_equivalent = left.loc[public_joined.index.unique()]

    assert matched is not None
    assert matched.index_plan.name == "site_id"
    assert matched.to_native_tabular_result().attributes.index.tolist() == ["dup"]
    assert matched.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [1]
    assert public_loc_equivalent["value"].tolist() == [1, 2]
    assert export_result.left_unique_label_semijoin_native_frame() is None


def test_sjoin_bridge_unique_named_index_admits_public_label_semijoin() -> None:
    left = GeoDataFrame(
        {
            "value": [1, 2, 3],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (5 5)", "POINT (1 1)"],
                name="geometry",
            ),
        }
    )
    left.index = pd.Index(["a", "b", "c"], name="site_id")
    left_result = NativeTabularResult(
        attributes=NativeAttributeTable(dataframe=left.drop(columns=["geometry"])),
        geometry=GeometryNativeResult.from_geoseries(left.geometry),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state(left, NativeFrameState.from_native_tabular_result(left_result))
    right = GeoDataFrame(
        {
            "zone": ["z"],
            "geometry": GeoSeries.from_wkt(
                ["POLYGON ((-1 -1, 2 -1, 2 2, -1 2, -1 -1))"],
                name="geometry",
            ),
        }
    )

    export_result, _implementation, _execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
    )
    native_frame = export_result.left_unique_label_semijoin_native_frame()
    public_joined = export_result.to_geodataframe()
    public_loc_equivalent = left.loc[public_joined.index.unique()]

    assert native_frame is not None
    assert native_frame.index_plan.name == "site_id"
    assert native_frame.to_native_tabular_result().attributes.index.tolist() == ["a", "c"]
    assert native_frame.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        3,
    ]
    assert public_loc_equivalent["value"].tolist() == [1, 3]


def test_sjoin_bridge_multiindex_unique_label_semijoin_declines() -> None:
    left = GeoDataFrame(
        {
            "value": [1, 2],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (5 5)"],
                name="geometry",
            ),
        }
    )
    left.index = pd.MultiIndex.from_tuples(
        [("a", 1), ("b", 2)],
        names=["site", "rank"],
    )
    left_result = NativeTabularResult(
        attributes=NativeAttributeTable(dataframe=left.drop(columns=["geometry"])),
        geometry=GeometryNativeResult.from_geoseries(left.geometry),
        geometry_name="geometry",
        column_order=("value", "geometry"),
    )
    attach_native_state(left, NativeFrameState.from_native_tabular_result(left_result))
    right = GeoDataFrame(
        {
            "zone": ["z"],
            "geometry": GeoSeries.from_wkt(
                ["POLYGON ((-1 -1, 2 -1, 2 2, -1 2, -1 -1))"],
                name="geometry",
            ),
        }
    )

    export_result, _implementation, _execution = _sjoin_export_result(
        left,
        right,
        "inner",
        "intersects",
        None,
        "left",
        "right",
    )

    assert export_result.left_unique_label_semijoin_native_frame() is None


def test_geodataframe_delete_preserves_projected_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    del gdf["value"]

    state = get_native_state(gdf)
    assert state is not None
    assert state.column_order == ("name", "geometry")


def test_geodataframe_to_arrow_uses_private_native_state(monkeypatch: pytest.MonkeyPatch) -> None:
    gdf, _state = _native_backed_geodataframe()

    def _fail_geodataframe_export(*_args, **_kwargs):
        raise AssertionError("native-backed to_arrow should not use GeoDataFrame export")

    monkeypatch.setattr(
        "vibespatial.io.arrow.geodataframe_to_arrow",
        _fail_geodataframe_export,
    )

    table = pa.table(gdf.to_arrow())

    assert table.num_rows == 2
    assert "geometry" in table.column_names


def test_native_backed_geodataframe_to_arrow_records_public_boundary() -> None:
    gdf, _state = _native_backed_geodataframe()

    clear_materialization_events()
    table = pa.table(gdf.to_arrow())
    events = get_materialization_events(clear=True)

    assert table.num_rows == 2
    public_events = [event for event in events if event.operation == "geodataframe_to_arrow"]
    native_events = [event for event in events if event.operation == "native_tabular_to_arrow"]
    assert len(public_events) == 1
    assert public_events[0].surface == "vibespatial.api.GeoDataFrame.to_arrow"
    assert "native_export_target=arrow" in public_events[0].detail
    assert native_events == []


def test_geodataframe_to_parquet_uses_private_native_state(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gdf, _state = _native_backed_geodataframe()

    def _fail_geodataframe_export(*_args, **_kwargs):
        raise AssertionError("native-backed to_parquet should not use GeoDataFrame export")

    monkeypatch.setattr(
        "vibespatial.io.arrow.write_geoparquet",
        _fail_geodataframe_export,
    )

    path = tmp_path / "native.parquet"
    gdf.to_parquet(path)

    assert path.exists()


def test_geodataframe_to_parquet_keeps_device_attributes_and_host_index_plan(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for indexed device attribute export")
    pytest.importorskip("pylibcudf")
    gdf = GeoDataFrame(
        {
            "value": [1, 2],
            "name": ["a", "b"],
            "geometry": GeoSeries.from_wkt(
                ["POINT (0 0)", "POINT (1 1)"],
                name="geometry",
            ),
        },
        index=pd.Index([10, 20], name="parcel_id"),
    )
    state = _attach_owned_native_tabular_state(gdf, attribute_storage="device")
    assert state.attributes.device_table is not None

    captured = {}

    def _capture_native_payload(payload, path, **_kwargs):
        captured["payload"] = payload
        Path(path).write_bytes(b"")

    monkeypatch.setattr(
        "vibespatial.io.geoparquet._write_geoparquet_native_tabular_result",
        _capture_native_payload,
    )

    path = tmp_path / "native-indexed.parquet"
    clear_materialization_events()
    gdf.to_parquet(path)
    events = get_materialization_events(clear=True)

    payload = captured["payload"]
    assert path.exists()
    assert payload.attributes.dataframe is None
    assert payload.attributes.device_table is not None
    assert payload.attributes.index.equals(gdf.index)
    assert payload.index_plan.kind == "host-labels"
    assert get_native_state(gdf) is state
    assert get_native_state(gdf).attributes.device_table is not None
    assert all(event.operation != "device_attributes_to_arrow" for event in events)


def test_device_wkb_writer_encodes_indexed_view_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for indexed-view WKB writer probe")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.io.wkb import _encode_owned_wkb_column_device

    owned = from_shapely_geometries(
        [
            Point(0, 0),
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            MultiPolygon(
                [
                    Polygon([(10, 0), (11, 0), (11, 1), (10, 1)]),
                ]
            ),
            LineString([(0, 0), (1, 1)]),
        ],
        residency=Residency.DEVICE,
    )
    indexed = owned.device_take(cp.asarray([2, 1, 2, 3], dtype=cp.int64))
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    with assert_zero_d2h_transfers():
        column = _encode_owned_wkb_column_device(indexed)

    assert column.size() == 4
    assert get_d2h_transfer_events(clear=True) == []
    reset_d2h_transfer_count()


def test_native_backed_geodataframe_writers_record_public_boundaries(tmp_path) -> None:
    gdf, _state = _native_backed_geodataframe()

    parquet_path = tmp_path / "native.parquet"
    clear_materialization_events()
    gdf.to_parquet(parquet_path)
    parquet_events = get_materialization_events(clear=True)

    assert parquet_path.exists()
    public_parquet = [
        event for event in parquet_events if event.operation == "geodataframe_to_parquet"
    ]
    native_parquet = [
        event for event in parquet_events if event.operation == "native_tabular_to_parquet"
    ]
    assert len(public_parquet) == 1
    assert public_parquet[0].surface == "vibespatial.api.GeoDataFrame.to_parquet"
    assert "native_export_target=geoparquet" in public_parquet[0].detail
    assert native_parquet == []

    feather_path = tmp_path / "native.feather"
    clear_materialization_events()
    gdf.to_feather(feather_path)
    feather_events = get_materialization_events(clear=True)

    assert feather_path.exists()
    public_feather = [
        event for event in feather_events if event.operation == "geodataframe_to_feather"
    ]
    native_feather = [
        event for event in feather_events if event.operation == "native_tabular_to_feather"
    ]
    assert len(public_feather) == 1
    assert public_feather[0].surface == "vibespatial.api.GeoDataFrame.to_feather"
    assert "native_export_target=feather" in public_feather[0].detail
    assert native_feather == []


def test_geodataframe_to_feather_uses_private_native_state(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibespatial.api.io.arrow as api_arrow
    from vibespatial.api._native_result_core import NativeTabularResult

    gdf, _state = _native_backed_geodataframe()
    original_to_feather = api_arrow._to_feather
    seen_native = False

    def _assert_native_export(value, *args, **kwargs):
        nonlocal seen_native
        seen_native = isinstance(value, NativeTabularResult)
        return original_to_feather(value, *args, **kwargs)

    monkeypatch.setattr(
        "vibespatial.api.io.arrow._to_feather",
        _assert_native_export,
    )

    path = tmp_path / "native.feather"
    gdf.to_feather(path)

    assert seen_native is True
    assert path.exists()
