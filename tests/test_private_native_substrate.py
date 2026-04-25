from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pandas.testing import assert_series_equal
from shapely.geometry import Point

from vibespatial.api import GeoDataFrame, GeoSeries
from vibespatial.api._native_grouped import NativeGrouped
from vibespatial.api._native_relation import NativeRelation
from vibespatial.api._native_result_core import (
    GeometryNativeResult,
    NativeAttributeTable,
    NativeGeometryColumn,
    NativeTabularResult,
    _host_array,
    _host_row_positions,
)
from vibespatial.api._native_results import (
    RelationIndexResult,
    RelationJoinExportResult,
    RelationJoinResult,
)
from vibespatial.api._native_rowset import NativeIndexPlan, NativeRowSet
from vibespatial.api._native_state import (
    NativeFrameState,
    attach_native_state,
    drop_native_state,
    get_native_state,
)
from vibespatial.api.tools.sjoin import _sjoin_export_result
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import has_gpu_runtime
from vibespatial.runtime.fallbacks import STRICT_NATIVE_ENV_VAR
from vibespatial.runtime.materialization import (
    MaterializationBoundary,
    StrictNativeMaterializationError,
    clear_materialization_events,
    get_materialization_events,
    record_materialization_event,
)
from vibespatial.runtime.residency import Residency, TransferTrigger


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
    clear_materialization_events()

    got = _host_row_positions(cp.asarray([0, 2, 4], dtype=cp.int64))

    assert got.tolist() == [0, 2, 4]
    events = get_materialization_events(clear=True)
    assert len(events) == 1
    assert events[0].surface == "vibespatial.api._native_result_core._host_row_positions"
    assert events[0].detail == "rows=3, bytes=24"


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


def test_native_frame_device_range_index_take_preserves_labels_without_hot_materialization() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device native frame take")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    attributes = NativeAttributeTable(
        arrow_table=pa.table({"value": pa.array([10, 20, 30], type=pa.int64())})
    )
    owned = from_shapely_geometries(
        [Point(0, 0), Point(1, 1), Point(2, 2)]
    ).move_to(
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
    assert payload.attributes.index.tolist() == [0, 2]
    assert any(event.operation == "index_plan_to_host" for event in events)
    assert payload.attributes.to_pandas()["value"].tolist() == [10, 30]
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


def test_all_valid_single_family_owned_device_take_avoids_scalar_d2h_probes() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device geometry take probe")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)]
    )
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
    assert relation.right_semijoin_rowset().positions.tolist() == [0, 1]
    assert relation.right_semijoin_rowset(order="first").positions.tolist() == [1, 0]
    assert relation.right_antijoin_rowset().positions.tolist() == [2]
    assert relation.right_match_counts().tolist() == [1, 2, 0]

    with pytest.raises(ValueError, match="semijoin order"):
        relation.left_semijoin_rowset(order="labels")


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
    reduced_table = grouped.reduce_numeric_columns(
        {
            "value": values,
            "rows": values,
        },
        {
            "value": "sum",
            "rows": "count",
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
    assert reduced_table.is_device is False
    assert reduced_attributes.index.equals(output_index)
    assert reduced_attributes.to_pandas()["value"].tolist() == [5, 15, 7]
    assert reduced_attributes.to_pandas()["rows"].tolist() == [2, 2, 1]


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

    assert minimum.tolist()[:2] == [2.0, 5.0]
    assert maximum.tolist()[:2] == [2.0, 10.0]
    assert firsted.tolist()[:2] == [2.0, 10.0]
    assert lasted.tolist()[:2] == [2.0, 5.0]
    assert flag_max.tolist()[:2] == [0.0, 1.0]
    assert flag_last.tolist()[:2] == [0.0, 1.0]
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
    reset_d2h_transfer_count()
    clear_materialization_events()

    with assert_zero_d2h_transfers():
        grouped = NativeGrouped.from_dense_codes(codes, group_count=3)
        reduced = grouped.reduce_numeric(values, "sum")
        minimized = grouped.reduce_numeric(values, "min")
        maximized = grouped.reduce_numeric(values, "max")
        firsted = grouped.reduce_numeric(values, "first")
        lasted = grouped.reduce_numeric(values, "last")
        reduced_table = grouped.reduce_numeric_columns(
            {
                "value": values,
                "rows": values,
            },
            {
                "value": "sum",
                "rows": "count",
            },
        )

    assert grouped.is_device
    assert reduced.is_device
    assert minimized.is_device
    assert maximized.is_device
    assert firsted.is_device
    assert lasted.is_device
    assert reduced_table.is_device
    assert get_materialization_events(clear=True) == []
    assert cp.asnumpy(reduced.values).tolist() == [5, 15, 7]
    assert cp.asnumpy(minimized.values).tolist() == [2, 5, 7]
    assert cp.asnumpy(maximized.values).tolist() == [3, 10, 7]
    assert cp.asnumpy(firsted.values).tolist() == [2, 10, 7]
    assert cp.asnumpy(lasted.values).tolist() == [3, 5, 7]
    assert cp.asnumpy(reduced_table.columns["value"].values).tolist() == [5, 15, 7]
    assert cp.asnumpy(reduced_table.columns["rows"].values).tolist() == [2, 2, 1]
    reset_d2h_transfer_count()


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
    taken = left_state.take(rowset, preserve_index=False)

    assert relation.left_token == left_state.lineage_token
    assert relation.right_token == right_state.lineage_token
    assert rowset.source_token == left_state.lineage_token
    assert taken.row_count == 2
    assert taken.to_native_tabular_result().attributes.to_pandas()["value"].tolist() == [
        1,
        2,
    ]


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
    assert unmatched_right.to_native_tabular_result().attributes.to_pandas()["zone"].tolist() == ["b"]
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

    assert export_result.left_reduce_right_numeric_columns(
        {"missing_sum": "missing"},
        {"missing_sum": "sum"},
    ) is None
    assert export_result.left_reduce_right_numeric_columns(
        {"label_count": "label"},
        {"label_count": "count"},
    ) is None
    drop_native_state(right)
    assert export_result.left_reduce_right_numeric_columns(
        {"label_count": "label"},
        {"label_count": "count"},
    ) is None


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

    assert reduced is not None
    assert reduced.is_device
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


def test_native_tabular_geodataframe_materialization_records_storage_detail() -> None:
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

    materialized = result.to_geodataframe()
    event = get_materialization_events(clear=True)[0]
    state = get_native_state(materialized)

    assert len(materialized) == 2
    assert state is not None
    assert state.row_count == 2
    assert state.column_order == ("value", "geometry")
    assert "rows=2" in event.detail
    assert "attribute_columns=1" in event.detail
    assert "attribute_storage=pandas" in event.detail
    assert "geometry_storage=geoseries" in event.detail


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


def test_geodataframe_shallow_copy_preserves_private_native_state() -> None:
    gdf, state = _native_backed_geodataframe()

    copied = gdf.copy(deep=False)

    assert get_native_state(copied) is state


def test_geodataframe_deep_copy_preserves_valid_private_native_state() -> None:
    gdf, state = _native_backed_geodataframe()

    copied = gdf.copy(deep=True)

    assert copied is not gdf
    assert get_native_state(copied) is state

    copied["new_value"] = [3, 4]

    copied_state = get_native_state(copied)
    assert copied_state is not None
    assert copied_state is not state
    assert copied_state.column_order == ("value", "name", "geometry", "new_value")
    assert get_native_state(gdf) is state


def test_geodataframe_projection_preserves_projected_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    projected = gdf[["name", "geometry"]]
    projected_state = get_native_state(projected)

    assert projected_state is not None
    assert projected_state.column_order == ("name", "geometry")
    table = pa.table(projected.to_arrow())
    assert "name" in table.column_names
    assert "value" not in table.column_names


def test_geodataframe_boolean_filter_preserves_range_index_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()
    clear_materialization_events()

    filtered = gdf[gdf["value"] > 1]
    filtered_state = get_native_state(filtered)

    assert len(filtered) == 1
    assert filtered.index.tolist() == [1]
    assert filtered_state is not None
    assert filtered_state.row_count == 1
    assert filtered_state.to_native_tabular_result().attributes.to_pandas()[
        "value"
    ].tolist() == [2]
    assert get_materialization_events(clear=True) == []


def test_geodataframe_boolean_filter_non_range_index_drops_private_native_state() -> None:
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

    assert filtered.index.tolist() == ["a"]
    assert get_native_state(filtered) is None


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


def test_geodataframe_iloc_take_preserves_projected_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    selected = gdf.iloc[[1, 0], [1, 2]]
    selected_state = get_native_state(selected)

    assert selected.columns.tolist() == ["name", "geometry"]
    assert selected.index.tolist() == [1, 0]
    assert selected_state is not None
    assert selected_state.row_count == 2
    assert selected_state.column_order == ("name", "geometry")
    assert selected_state.to_native_tabular_result().attributes.to_pandas()[
        "name"
    ].tolist() == ["b", "a"]


def test_geodataframe_take_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    selected = gdf.take([1, 0])
    selected_state = get_native_state(selected)

    assert selected["value"].tolist() == [2, 1]
    assert selected.index.tolist() == [1, 0]
    assert selected_state is not None
    assert selected_state.row_count == 2
    assert selected_state.to_native_tabular_result().attributes.to_pandas()[
        "value"
    ].tolist() == [2, 1]


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
    assert selected_state.to_native_tabular_result().attributes.to_pandas()[
        "value"
    ].tolist() == [1, 3]


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


def test_geodataframe_drop_duplicate_index_drops_private_native_state() -> None:
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
    assert get_native_state(selected) is None


def test_geodataframe_drop_inplace_clears_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    result = gdf.drop(index=[0], inplace=True)

    assert result is None
    assert gdf.index.tolist() == [1]
    assert get_native_state(gdf) is None


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
    table = NativeAttributeTable(
        dataframe=pd.DataFrame([[1, "a"]], columns=columns)
    )

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


def test_native_attribute_table_assign_columns_declines_device_payload() -> None:
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
        {"score": pd.Series([10, 20])},
        columns=("value", "name", "score"),
    )

    assert assigned is None


def test_geodataframe_rename_attribute_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    renamed = gdf.rename(columns={"value": "score"})
    renamed_state = get_native_state(renamed)

    assert renamed.columns.tolist() == ["score", "name", "geometry"]
    assert renamed_state is not None
    assert renamed_state.column_order == ("score", "name", "geometry")
    assert renamed_state.geometry_name == "geometry"
    assert renamed_state.to_native_tabular_result().attributes.to_pandas()[
        "score"
    ].tolist() == [1, 2]


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
    assert reset_state.to_native_tabular_result().attributes.index.equals(
        pd.RangeIndex(2)
    )


def test_geodataframe_reset_index_without_drop_drops_private_native_state() -> None:
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
    assert get_native_state(reset) is None


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


def test_geodataframe_set_index_duplicate_labels_drops_private_native_state() -> None:
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

    assert indexed.index.tolist() == ["same", "same"]
    assert get_native_state(indexed) is None


def test_geodataframe_set_index_multiindex_drops_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    indexed = gdf.set_index(["name", "value"])

    assert indexed.index.nlevels == 2
    assert get_native_state(indexed) is None


def test_geodataframe_set_index_append_drops_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    indexed = gdf.set_index("name", append=True)

    assert indexed.index.nlevels == 2
    assert get_native_state(indexed) is None


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
    assert reindexed_state.to_native_tabular_result().attributes.to_pandas()[
        "value"
    ].tolist() == [3, 1]


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
    assert reindexed_state.to_native_tabular_result().attributes.to_pandas()[
        "value"
    ].tolist() == [3, 1]


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


def test_geodataframe_reindex_duplicate_target_drops_private_native_state() -> None:
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

    assert reindexed.index.tolist() == ["b", "b"]
    assert get_native_state(reindexed) is None


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
    assert filtered_state.to_native_tabular_result().attributes.to_pandas()[
        "value"
    ].tolist() == [2, 3]


def test_geodataframe_filter_without_geometry_drops_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    filtered = gdf.filter(items=["value", "name"])

    assert filtered.columns.tolist() == ["value", "name"]
    assert get_native_state(filtered) is None


def test_geodataframe_assign_new_attribute_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    assigned = gdf.assign(score=[10, 20])
    assigned_state = get_native_state(assigned)

    assert assigned.columns.tolist() == ["value", "name", "geometry", "score"]
    assert assigned_state is not None
    assert assigned_state.column_order == ("value", "name", "geometry", "score")
    assert assigned_state.to_native_tabular_result().attributes.to_pandas()[
        "score"
    ].tolist() == [10, 20]


def test_geodataframe_assign_replace_attribute_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    assigned = gdf.assign(value=[10, 20])
    assigned_state = get_native_state(assigned)

    assert assigned["value"].tolist() == [10, 20]
    assert assigned_state is not None
    assert assigned_state.column_order == ("value", "name", "geometry")
    assert assigned_state.to_native_tabular_result().attributes.to_pandas()[
        "value"
    ].tolist() == [10, 20]


def test_geodataframe_assign_callable_attribute_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    assigned = gdf.assign(score=lambda frame: frame["value"] + 10)
    assigned_state = get_native_state(assigned)

    assert assigned["score"].tolist() == [11, 12]
    assert assigned_state is not None
    assert assigned_state.to_native_tabular_result().attributes.to_pandas()[
        "score"
    ].tolist() == [11, 12]


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


def test_geodataframe_assign_device_attribute_state_declines_private_native_state() -> None:
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

    assigned = gdf.assign(score=[10, 20])

    assert get_native_state(assigned) is None


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
    assert sorted_state.to_native_tabular_result().attributes.to_pandas()[
        "value"
    ].tolist() == [1, 2]


def test_geodataframe_sort_values_ignore_index_drops_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    sorted_gdf = gdf.sort_values("value", ascending=False, ignore_index=True)

    assert sorted_gdf.index.equals(pd.RangeIndex(2))
    assert get_native_state(sorted_gdf) is None


def test_geodataframe_sort_values_duplicate_index_drops_private_native_state() -> None:
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

    assert sorted_gdf["value"].tolist() == [1, 2]
    assert get_native_state(sorted_gdf) is None


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


def test_geodataframe_loc_read_does_not_lower_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    selected = gdf.loc[[0]]

    assert selected["value"].tolist() == [1]
    assert get_native_state(selected) is None


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
    assert setitem_state.to_native_tabular_result().attributes.to_pandas()[
        "value"
    ].tolist() == [10, 20]


def test_geodataframe_setitem_geometry_clears_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    gdf["geometry"] = GeoSeries.from_wkt(
        ["POINT (2 2)", "POINT (3 3)"],
        name="geometry",
    )

    assert gdf.geometry.to_wkt().tolist() == ["POINT (2 2)", "POINT (3 3)"]
    assert get_native_state(gdf) is None


def test_geodataframe_setitem_multiple_columns_clears_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    gdf[["value", "name"]] = [[10, "x"], [20, "y"]]

    assert gdf["value"].tolist() == [10, 20]
    assert get_native_state(gdf) is None


def test_geodataframe_setitem_device_attribute_state_declines_private_native_state() -> None:
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

    gdf["score"] = [10, 20]

    assert get_native_state(gdf) is None


def test_geodataframe_insert_attribute_preserves_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    result = gdf.insert(1, "score", [10, 20])
    insert_state = get_native_state(gdf)

    assert result is None
    assert gdf.columns.tolist() == ["value", "score", "name", "geometry"]
    assert insert_state is not None
    assert insert_state.column_order == ("value", "score", "name", "geometry")
    assert insert_state.to_native_tabular_result().attributes.to_pandas()[
        "score"
    ].tolist() == [10, 20]


def test_geodataframe_insert_duplicate_column_declines_private_native_state() -> None:
    gdf, _state = _native_backed_geodataframe()

    gdf.insert(1, "value", [10, 20], allow_duplicates=True)

    assert gdf.columns.tolist() == ["value", "value", "name", "geometry"]
    assert get_native_state(gdf) is None


def test_geodataframe_insert_device_attribute_state_declines_private_native_state() -> None:
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

    gdf.insert(1, "score", [10, 20])

    assert get_native_state(gdf) is None


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
