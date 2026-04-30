from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)

import vibespatial.api as geopandas
from vibespatial import (
    benchmark_make_valid,
    fusion_plan_for_make_valid,
    has_gpu_runtime,
    make_valid_owned,
    plan_make_valid_pipeline,
)
from vibespatial.api import GeoSeries
from vibespatial.api._native_result_core import NativeGeometryProvenance, NativeTabularResult
from vibespatial.api._native_rowset import NativeRowSet
from vibespatial.api.testing import assert_geoseries_equal
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events
from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events
from vibespatial.runtime.fusion import IntermediateDisposition
from vibespatial.runtime.residency import Residency


def test_make_valid_gpu_has_no_raw_cupy_scalar_syncs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "vibespatial" / "constructive" / "make_valid_gpu.py"
    tree = ast.parse(path.read_text(), filename=str(path))

    def _contains_cupy_reduction(node: ast.AST) -> bool:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cp"
            and node.func.attr in {"any", "all", "sum", "count_nonzero", "max", "min"}
        ):
            return True
        return any(_contains_cupy_reduction(child) for child in ast.iter_child_nodes(node))

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "item":
            offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}: .item()")
        if (
            isinstance(func, ast.Name)
            and func.id in {"bool", "int", "float"}
            and node.args
            and _contains_cupy_reduction(node.args[0])
        ):
            offenders.append(
                f"{path.relative_to(repo_root)}:{node.lineno}: {func.id}(cp.*)"
            )

    assert offenders == []


def test_make_valid_plan_compacts_invalid_rows_before_repair() -> None:
    plan = plan_make_valid_pipeline()

    assert [stage.name for stage in plan.stages] == [
        "compute_validity_mask",
        "compact_invalid_rows",
        "repair_invalid_topology",
        "scatter_repaired_rows",
        "emit_geometry",
    ]
    assert plan.stages[-1].disposition is IntermediateDisposition.PERSIST


def test_make_valid_fusion_plan_persists_final_geometry_only() -> None:
    fusion = fusion_plan_for_make_valid(method="structure", keep_collapsed=False)

    assert fusion.stages[-1].disposition is IntermediateDisposition.PERSIST
    assert fusion.stages[-1].steps[-1].output_name == "geometry_buffers"


def test_make_valid_owned_repairs_only_invalid_subset() -> None:
    polygon1 = Polygon([(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)])
    polygon2 = Polygon([(0, 2), (0, 1), (2, 0), (0, 0), (0, 2)])
    linestring = LineString([(0, 0), (1, 1), (1, 0)])
    result = make_valid_owned([polygon1, polygon2, linestring, None])

    expected = GeoSeries(
        [
            MultiPolygon(
                [
                    Polygon([(1, 1), (0, 0), (0, 2), (1, 1)]),
                    Polygon([(2, 0), (1, 1), (2, 2), (2, 0)]),
                ]
            ),
            GeometryCollection(
                [Polygon([(2, 0), (0, 0), (0, 1), (2, 0)]), LineString([(0, 2), (0, 1)])]
            ),
            linestring,
            None,
        ]
    )

    assert result.repaired_rows.tolist() == [0, 1]
    assert result.valid_rows.tolist() == [2]
    assert result.null_rows.tolist() == [3]
    assert_geoseries_equal(GeoSeries(result.geometries), expected)
    native = result.to_native_tabular_result()
    assert isinstance(native.provenance, NativeGeometryProvenance)
    assert native.provenance.repaired_mask.tolist() == [True, True, False, False]
    taken = native.take(np.asarray([1, 2], dtype=np.int64), preserve_index=False)
    assert isinstance(taken.provenance, NativeGeometryProvenance)
    assert taken.provenance.repaired_mask.tolist() == [True, False]


def test_geopandas_make_valid_uses_compacted_pipeline() -> None:
    polygon = Polygon([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)])
    series = geopandas.GeoSeries([polygon])
    expected = GeoSeries([MultiLineString([[(0, 0), (1, 1)], [(1, 1), (1, 2)]])])

    result = series.make_valid()

    assert_geoseries_equal(result, expected, check_geom_type=True)


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime not available")
def test_geopandas_make_valid_host_supported_values_promotes_to_device_owned() -> None:
    series = geopandas.GeoSeries([
        Polygon([(0, 0), (2, 0), (2, 1), (0, 1), (0, 0)]),
        Polygon([(3, 0), (5, 0), (5, 1), (3, 1), (3, 0)]),
    ])

    clear_dispatch_events()
    result = series.make_valid()
    events = get_dispatch_events(clear=True)

    assert getattr(result.values, "_owned", None) is not None
    assert result.values._owned.residency is Residency.DEVICE
    assert any(
        event.surface == "geopandas.array.make_valid"
        and event.selected is ExecutionMode.GPU
        for event in events
    )


def test_make_valid_benchmark_reports_repaired_rows() -> None:
    values = [
        Polygon([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)]),
        Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
    ]
    benchmark = benchmark_make_valid(values)

    assert benchmark.rows == 2
    assert benchmark.repaired_rows == 1
    assert benchmark.compact_elapsed_seconds >= 0.0
    assert benchmark.baseline_elapsed_seconds >= 0.0


def test_make_valid_gpu_detection_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    owned = from_shapely_geometries([bowtie])

    def _boom(*args, **kwargs):
        raise RuntimeError("gpu-detect-boom")

    monkeypatch.setattr(
        "vibespatial.constructive.validity.is_valid_owned",
        lambda owned, **kwargs: np.array([False], dtype=bool),
    )
    monkeypatch.setattr(
        "vibespatial.constructive.make_valid_pipeline._detect_self_intersections_gpu",
        _boom,
    )
    monkeypatch.setattr(
        "vibespatial.constructive.make_valid_pipeline._try_device_validity_expression_rows",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vibespatial.constructive.make_valid_pipeline.plan_dispatch_selection",
        lambda *args, **kwargs: SimpleNamespace(
            selected=ExecutionMode.GPU,
            requested=ExecutionMode.GPU,
            precision_plan=None,
            reason="test",
        ),
    )

    with pytest.raises(RuntimeError, match="gpu-detect-boom"):
        make_valid_owned(owned=owned, dispatch_mode=ExecutionMode.GPU)


def test_make_valid_gpu_repair_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    owned = from_shapely_geometries([bowtie])

    def _boom(*args, **kwargs):
        raise RuntimeError("gpu-repair-boom")

    monkeypatch.setattr(
        "vibespatial.constructive.validity.is_valid_owned",
        lambda owned, **kwargs: np.array([False], dtype=bool),
    )
    monkeypatch.setattr(
        "vibespatial.constructive.make_valid_pipeline._detect_self_intersections_gpu",
        lambda owned, valid_mask: valid_mask,
    )
    monkeypatch.setattr(
        "vibespatial.constructive.make_valid_pipeline._make_valid_gpu_repair",
        _boom,
    )
    monkeypatch.setattr(
        "vibespatial.constructive.make_valid_pipeline.plan_dispatch_selection",
        lambda *args, **kwargs: SimpleNamespace(
            selected=ExecutionMode.GPU,
            requested=ExecutionMode.GPU,
            precision_plan=None,
            reason="test",
        ),
    )

    with pytest.raises(RuntimeError, match="gpu-repair-boom"):
        make_valid_owned(owned=owned, dispatch_mode=ExecutionMode.GPU)


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime not available")
def test_make_valid_auto_keeps_inner_is_valid_on_gpu_for_device_resident_owned() -> None:
    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1), (2, 0)]),
        ],
        residency=Residency.DEVICE,
    )

    clear_dispatch_events()
    result = make_valid_owned(owned=owned, dispatch_mode=ExecutionMode.AUTO)
    events = get_dispatch_events(clear=True)

    assert result.owned is owned
    assert all(
        not (
            event.surface == "geopandas.array.is_valid"
            and event.selected is ExecutionMode.CPU
        )
        for event in events
    )
    assert any(
        event.surface == "geopandas.array.make_valid"
        and event.selected is ExecutionMode.GPU
        for event in events
    )


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime not available")
def test_make_valid_gpu_rectangles_skip_generic_validity_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibespatial.constructive.validity as validity_module
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (4, 0), (4, 2), (0, 2), (0, 0)]),
            Polygon([(10, 5), (13, 5), (13, 9), (10, 9), (10, 5)]),
        ],
        residency=Residency.DEVICE,
    )
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None

    monkeypatch.setattr(
        validity_module,
        "is_valid_owned",
        lambda *args, **kwargs: pytest.fail(
            "exact rectangle batches should return from make_valid before the generic validity scan"
        ),
    )

    clear_dispatch_events()
    reset_d2h_transfer_count()
    result = make_valid_owned(owned=owned, dispatch_mode=ExecutionMode.GPU)
    transfers = get_d2h_transfer_events(clear=True)
    events = get_dispatch_events(clear=True)
    reasons = {event.reason for event in transfers}

    assert result.owned is owned
    assert result.valid_rows.tolist() == [0, 1]
    assert result.repaired_rows.size == 0
    assert result.null_rows.size == 0
    assert any(
        event.surface == "geopandas.array.make_valid"
        and event.implementation == "rectangle_valid_fast_path"
        for event in events
    )
    assert "polygon-rectangle dense single-ring scalar fence" not in reasons
    assert "polygon-rectangle empty-mask scalar fence" not in reasons
    assert "polygon-rectangle ring-offset scalar fence" not in reasons
    assert "owned geometry host metadata validity boundary" not in reasons
    reset_d2h_transfer_count()


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime not available")
def test_make_valid_gpu_no_repair_uses_native_validity_expression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibespatial.constructive.validity as validity_module
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (2, 0), (1, 1), (0, 0)]),
            Polygon([(3, 0), (5, 0), (4, 2), (3, 0)]),
        ],
        residency=Residency.DEVICE,
    )
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None

    monkeypatch.setattr(
        validity_module,
        "is_valid_owned",
        lambda *args, **kwargs: pytest.fail(
            "no-repair device make_valid should use validity_expression_owned"
        ),
    )

    clear_dispatch_events()
    reset_d2h_transfer_count()
    result = make_valid_owned(owned=owned, dispatch_mode=ExecutionMode.GPU)
    transfers = get_d2h_transfer_events(clear=True)
    events = get_dispatch_events(clear=True)
    reasons = {event.reason for event in transfers}

    assert result.owned is owned
    assert result.valid_rows.tolist() == [0, 1]
    assert result.repaired_rows.size == 0
    assert result.null_rows.size == 0
    assert any(
        event.surface == "geopandas.array.make_valid"
        and event.implementation == "validity_expression_no_repair"
        for event in events
    )
    assert "make-valid validity-expression repair-count scalar fence" in reasons
    assert "make-valid validity-expression null-count scalar fence" in reasons
    assert "make-valid polygon validity mask host export" not in reasons
    assert "owned geometry host metadata validity boundary" not in reasons
    reset_d2h_transfer_count()


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime not available")
def test_make_valid_uses_owned_validity_cache_before_device_expression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibespatial.constructive.validity as validity_module

    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (2, 0), (1, 1), (0, 0)]),
            Polygon([(3, 0), (5, 0), (4, 2), (3, 0)]),
        ],
        residency=Residency.DEVICE,
    )
    owned._cached_is_valid_mask = np.ones(owned.row_count, dtype=bool)

    monkeypatch.setattr(
        validity_module,
        "validity_expression_owned",
        lambda *args, **kwargs: pytest.fail(
            "cached-validity make_valid should not rerun the device expression"
        ),
    )
    monkeypatch.setattr(
        validity_module,
        "is_valid_owned",
        lambda *args, **kwargs: pytest.fail(
            "cached-validity make_valid should not rerun public validity"
        ),
    )

    clear_dispatch_events()
    result = make_valid_owned(owned=owned, dispatch_mode=ExecutionMode.GPU)
    events = get_dispatch_events(clear=True)

    assert result.owned is owned
    assert result.valid_rows.tolist() == [0, 1]
    assert result.repaired_rows.size == 0
    assert result.null_rows.size == 0
    assert any(
        event.surface == "geopandas.array.make_valid"
        and event.implementation == "cached_validity_no_repair"
        for event in events
    )


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime not available")
def test_make_valid_gpu_repair_fences_are_operation_named() -> None:
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)]),
            Polygon([(3, 0), (4, 0), (4, 1), (3, 1), (3, 0)]),
        ],
        residency=Residency.DEVICE,
    )
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None

    reset_d2h_transfer_count()
    result = make_valid_owned(owned=owned, dispatch_mode=ExecutionMode.GPU)
    events = get_d2h_transfer_events(clear=True)
    reasons = {event.reason for event in events}

    assert result.owned is not None
    assert result.repaired_rows.tolist() == [0]
    assert "make-valid validity-expression repair-row compact fence" in reasons
    assert "make-valid device invalid-family-row compact fence" in reasons
    assert "make-valid self-split event allocation fence" in reasons
    assert "owned geometry device-take nested slice-size allocation fence" not in reasons
    assert "make-valid polygon validity mask host export" not in reasons
    assert "owned geometry host metadata validity boundary" not in reasons
    assert "owned geometry host metadata family-tag boundary" not in reasons
    assert "owned geometry host metadata family-row-offset boundary" not in reasons
    assert "count-scatter total allocation fence" not in reasons
    assert "CudaRuntime.copy_device_to_host_async" not in reasons
    reset_d2h_transfer_count()


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime not available")
def test_make_valid_gpu_repairs_adjacent_hole_rings_without_cpu_fallback() -> None:
    from vibespatial.constructive.validity import is_valid_owned

    invalid = Polygon(
        [(0, 4), (0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[
            [(2, 3), (2, 1), (1, 1), (1, 3), (2, 3)],
            [(3, 3), (3, 1), (2, 1), (2, 3), (3, 3)],
        ],
    )
    assert not invalid.is_valid

    owned = from_shapely_geometries([invalid], residency=Residency.DEVICE)
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None

    clear_fallback_events()
    result = make_valid_owned(owned=owned, dispatch_mode=ExecutionMode.GPU)
    fallback_events = get_fallback_events(clear=True)

    assert result.owned is not None
    assert result.repaired_rows.tolist() == [0]
    assert fallback_events == []
    valid = is_valid_owned(
        result.owned,
        dispatch_mode=ExecutionMode.GPU,
        _exact_collinearity=True,
    )
    assert valid.tolist() == [True]
    repaired = result.owned.to_shapely()[0]
    assert repaired.is_valid
    assert repaired.area == pytest.approx(12.0)


def test_make_valid_result_lowers_to_native_tabular_result_with_provenance() -> None:
    index = pd.Index(["a", "b"], name="row")
    owned = from_shapely_geometries(
        [
            Point(0, 0),
            Point(1, 1),
        ]
    )

    result = make_valid_owned(owned=owned, dispatch_mode=ExecutionMode.CPU)
    tabular = result.to_native_tabular_result(
        crs="EPSG:4326",
        geometry_name="geometry",
        index=index,
    )
    state = tabular.to_native_frame_state()
    rowset = NativeRowSet.from_positions(
        np.asarray([1], dtype=np.int64),
        source_token=state.lineage_token,
        source_row_count=state.row_count,
    )
    taken = state.take(rowset, preserve_index=True)

    assert isinstance(tabular, NativeTabularResult)
    assert tabular.geometry.owned is owned
    assert isinstance(tabular.provenance, NativeGeometryProvenance)
    assert tabular.provenance.operation == "make_valid"
    assert tabular.provenance.source_rows.tolist() == [0, 1]
    assert tabular.provenance.repaired_mask.tolist() == [False, False]
    assert tabular.geometry_metadata is not None
    assert taken.index_plan.to_public_index().tolist() == ["b"]
    assert isinstance(taken.provenance, NativeGeometryProvenance)
    assert taken.provenance.source_rows.tolist() == [1]
    assert taken.provenance.repaired_mask.tolist() == [False]


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime not available")
def test_make_valid_result_native_tabular_device_provenance_survives_rowset_take() -> None:
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (4, 0), (4, 2), (0, 2), (0, 0)]),
            Polygon([(10, 5), (13, 5), (13, 9), (10, 9), (10, 5)]),
        ],
        residency=Residency.DEVICE,
    )
    result = make_valid_owned(owned=owned, dispatch_mode=ExecutionMode.GPU)
    reset_d2h_transfer_count()

    with assert_zero_d2h_transfers():
        tabular = result.to_native_tabular_result(crs=None)
        state = tabular.to_native_frame_state()
        rowset = NativeRowSet.from_positions(
            cp.asarray([1], dtype=cp.int32),
            source_token=state.lineage_token,
            source_row_count=state.row_count,
        )
        taken = state.take(rowset, preserve_index=True)

    assert isinstance(tabular.provenance, NativeGeometryProvenance)
    assert tabular.provenance.is_device
    assert isinstance(taken.provenance, NativeGeometryProvenance)
    assert taken.provenance.is_device
    assert cp.asnumpy(taken.provenance.source_rows).tolist() == [1]
    assert cp.asnumpy(taken.provenance.repaired_mask).tolist() == [False]
    reset_d2h_transfer_count()
