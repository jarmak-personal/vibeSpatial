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
from vibespatial.runtime.fallbacks import (
    StrictNativeFallbackError,
    clear_fallback_events,
    get_fallback_events,
)
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


def test_make_valid_repair_mapping_stays_device_rowset_shaped() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (
        repo_root / "src" / "vibespatial" / "constructive" / "make_valid_gpu.py"
    ).read_text()

    assert "make-valid device invalid-family-row compact fence" not in source
    assert "make-valid device invalid-global-row compact fence" not in source
    assert "make-valid device invalid-family-offset compact fence" not in source
    assert "make-valid repaired-batch valid-row compact fence" not in source
    assert "make-valid repaired-batch valid-count scalar fence" not in source
    assert "make-valid validity-expression repair-row compact fence" not in source
    assert "make-valid validity-expression null-row compact fence" not in source
    assert "make-valid ring-orientation reversal-count scalar fence" not in source
    assert "make-valid repaired-ring validity scalar fence" not in source
    assert "make-valid touching-hole ring-count scalar fence" not in source
    assert "make-valid touching-hole problem-row count scalar fence" not in source
    assert "make-valid touching-hole area-preservation count scalar fence" not in source
    assert "make-valid touching-hole repaired-validity count scalar fence" not in source
    assert "make-valid ring-closure count scalar fence" not in source
    assert "make-valid ring-closure output-size scalar fence" not in source
    assert "make-valid duplicate-vertex kept-count scalar fence" not in source
    assert "fam_to_global" not in source
    assert "d_repaired_global_mask" in source
    assert "d_accepted = problem_selection.active_capacity_mask()" in source
    touching_hole_source = source.split(
        "def _build_hole_ring_polygons_gpu",
        1,
    )[1].split("\ndef _device_scatter_repaired", 1)[0]
    assert "NativeDeviceSelection.from_mask(d_problem_mask)" in touching_hole_source
    assert "NativeDeviceSelection.from_mask(" in touching_hole_source
    assert "problem_selection.gather_capacity(" in touching_hole_source
    assert "hole_selection.gather_capacity(" in touching_hole_source
    assert "device_scatter_owned_capacity_selection(" in touching_hole_source
    assert "cp.flatnonzero(" not in touching_hole_source
    assert ".device_take(" not in touching_hole_source
    assert "_repair_multipolygon_rows_grouped_device" in source
    multipart_repair = source.split(
        "def _repair_multipolygon_rows_grouped_device",
        1,
    )[1].split("\ndef ", 1)[0]
    assert "_explode_polygonal_rows_to_polygon_capacity_gpu" in multipart_repair
    assert "NativeGroupedSelection" in multipart_repair
    assert "sentinel_group" not in multipart_repair
    assert "d_group_sizes[0] +=" in multipart_repair
    assert "allow_capacity_allocation=True" in multipart_repair
    assert "_explode_polygonal_rows_to_polygons_gpu" not in multipart_repair
    assert "cp.unique" not in multipart_repair
    assert "cp.argsort" not in multipart_repair
    repair_entrypoint = source.split("def gpu_repair_invalid_polygons", 1)[1]
    assert "_split_self_intersections_gpu" not in source
    assert "_repolygonize_from_split_rings" not in source
    assert "_detect_intra_ring_intersections" not in source
    assert "_build_batch_repaired_owned" not in source
    assert "def _extract_batch_coords(" not in source
    assert "make-valid batch polygon x-coordinate host export" not in source
    assert "make-valid batch polygon ring-offset host export" not in source
    assert "_repolygonize_owned_rows_via_overlay(" in repair_entrypoint
    topology_block = repair_entrypoint.split("d_topology_mask = (", 1)[1].split(
        "if batch_result is not None:",
        1,
    )[0]
    assert "NativeDeviceSelection.from_mask(d_topology_mask)" in topology_block
    assert "make-valid topology plan admission scalar fence" in topology_block
    assert "device_mask_owned_capacity(" in topology_block
    assert "cp.arange(batch_result.row_count" in topology_block
    assert "cp.flatnonzero(" not in topology_block
    assert "batch_result.device_take(" not in topology_block
    assert "_include_same_side_splits=True" in source
    assert 'kernels["scatter_kept_vertices"]' in source
    assert "allocation_capacity=int(d_x.size)" in source
    repaired_builder = source.split(
        "def _build_batch_repaired_device",
        1,
    )[1].split("\ndef ", 1)[0]
    assert "NativeDeviceSelection.from_mask(d_valid_rings)" in repaired_builder
    assert "active_row_count=ring_selection.logical_count" in repaired_builder
    assert "ring_selection.gather_capacity(" in repaired_builder
    assert "cp.flatnonzero" not in repaired_builder
    scatter_source = source.split(
        "def _device_scatter_repaired",
        1,
    )[1].split("\ndef ", 1)[0]
    assert "device_scatter_owned_capacity_selection(" in scatter_source
    assert "_concat_device_family_buffers" not in scatter_source
    assert "_device_take_family_buffer" not in scatter_source
    assert "cp.flatnonzero" not in scatter_source
    valid_scatter_source = source.split(
        "def _scatter_valid_repaired_batch",
        1,
    )[1].split("\ndef ", 1)[0]
    assert "selection.source_mask(active_mask=d_batch_valid)" in valid_scatter_source
    assert "device_scatter_owned_capacity_selection(" in valid_scatter_source
    assert "cp.flatnonzero" not in valid_scatter_source

    kernel_source = (
        repo_root
        / "src"
        / "vibespatial"
        / "constructive"
        / "make_valid_gpu_kernels.py"
    ).read_text()
    assert "scatter_kept_vertices" in kernel_source
    assert "generate_intra_ring_pairs" not in kernel_source
    assert "scatter_self_split_events" not in kernel_source

    face_source = (
        repo_root / "src" / "vibespatial" / "overlay" / "faces.py"
    ).read_text()
    assert 'elif operation == "polygonize"' in face_source
    assert "cp.asarray(ds.signed_area) > 0" in face_source

    pipeline_source = (
        repo_root
        / "src"
        / "vibespatial"
        / "constructive"
        / "make_valid_pipeline.py"
    ).read_text()
    assert "_gpu_polygon_validity_mask" not in pipeline_source
    assert "_detect_self_intersections_gpu" not in pipeline_source
    assert "still_invalid_rows" not in pipeline_source
    assert "_host_residual_invalid_rows_for_cpu_repair" not in pipeline_source
    assert "make-valid residual invalid-row compatibility export" not in pipeline_source

    repair_contract = source.split("class GPURepairResult", 1)[1].split(
        "def _extract_batch_coords_device",
        1,
    )[0]
    assert "still_invalid_rows" not in repair_contract
    repair_exit = source.split("def gpu_repair_invalid_polygons", 1)[1]
    assert "make-valid atomic repair completion admission scalar fence" in repair_exit
    assert "d_repaired_global_mask |= d_scattered_global_mask" in repair_exit
    boundary_source = (
        repo_root
        / "src"
        / "vibespatial"
        / "constructive"
        / "boundary_remnants.py"
    ).read_text()
    linework_source = boundary_source.split(
        "def polygon_make_valid_linework_composition_device",
        1,
    )[1].split("\n\n__all__", 1)[0]
    assert "device_mask_owned_capacity(" in linework_source
    assert "_geometry_composition_from_owned_parts_at_capacity(" in linework_source
    assert "cp.flatnonzero" not in linework_source
    assert ".device_take(" not in linework_source


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


def test_make_valid_gpu_validity_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    owned = from_shapely_geometries([bowtie])

    def _boom(*args, **kwargs):
        raise RuntimeError("gpu-detect-boom")

    monkeypatch.setattr(
        "vibespatial.constructive.validity.is_valid_owned",
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
    monkeypatch.setattr(
        "vibespatial.constructive.make_valid_pipeline._owned_all_polygon_rectangles",
        lambda *args, **kwargs: False,
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
    monkeypatch.setattr(
        "vibespatial.constructive.make_valid_pipeline._owned_all_polygon_rectangles",
        lambda *args, **kwargs: False,
    )

    with pytest.raises(RuntimeError, match="gpu-repair-boom"):
        make_valid_owned(owned=owned, dispatch_mode=ExecutionMode.GPU)


def test_incomplete_native_make_valid_declines_before_host_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    owned = from_shapely_geometries([bowtie])
    owned.device_state = SimpleNamespace(trusted_all_valid=None)

    monkeypatch.setattr(
        "vibespatial.constructive.make_valid_pipeline.plan_dispatch_selection",
        lambda *args, **kwargs: SimpleNamespace(
            selected=ExecutionMode.GPU,
            requested=ExecutionMode.GPU,
            precision_plan=None,
            reason="test",
        ),
    )
    monkeypatch.setattr(
        "vibespatial.constructive.make_valid_pipeline._try_device_validity_expression_rows",
        lambda *args, **kwargs: SimpleNamespace(
            valid_rows=np.asarray([], dtype=np.int32),
            repaired_rows=np.asarray([0], dtype=np.int32),
            null_rows=np.asarray([], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(
        "vibespatial.constructive.make_valid_pipeline._make_valid_gpu_repair",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vibespatial.constructive.make_valid_pipeline._owned_all_polygon_rectangles",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        owned,
        "to_shapely",
        lambda: pytest.fail("strict-native decline must precede host materialization"),
    )
    monkeypatch.setenv("VIBESPATIAL_STRICT_NATIVE", "1")

    clear_fallback_events()
    with pytest.warns(UserWarning, match="whole-operation shapely.make_valid"):
        with pytest.raises(StrictNativeFallbackError):
            make_valid_owned(owned=owned, dispatch_mode=ExecutionMode.GPU)
    events = get_fallback_events(clear=True)

    assert len(events) == 1
    assert events[0].d2h_transfer is True
    assert "complete aligned result" in events[0].reason


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
    owned._cached_is_valid_mask = None
    if owned.device_state is not None:
        owned.device_state.trusted_all_ogc_valid = None

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
    if owned.device_state is not None:
        owned.device_state.trusted_all_valid = None

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
    assert "make-valid validity-expression repair-count scalar fence" not in reasons
    assert "make-valid validity-expression null-count scalar fence" not in reasons
    assert "make-valid rectangle invalid-count scalar fence" not in reasons
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
def test_make_valid_uses_trusted_device_validity_before_expression(
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
    owned._cached_is_valid_mask = None
    state = owned._ensure_device_state(preserve_indexed_view=True)
    state.trusted_all_valid = True
    state.trusted_all_ogc_valid = True

    monkeypatch.setattr(
        validity_module,
        "validity_expression_owned",
        lambda *args, **kwargs: pytest.fail(
            "trusted OGC-valid make_valid should not run the device expression"
        ),
    )
    monkeypatch.setattr(
        validity_module,
        "is_valid_owned",
        lambda *args, **kwargs: pytest.fail(
            "trusted OGC-valid make_valid should not rerun public validity"
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
        and event.implementation == "trusted_ogc_valid_no_repair"
        for event in events
    )
    assert "make-valid validity-expression repair-count scalar fence" not in reasons
    reset_d2h_transfer_count()


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
    assert "make-valid validity-expression repair-row compact fence" not in reasons
    assert "make-valid validity-expression null-row compact fence" not in reasons
    assert "make-valid device invalid-family-row compact fence" not in reasons
    assert "make-valid device invalid-global-row compact fence" not in reasons
    assert "make-valid device invalid-family-offset compact fence" not in reasons
    assert "make-valid repaired-batch valid-row compact fence" not in reasons
    assert "make-valid self-split event allocation fence" not in reasons
    assert "make-valid topology plan admission scalar fence" in reasons
    assert "make-valid atomic repair completion admission scalar fence" in reasons
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
