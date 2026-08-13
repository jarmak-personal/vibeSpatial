from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import shapely
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box

import vibespatial.api as geopandas
from vibespatial import (
    DissolveUnionMethod,
    benchmark_dissolve_pipeline,
    fusion_plan_for_dissolve,
    has_gpu_runtime,
    plan_dissolve_pipeline,
)
from vibespatial.api._native_grouped import NativeGrouped, NativeGroupedAttributeReduction
from vibespatial.api._native_result_core import NativeAttributeTable
from vibespatial.api._native_results import GeometryNativeResult, NativeTabularResult
from vibespatial.api._native_state import NativeFrameState, attach_native_state, get_native_state
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.api.testing import assert_geodataframe_equal
from vibespatial.bench.pipeline import _dissolve_join_heavy_groups, _regular_polygons_frame
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import (
    OwnedGeometryArray,
    from_shapely_geometries,
    seed_all_validity_cache,
)
from vibespatial.kernels.constructive.segmented_union import segmented_union_all
from vibespatial.overlay.dissolve import (
    evaluate_geopandas_dissolve,
    evaluate_geopandas_dissolve_native,
    execute_grouped_union,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.fallbacks import StrictNativeFallbackError
from vibespatial.runtime.fusion import IntermediateDisposition
from vibespatial.runtime.materialization import (
    clear_materialization_events,
    get_materialization_events,
)
from vibespatial.runtime.provenance import clear_rewrite_events, get_rewrite_events
from vibespatial.runtime.residency import Residency
from vibespatial.testing import strict_native_environment

dissolve_module = importlib.import_module("vibespatial.overlay.dissolve")


def test_dissolve_pipeline_plan_uses_group_encoding_and_grouped_union() -> None:
    plan = plan_dissolve_pipeline("unary")

    assert [stage.name for stage in plan.stages] == [
        "encode_groups",
        "stable_sort_rows",
        "segment_groups",
        "aggregate_attributes",
        "union_group_geometries",
        "assemble_result_frame",
    ]
    assert plan.stages[-1].disposition is IntermediateDisposition.PERSIST
    assert plan.stages[-1].geometry_producing is True


def test_dissolve_fusion_plan_persists_final_frame_only() -> None:
    fusion = fusion_plan_for_dissolve(DissolveUnionMethod.COVERAGE)

    assert len(fusion.stages) >= 2
    assert fusion.stages[-1].disposition is IntermediateDisposition.PERSIST
    assert fusion.stages[-1].steps[-1].output_name == "dissolved_frame"


def test_public_dissolve_exports_grouped_attributes_with_pandas_dtypes() -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": [0, 0, 1],
            "label": pd.Series(["alpha", "beta", "gamma"], dtype="string"),
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(4, 0, 5, 1)],
        }
    )

    result = frame.dissolve("group", aggfunc="first")

    assert result["label"].dtype == frame["label"].dtype
    assert result["label"].tolist() == ["alpha", "gamma"]
    assert get_native_state(result) is not None


def test_execute_grouped_union_emits_empty_geometry_for_unobserved_group() -> None:
    geometries = [Point(0, 0), Point(1, 1)]
    grouped = execute_grouped_union(
        geometries, [pd.Index([0]).to_numpy(), pd.Index([], dtype="int64").to_numpy()]
    )

    assert grouped.group_count == 2
    assert grouped.empty_groups == 1
    assert grouped.geometries[1].geom_type == "GeometryCollection"
    assert grouped.geometries[1].is_empty


@pytest.mark.gpu
def test_execute_grouped_union_owned_coverage_avoids_geometry_materialization() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    class ExplodingGeometries:
        def __array__(self, dtype=None):
            raise AssertionError("owned coverage dissolve materialized geometry objects")

    owned = from_shapely_geometries([box(0, 0, 1, 1), box(2, 0, 3, 1)])
    grouped = execute_grouped_union(
        ExplodingGeometries(),
        [np.asarray([0], dtype=np.int32), np.asarray([1], dtype=np.int32)],
        method=DissolveUnionMethod.COVERAGE,
        owned=owned,
    )

    assert grouped.geometries is None
    assert grouped.owned is not None
    assert grouped.owned.row_count == 2


@pytest.mark.gpu
def test_execute_grouped_union_codes_owned_coverage_avoids_geometry_materialization(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    class ExplodingGeometries:
        def __array__(self, dtype=None):
            raise AssertionError("owned coverage dissolve codes path materialized geometry objects")

    owned = from_shapely_geometries(
        [box(0, 0, 1, 1), box(2, 0, 3, 1)],
        residency=Residency.DEVICE,
    )
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None

    def _fail_host_metadata(*_args, **_kwargs):
        raise AssertionError("device-owned coverage path should not materialize host metadata")

    monkeypatch.setattr(type(owned), "_ensure_host_metadata", _fail_host_metadata)

    grouped = dissolve_module.execute_grouped_union_codes(
        ExplodingGeometries(),
        np.asarray([0, 1], dtype=np.int32),
        group_count=2,
        method=DissolveUnionMethod.COVERAGE,
        owned=owned,
    )

    assert grouped is not None
    assert grouped.geometries is None
    assert grouped.owned is not None
    assert grouped.owned.row_count == 2


@pytest.mark.gpu
def test_grouped_union_payload_device_repair_gate_avoids_host_metadata(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [box(0, 0, 1, 1), box(2, 0, 3, 1)],
        residency=Residency.DEVICE,
    )
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None

    def _fail_host_metadata(*_args, **_kwargs):
        raise AssertionError("grouped union repair gate should stay device-native")

    monkeypatch.setattr(type(owned), "_ensure_host_metadata", _fail_host_metadata)
    reset_d2h_transfer_count()

    grouped = dissolve_module.GroupedUnionResult(
        geometries=None,
        group_count=2,
        non_empty_groups=2,
        empty_groups=0,
        method=DissolveUnionMethod.COVERAGE,
        owned=owned,
    )

    payload = dissolve_module._grouped_union_geometry_payload(
        grouped,
        geometry_name="geometry",
        crs=None,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert payload.geometry.owned is not None
    assert payload.repaired is False
    assert reasons == []
    assert payload.geometry.owned._validity is None
    assert payload.geometry.owned._tags is None
    assert payload.geometry.owned._family_row_offsets is None


def test_grouped_union_repair_gate_returns_sparse_invalid_rows() -> None:
    owned = from_shapely_geometries(
        [box(0, 0, 1, 1), box(2, 0, 3, 1), box(4, 0, 5, 1)],
        residency=Residency.HOST,
    )
    owned._cached_is_valid_mask = np.asarray([True, False, True], dtype=bool)

    invalid_rows = dissolve_module._device_grouped_union_invalid_rows(owned)

    assert invalid_rows.tolist() == [1]


@pytest.mark.gpu
def test_grouped_union_payload_reuses_validity_cache_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [box(0, 0, 1, 1), box(2, 0, 3, 1)],
        residency=Residency.DEVICE,
    )
    seed_all_validity_cache(owned)
    reset_d2h_transfer_count()

    grouped = dissolve_module.GroupedUnionResult(
        geometries=None,
        group_count=2,
        non_empty_groups=2,
        empty_groups=0,
        method=DissolveUnionMethod.COVERAGE,
        owned=owned,
    )

    payload = dissolve_module._grouped_union_geometry_payload(
        grouped,
        geometry_name="geometry",
        crs=None,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert payload.geometry.owned is owned
    assert payload.repaired is False
    assert reasons == []


@pytest.mark.gpu
def test_unary_dissolve_device_path_does_not_materialize_host_metadata() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(0.5, 0.0, 1.5, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
        ],
        residency=Residency.DEVICE,
    )
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None
    frame = geopandas.GeoDataFrame(
        {"value": np.asarray([1, 2, 3], dtype=np.int64)},
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(owned),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )

    reset_d2h_transfer_count()
    with strict_native_environment():
        result = frame.dissolve(by=None)
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result.geometry.dtype.name == "device_geometry"
    assert not any(reason.startswith("owned geometry host metadata") for reason in reasons)


@pytest.mark.gpu
def test_cached_device_row_bounds_public_export_avoids_host_metadata_cache(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.kernels.core.geometry_analysis import (
        compute_geometry_bounds,
        compute_geometry_bounds_device,
    )

    owned = from_shapely_geometries(
        [box(0, 0, 1, 1), box(2, 0, 3, 1)],
        residency=Residency.DEVICE,
    )
    compute_geometry_bounds_device(owned)
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None

    def _fail_host_metadata(*_args, **_kwargs):
        raise AssertionError("cached row-bounds export should not cache host metadata")

    monkeypatch.setattr(type(owned), "_ensure_host_metadata", _fail_host_metadata)
    reset_d2h_transfer_count()

    bounds = compute_geometry_bounds(owned, dispatch_mode=ExecutionMode.GPU)
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert bounds.shape == (2, 4)
    assert reasons == ["geometry analysis cached row-bounds host export"]
    assert owned._validity is None
    assert owned._tags is None
    assert owned._family_row_offsets is None


def test_union_all_has_no_host_bounds_probe() -> None:
    source = (
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("src/vibespatial/constructive/union_all.py")
        .read_text()
    )

    assert "_compute_union_bounds_host" not in source


@pytest.mark.gpu
def test_union_all_bbox_interaction_proof_uses_scalar_device_fence() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.constructive.union_all import (
        _polygon_inputs_have_bbox_interactions_requiring_exact_union,
    )
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [
            box(0, 0, 2, 2),
            box(1, 1, 3, 3),
            box(10, 10, 11, 11),
        ],
        residency=Residency.DEVICE,
    )
    reset_d2h_transfer_count()

    assert _polygon_inputs_have_bbox_interactions_requiring_exact_union(owned) is True
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert reasons == ["union_all bbox-interaction overlap scalar fence"]


def test_union_all_bbox_interaction_has_no_large_batch_host_crossover() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "vibespatial" / "constructive" / "union_all.py").read_text()
    kernels = (root / "src" / "vibespatial" / "constructive" / "union_all_kernels.py").read_text()

    assert "_BBOX_INTERACTION_DEVICE_PAIR_LIMIT" not in source
    assert "union_all bbox-interaction candidate bounds fence" not in source
    assert "bbox_any_overlap_sorted" in kernels
    assert "const double* __restrict__ xmin" in kernels


@pytest.mark.gpu
def test_union_all_large_bbox_interaction_proof_stays_scalar_device_shaped() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.constructive.union_all import (
        _device_polygon_bbox_interactions_requiring_exact_union,
    )
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    row_count = 5_000
    xmin = cp.arange(row_count, dtype=cp.float64) * 2.0
    d_bounds = cp.column_stack((xmin, cp.zeros(row_count), xmin + 1.0, cp.ones(row_count)))
    d_validity = cp.ones(row_count, dtype=cp.bool_)
    reset_d2h_transfer_count()

    assert not _device_polygon_bbox_interactions_requiring_exact_union(
        d_bounds,
        d_validity,
        row_count=row_count,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    assert reasons == ["union_all bbox-interaction overlap scalar fence"]

    d_bounds[-1] = cp.asarray([0.5, 0.25, 1.5, 0.75], dtype=cp.float64)
    reset_d2h_transfer_count()
    assert _device_polygon_bbox_interactions_requiring_exact_union(
        d_bounds,
        d_validity,
        row_count=row_count,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    assert reasons == ["union_all bbox-interaction overlap scalar fence"]


@pytest.mark.gpu
def test_cache_bounds_reuses_existing_host_metadata_without_runtime_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [box(0, 0, 1, 1), box(2, 0, 3, 1)],
        residency=Residency.DEVICE,
    )
    reset_d2h_transfer_count()

    owned.cache_bounds(
        np.asarray(
            [
                [0.0, 0.0, 1.0, 1.0],
                [2.0, 0.0, 3.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert reasons == []


@pytest.mark.gpu
def test_grouped_union_invalid_recompute_valid_device_rows_avoids_host_metadata(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [box(0, 0, 1, 1), box(2, 0, 3, 1)],
        residency=Residency.DEVICE,
    )
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None

    def _fail_host_metadata(*_args, **_kwargs):
        raise AssertionError("valid grouped-union recompute should stay device-native")

    monkeypatch.setattr(type(owned), "_ensure_host_metadata", _fail_host_metadata)
    reset_d2h_transfer_count()

    result = dissolve_module._recompute_invalid_grouped_union_owned_rows(
        owned,
        ordered_owned=owned,
        offsets=np.asarray([0, 1, 2], dtype=np.int32),
        group_count=2,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result is owned
    assert reasons == []
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None


@pytest.mark.gpu
def test_grouped_union_invalid_recompute_uses_gpu_make_valid_before_host_recompute(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.constructive import make_valid_gpu as make_valid_module

    cp = pytest.importorskip("cupy")

    owned = from_shapely_geometries(
        [box(0, 0, 1, 1), box(0.5, 0.5, 1.5, 1.5)],
        residency=Residency.DEVICE,
    )
    repair_needed_calls = 0

    def _invalid_rows(_owned, *, exact_collinearity=False):
        nonlocal repair_needed_calls
        assert exact_collinearity is True
        repair_needed_calls += 1
        if repair_needed_calls == 1:
            return cp.asarray([0], dtype=cp.int64)
        return cp.empty(0, dtype=cp.int64)

    def _fake_gpu_repair_invalid_polygons(
        owned,
        invalid_rows,
        *,
        method,
        keep_collapsed,
    ):
        assert owned is not None
        assert invalid_rows.size == 1
        assert method == "linework"
        assert keep_collapsed is True
        return SimpleNamespace(
            repaired_owned=owned,
            repaired_count=1,
        )

    def _fail_host_recompute(*_args, **_kwargs):
        raise AssertionError("GPU make-valid repair should avoid Shapely recompute")

    monkeypatch.setattr(dissolve_module, "_device_grouped_union_invalid_rows", _invalid_rows)
    monkeypatch.setattr(
        make_valid_module,
        "gpu_repair_invalid_polygons",
        _fake_gpu_repair_invalid_polygons,
    )
    monkeypatch.setattr(dissolve_module.shapely, "union_all", _fail_host_recompute)

    with strict_native_environment():
        result = dissolve_module._recompute_invalid_grouped_union_owned_rows(
            owned,
            ordered_owned=owned,
            offsets=np.asarray([0, 1, 2], dtype=np.int32),
            group_count=2,
        )

    assert result.row_count == owned.row_count
    assert result.residency is Residency.DEVICE
    assert repair_needed_calls == 2


@pytest.mark.gpu
def test_execute_grouped_union_codes_device_codes_reports_host_fallback_materialization() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    owned = from_shapely_geometries(
        [
            LineString([(0.0, 0.0), (1.0, 0.0)]),
            LineString([(1.0, 0.0), (2.0, 0.0)]),
        ],
        residency=Residency.DEVICE,
    )
    row_group_codes = cp.asarray([0, 0], dtype=cp.int32)
    clear_materialization_events()

    grouped = dissolve_module.execute_grouped_union_codes(
        (),
        row_group_codes,
        group_count=1,
        method=DissolveUnionMethod.COVERAGE,
        owned=owned,
    )

    assert grouped is None
    events = get_materialization_events(clear=True)
    assert len(events) == 1
    assert events[0].operation == "device_group_codes_to_host"
    assert events[0].d2h_transfer is True
    assert events[0].detail == "rows=2, bytes=8"


@pytest.mark.gpu
def test_execute_grouped_union_codes_device_codes_bulk_disjoint_coverage_stays_native() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    geoms = [
        box(0.0, 0.0, 1.0, 1.0),
        box(10.0, 0.0, 11.0, 1.0),
        box(0.0, 10.0, 1.0, 11.0),
        box(10.0, 10.0, 11.0, 11.0),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    row_group_codes = cp.asarray([0, 1, 0, 1], dtype=cp.int32)
    clear_materialization_events()
    reset_d2h_transfer_count()

    with assert_zero_d2h_transfers():
        grouped = dissolve_module.execute_grouped_union_codes(
            (),
            row_group_codes,
            group_count=2,
            method=DissolveUnionMethod.COVERAGE,
            owned=owned,
        )

    assert grouped is not None
    assert grouped.geometries is None
    assert grouped.owned is not None
    assert grouped.owned.row_count == 2
    assert get_materialization_events(clear=True) == []
    actual = grouped.owned.to_shapely()
    expected = [
        shapely.coverage_union_all(np.asarray([geoms[0], geoms[2]], dtype=object)),
        shapely.coverage_union_all(np.asarray([geoms[1], geoms[3]], dtype=object)),
    ]
    assert shapely.equals(actual[0], expected[0])
    assert shapely.equals(actual[1], expected[1])


@pytest.mark.gpu
def test_execute_grouped_union_codes_low_fan_in_dropped_rows_skip_admission_probes() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    geoms = [
        box(0.0, 0.0, 1.0, 1.0),
        box(1.0, 0.0, 2.0, 1.0),
        box(10.0, 0.0, 11.0, 1.0),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    reset_d2h_transfer_count()

    grouped = dissolve_module.execute_grouped_union_codes(
        (),
        cp.asarray([0, 0, -1], dtype=cp.int32),
        group_count=3,
        method=DissolveUnionMethod.COVERAGE,
        owned=owned,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert grouped is not None
    assert grouped.geometries is None
    assert grouped.owned is not None
    assert grouped.owned.row_count == 3
    assert "dissolve rectangle bounds structural scalar fence" not in reasons
    assert "dissolve rectangle bounds axis-aligned area scalar fence" not in reasons
    assert "dissolve grouped-box observed-row scalar fence" not in reasons
    assert "dissolve grouped-box strip-coverage scalar fence" not in reasons
    assert "dissolve disjoint-subset admissibility scalar fence" not in reasons

    actual = np.asarray(grouped.owned.to_shapely(), dtype=object)
    assert shapely.equals(
        actual[0],
        shapely.coverage_union_all(np.asarray(geoms[:2], dtype=object)),
    )
    assert actual[1] is not None and actual[1].is_empty
    assert actual[2] is not None and actual[2].is_empty


@pytest.mark.gpu
def test_execute_grouped_union_codes_device_unary_uses_native_grouped_union() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    geoms = [
        box(0.0, 0.0, 1.0, 1.0),
        box(1.0, 0.0, 2.0, 1.0),
        box(10.0, 0.0, 11.0, 1.0),
        box(11.0, 0.0, 12.0, 1.0),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    row_group_codes = cp.asarray([0, 0, 1, 1], dtype=cp.int32)
    clear_materialization_events()

    grouped = dissolve_module.execute_grouped_union_codes(
        (),
        row_group_codes,
        group_count=2,
        method=DissolveUnionMethod.UNARY,
        owned=owned,
    )

    assert grouped is not None
    assert grouped.geometries is None
    assert grouped.owned is not None
    assert grouped.owned.residency is Residency.DEVICE
    assert grouped.owned.row_count == 2
    assert grouped.non_empty_groups == 2
    assert get_materialization_events(clear=True) == []

    actual = grouped.owned.to_shapely()
    expected = [
        shapely.union_all(np.asarray([geoms[0], geoms[1]], dtype=object)),
        shapely.union_all(np.asarray([geoms[2], geoms[3]], dtype=object)),
    ]
    assert shapely.area(shapely.symmetric_difference(actual[0], expected[0])) == 0.0
    assert shapely.area(shapely.symmetric_difference(actual[1], expected[1])) == 0.0


@pytest.mark.gpu
def test_execute_native_grouped_union_device_disjoint_subset_uses_native_grouped_carrier() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    geoms = [
        box(0.0, 0.0, 1.0, 1.0),
        box(2.0, 0.0, 3.0, 1.0),
        box(10.0, 0.0, 11.0, 1.0),
        box(12.0, 0.0, 13.0, 1.0),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_sorted_offsets(
        cp.asarray([0, 2, 4], dtype=cp.int64),
        row_count=owned.row_count,
        source_token="device-disjoint-test",
    )
    clear_dispatch_events()

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method=DissolveUnionMethod.DISJOINT_SUBSET,
        owned=owned,
    )
    events = get_dispatch_events(clear=True)

    assert result is not None
    assert result.geometries is None
    assert result.owned is not None
    assert result.owned.residency is Residency.DEVICE
    assert result.owned.row_count == 2
    assert any(
        event.implementation == "native_grouped_disjoint_subset_device_assembly" for event in events
    )

    actual = np.asarray(result.owned.to_shapely(), dtype=object)
    expected = [
        shapely.disjoint_subset_union_all(np.asarray(geoms[:2], dtype=object)),
        shapely.disjoint_subset_union_all(np.asarray(geoms[2:], dtype=object)),
    ]
    assert all(
        shapely.normalize(got).equals_exact(shapely.normalize(want), tolerance=1.0e-9)
        for got, want in zip(actual, expected, strict=True)
    )


@pytest.mark.gpu
def test_native_grouped_unary_union_uses_grouped_disjoint_pack_above_global_cap() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    group_count = 260
    geoms = []
    for group in range(group_count):
        x0 = float(group * 10)
        geoms.append(box(x0, 0.0, x0 + 1.0, 1.0))
        geoms.append(box(x0 + 3.0, 0.0, x0 + 4.0, 1.0))
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_sorted_offsets(
        cp.arange(group_count + 1, dtype=cp.int64) * np.int64(2),
        row_count=owned.row_count,
        source_token="device-disjoint-pack-test",
        all_groups_observed=True,
        group_size_min=2,
        group_size_max=2,
    )
    clear_dispatch_events()

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method=DissolveUnionMethod.UNARY,
        owned=owned,
    )
    events = get_dispatch_events(clear=True)

    assert result is not None
    assert result.owned is not None
    assert result.owned.residency is Residency.DEVICE
    assert result.owned.row_count == group_count
    assert any(event.implementation == "native_grouped_disjoint_pack_union" for event in events)
    assert not any(event.implementation == "native_grouped_overlay_union_plan" for event in events)

    actual = np.asarray(result.owned.to_shapely(), dtype=object)
    for group in (0, group_count // 2, group_count - 1):
        start = group * 2
        expected = shapely.union_all(np.asarray(geoms[start : start + 2], dtype=object))
        assert shapely.normalize(actual[group]).equals_exact(
            shapely.normalize(expected),
            tolerance=1.0e-9,
        )


@pytest.mark.gpu
def test_grouped_disjoint_pack_declines_segment_quadratic_admission() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.constructive.binary_constructive import (
        _native_grouped_strict_disjoint_mask_gpu,
    )

    owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), box(2.0, 0.0, 3.0, 1.0), box(4.0, 0.0, 5.0, 1.0)],
        residency=Residency.DEVICE,
    )
    owned._active_family_row_segment_capacity_bound = 29_219

    result = _native_grouped_strict_disjoint_mask_gpu(
        owned,
        cp.arange(3, dtype=cp.int64),
        cp.asarray([0, 3], dtype=cp.int64),
        group_size_max=3,
    )

    assert result is None


@pytest.mark.gpu
def test_execute_native_grouped_union_disjoint_proof_skips_scalar_admissions() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    geoms = [
        box(0.0, 0.0, 1.0, 1.0),
        box(2.0, 0.0, 3.0, 1.0),
        box(10.0, 0.0, 11.0, 1.0),
        box(12.0, 0.0, 13.0, 1.0),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_sorted_offsets(
        cp.asarray([0, 2, 4], dtype=cp.int64),
        row_count=owned.row_count,
        source_token="device-disjoint-proof-test",
        all_groups_observed=True,
        group_size_min=2,
        group_size_max=2,
        strictly_disjoint_group_bounds=True,
    )
    reset_d2h_transfer_count()

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method=DissolveUnionMethod.DISJOINT_SUBSET,
        owned=owned,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result is not None
    assert result.geometries is None
    assert result.owned is not None
    assert "dissolve disjoint-subset admissibility scalar fence" not in reasons
    assert "dissolve disjoint-subset small-pairwise scalar fence" not in reasons
    assert "dissolve disjoint-subset pair-separation scalar fence" not in reasons
    assert reasons == []


@pytest.mark.gpu
def test_execute_grouped_union_codes_device_unary_groups_all_rest_segments() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    geoms = [
        box(0.0, 0.0, 1.0, 1.0),
        box(1.0, 0.0, 2.0, 1.0),
        box(2.0, 0.0, 3.0, 1.0),
        box(10.0, 0.0, 11.0, 1.0),
        box(11.0, 0.0, 12.0, 1.0),
        box(12.0, 0.0, 13.0, 1.0),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    clear_materialization_events()

    grouped = dissolve_module.execute_grouped_union_codes(
        (),
        cp.asarray([0, 0, 0, 1, 1, 1], dtype=cp.int32),
        group_count=2,
        method=DissolveUnionMethod.UNARY,
        owned=owned,
    )

    assert grouped is not None
    assert grouped.geometries is None
    assert grouped.owned is not None
    assert grouped.owned.residency is Residency.DEVICE
    assert grouped.owned.row_count == 2
    assert get_materialization_events(clear=True) == []

    actual = grouped.owned.to_shapely()
    expected = [
        shapely.union_all(np.asarray(geoms[:3], dtype=object)),
        shapely.union_all(np.asarray(geoms[3:], dtype=object)),
    ]
    assert shapely.area(shapely.symmetric_difference(actual[0], expected[0])) == 0.0
    assert shapely.area(shapely.symmetric_difference(actual[1], expected[1])) == 0.0


@pytest.mark.gpu
def test_execute_grouped_union_codes_large_regular_grid_disjoint_coverage_uses_named_native_fences() -> (
    None
):
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    cols = 10
    rows = 7
    group_count = 7
    geoms = [
        box(float(col), float(row), float(col + 1), float(row + 1))
        for row in range(rows)
        for col in range(cols)
    ]
    row_ids = np.arange(len(geoms), dtype=np.int32)
    row_group_codes = row_ids % group_count
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    clear_materialization_events()
    reset_d2h_transfer_count()

    grouped = dissolve_module.execute_grouped_union_codes(
        (),
        cp.asarray(row_group_codes, dtype=cp.int32),
        group_count=group_count,
        method=DissolveUnionMethod.COVERAGE,
        owned=owned,
    )
    events = get_d2h_transfer_events(clear=True)

    assert grouped is not None
    assert grouped.geometries is None
    assert grouped.owned is not None
    assert grouped.owned.row_count == group_count
    assert get_materialization_events(clear=True) == []
    assert 1 <= len(events) <= 10
    assert sum(event.bytes_transferred for event in events) <= 10
    assert all(event.reason.startswith("dissolve ") for event in events)
    assert all("scalar fence" in event.reason for event in events)

    actual = grouped.owned.to_shapely()
    expected = [
        shapely.coverage_union_all(
            np.asarray(
                [geom for index, geom in enumerate(geoms) if row_group_codes[index] == group_index],
                dtype=object,
            )
        )
        for group_index in range(group_count)
    ]
    assert all(bool(shapely.equals(got, want)) for got, want in zip(actual, expected, strict=True))


@pytest.mark.gpu
def test_grouped_coverage_edge_union_device_codes_merges_touching_polygons() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    geoms = [
        Polygon([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.0, 0.0)]),
        Polygon([(2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0)]),
        Polygon([(10.0, 0.0), (12.0, 0.0), (10.0, 2.0), (10.0, 0.0)]),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    reset_d2h_transfer_count()

    grouped = dissolve_module.execute_grouped_coverage_edge_union_gpu_owned_codes(
        cp.asarray([0, 0, 1], dtype=cp.int32),
        group_count=2,
        owned=owned,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert grouped is not None
    assert grouped.geometries is None
    assert grouped.owned is not None
    assert grouped.owned.residency is Residency.DEVICE
    assert grouped.owned.row_count == 2
    assert reasons == []

    actual = grouped.owned.to_shapely()
    expected = [
        shapely.coverage_union_all(np.asarray(geoms[:2], dtype=object)),
        geoms[2],
    ]
    assert shapely.equals(actual[0], expected[0])
    assert shapely.equals(actual[1], expected[1])


@pytest.mark.gpu
def test_grouped_coverage_edge_union_device_codes_marks_unobserved_groups_empty() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    geoms = [
        Polygon([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.0, 0.0)]),
        Polygon([(2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0)]),
        Polygon([(10.0, 0.0), (12.0, 0.0), (10.0, 2.0), (10.0, 0.0)]),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    reset_d2h_transfer_count()

    grouped = dissolve_module.execute_grouped_coverage_edge_union_gpu_owned_codes(
        cp.asarray([0, 0, -1], dtype=cp.int32),
        group_count=3,
        owned=owned,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert grouped is not None
    assert grouped.geometries is None
    assert grouped.owned is not None
    assert grouped.non_empty_groups == 1
    assert grouped.empty_groups == 2
    assert grouped.owned.row_count == 3
    assert "overlay dissolve grouped coverage-edge valid-row count fence" not in reasons
    assert "grouped polygon coverage-union validity-count fence" not in reasons
    assert "grouped polygon coverage-union valid-group count fence" not in reasons
    assert "grouped polygon coverage-union segment-count fence" not in reasons
    assert "owned geometry device-take nested slice-size allocation fence" not in reasons

    actual = np.asarray(grouped.owned.to_shapely(), dtype=object)
    assert shapely.equals(
        actual[0], shapely.coverage_union_all(np.asarray(geoms[:2], dtype=object))
    )
    assert actual[1] is not None and actual[1].is_empty
    assert actual[2] is not None and actual[2].is_empty
    assert shapely.equals(
        actual[1:], np.asarray([GeometryCollection(), GeometryCollection()], dtype=object)
    ).tolist() == [
        True,
        True,
    ]


@pytest.mark.gpu
def test_grouped_coverage_edge_union_host_codes_reuse_group_sizing_mirrors() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    geoms = [
        Polygon([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.0, 0.0)]),
        Polygon([(2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0)]),
        Polygon([(10.0, 0.0), (12.0, 0.0), (10.0, 2.0), (10.0, 0.0)]),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    reset_d2h_transfer_count()

    grouped = dissolve_module.execute_grouped_coverage_edge_union_gpu_owned_codes(
        np.asarray([0, 0, 1], dtype=np.int32),
        group_count=2,
        owned=owned,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert grouped is not None
    assert grouped.geometries is None
    assert grouped.owned is not None
    assert grouped.non_empty_groups == 2
    assert grouped.empty_groups == 0
    assert "overlay dissolve grouped coverage-edge valid-row count fence" not in reasons
    assert "overlay dissolve grouped coverage-edge nonempty-group count fence" not in reasons


@pytest.mark.gpu
@pytest.mark.parametrize("group_count", [10, 11])
def test_large_regular_grid_disjoint_coverage_declines_touching_same_group_cells(
    group_count: int,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    cols = 10
    rows = 11
    geoms = [
        box(float(col), float(row), float(col + 1), float(row + 1))
        for row in range(rows)
        for col in range(cols)
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    row_group_codes = np.arange(len(geoms), dtype=np.int32) % group_count

    grouped = dissolve_module.execute_grouped_disjoint_subset_union_gpu_owned_codes(
        cp.asarray(row_group_codes, dtype=cp.int32),
        group_count=group_count,
        method=DissolveUnionMethod.COVERAGE,
        owned=owned,
    )

    assert grouped is None


@pytest.mark.gpu
@pytest.mark.parametrize(
    "geoms",
    [
        [box(0.0, 0.0, 1.0, 1.0), box(1.0, 0.0, 2.0, 1.0)],
        [box(0.0, 0.0, 2.0, 2.0), box(1.0, 1.0, 3.0, 3.0)],
    ],
)
def test_grouped_disjoint_subset_union_declines_touching_or_overlapping_groups(
    geoms,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = dissolve_module.execute_grouped_disjoint_subset_union_gpu_owned_codes(
        cp.asarray([0, 0], dtype=cp.int32),
        group_count=1,
        method=DissolveUnionMethod.COVERAGE,
        owned=owned,
    )

    assert grouped is None


@pytest.mark.gpu
def test_grouped_disjoint_subset_legacy_gpu_seeds_complex_exact_union_cache() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    nybb_path = Path(__file__).parent / "upstream" / "geopandas" / "tests" / "data" / "nybb_16a.zip"
    if not nybb_path.is_file():
        pytest.skip("NYBB dataset not found")

    frame = geopandas.read_file(f"zip://{nybb_path}")[["geometry", "BoroName", "BoroCode"]]
    frame = frame.rename(columns={"geometry": "myshapes"}).set_geometry("myshapes")
    owned = from_shapely_geometries(list(frame.geometry), residency=Residency.DEVICE)
    group_positions = [
        np.asarray([0, 1, 2], dtype=np.intp),
        np.asarray([3, 4], dtype=np.intp),
    ]

    grouped = dissolve_module.execute_grouped_disjoint_subset_union_gpu(
        None,
        group_positions,
        owned=owned,
    )

    assert grouped is not None
    assert grouped.owned is not None
    cache = getattr(grouped.owned, "_cached_is_valid_mask", None)
    assert cache is not None
    assert cache.tolist() == [True, True]


@pytest.mark.gpu
def test_disjoint_subset_union_nodes_partial_shared_boundary_before_area_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import vibespatial.constructive.union_all as union_all_module

    geoms = [
        box(0.0, 0.0, 2.0, 1.0),
        box(2.0, 0.25, 3.0, 0.75),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)

    def _fail_exact(*_args, **_kwargs):
        raise AssertionError("interior-disjoint coverage must not reopen full union")

    monkeypatch.setattr(union_all_module, "union_all_gpu_owned", _fail_exact)
    result = union_all_module.disjoint_subset_union_all_owned(owned)

    assert result is not None
    assert getattr(result, "_native_grouped_union_implementation", None) == (
        "gpu_noded_coverage_union_with_area_proof"
    )
    actual = result.to_shapely()[0]
    expected = shapely.union_all(np.asarray(geoms, dtype=object))
    assert shapely.is_valid(actual)
    assert shapely.equals(actual, expected)


@pytest.mark.gpu
def test_disjoint_subset_union_area_proof_rejects_positive_area_overlap() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import vibespatial.constructive.union_all as union_all_module

    geoms = [box(0.0, 0.0, 2.0, 2.0), box(1.0, 1.0, 3.0, 3.0)]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)

    coverage = union_all_module._try_polygon_coverage_union_for_bbox_interactions(
        owned,
        dispatch_mode=ExecutionMode.GPU,
    )
    result = union_all_module.disjoint_subset_union_all_owned(owned)

    assert coverage is None
    assert result is not None
    actual = result.to_shapely()[0]
    expected = shapely.union_all(np.asarray(geoms, dtype=object))
    assert shapely.is_valid(actual)
    assert shapely.equals(actual, expected)


def test_evaluate_geopandas_dissolve_matches_current_categorical_semantics() -> None:
    frame = geopandas.GeoDataFrame(
        {
            "cat": pd.Categorical(["a", "a", "b", "b"]),
            "noncat": [1, 1, 1, 2],
            "to_agg": [1, 2, 3, 4],
            "geometry": geopandas.array.from_wkt(
                ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)"]
            ),
        }
    )

    result = evaluate_geopandas_dissolve(
        frame,
        by=["cat", "noncat"],
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    expected = frame.copy().dissolve(["cat", "noncat"])

    assert_geodataframe_equal(result, expected)


def test_evaluate_geopandas_dissolve_can_use_dense_group_codes_without_group_positions(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(["b", "a", "b", "a"], categories=["a", "b", "c"]),
            "value": [1, 2, 3, 4],
            "geometry": geopandas.array.from_wkt(
                [
                    "POLYGON ((0 0, 1 0, 0 1, 0 0))",
                    "POLYGON ((1 0, 1 1, 0 1, 1 0))",
                    "POLYGON ((10 10, 11 10, 10 11, 10 10))",
                    "POLYGON ((11 10, 11 11, 10 11, 11 10))",
                ]
            ),
        }
    )

    expected = frame.dissolve(by="group", aggfunc="first", sort=False, method="coverage")

    calls = 0
    real_codes = dissolve_module.execute_grouped_union_codes

    def _count_codes(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_codes(*args, **kwargs)

    def _fail_positions(*args, **kwargs):
        raise AssertionError("dense group code path should avoid _normalize_group_positions")

    monkeypatch.setattr(dissolve_module, "execute_grouped_union_codes", _count_codes)
    monkeypatch.setattr(dissolve_module, "_normalize_group_positions", _fail_positions)

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=False,
        observed=False,
        dropna=True,
        method="coverage",
        grid_size=None,
        agg_kwargs={},
    )

    assert calls == 1
    assert_geodataframe_equal(result, expected)


def test_evaluate_geopandas_dissolve_uses_native_grouped_numeric_reducers(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": [2, 1, 2, 1],
            "value": [1, 2, 3, 4],
            "flag": [True, False, True, True],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
        }
    )
    expected = frame.dissolve(
        by="group",
        aggfunc={"value": "sum", "flag": "sum"},
        sort=True,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((tuple(columns), dict(reducers)))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("admitted native numeric dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    geopandas.clear_dispatch_events()
    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc={"value": "sum", "flag": "sum"},
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    events = geopandas.get_dispatch_events(clear=True)

    assert reducer_calls == [(("value", "flag"), {"value": "sum", "flag": "sum"})]
    assert any(event.implementation == "native_grouped_numeric_reducers" for event in events)
    assert_geodataframe_equal(result, expected)


def test_evaluate_geopandas_dissolve_uses_native_categorical_numeric_reducers(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(["b", "a", "b", "a"], categories=["a", "b", "c"]),
            "value": [1, 2, 3, 4],
            "weight": [1.5, 2.0, 2.5, 4.0],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
        }
    )
    expected = frame.dissolve(
        by="group",
        aggfunc={"value": "sum", "weight": "mean"},
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append(
            (
                tuple(columns),
                dict(reducers),
                self.output_index_plan.index.copy(),
                self.group_codes.copy(),
            )
        )
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("admitted categorical dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc={"value": "sum", "weight": "mean"},
        as_index=True,
        level=None,
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    reduced_columns, reducers, output_index, group_codes = reducer_calls[0]
    assert reduced_columns == ("value", "weight")
    assert reducers == {"value": "sum", "weight": "mean"}
    assert output_index.tolist() == ["b", "a", "c"]
    assert group_codes.tolist() == [0, 1, 0, 1]
    assert_geodataframe_equal(result, expected)


def test_reduce_native_grouped_dissolve_attributes_uses_device_non_numeric_take_reducers() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device grouped attribute reducers")

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
        reduced_result = dissolve_module._reduce_native_grouped_dissolve_attributes(
            attributes,
            grouped,
            {"name": "first", "when": "last"},
        )

    assert reduced_result is not None
    reduced, used_take_reducer = reduced_result
    assert used_take_reducer is True
    assert isinstance(reduced, NativeAttributeTable)
    assert reduced.device_table is not None
    assert get_materialization_events(clear=True) == []

    exported = reduced.to_pandas()
    assert exported["name"].tolist() == ["bravo", "alpha"]
    assert exported["when"].tolist() == [
        pd.Timestamp("2020-01-04"),
        pd.Timestamp("2020-01-03"),
    ]
    reset_d2h_transfer_count()


def test_evaluate_geopandas_dissolve_uses_native_grouped_min_max_reducers(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(["b", "a", "b", "a"], categories=["a", "b", "c"]),
            "value": [1, 2, 3, 4],
            "flag": [True, False, True, True],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
        }
    )
    expected = frame.dissolve(
        by="group",
        aggfunc={"value": "min", "flag": "max"},
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((tuple(columns), dict(reducers), self.output_index_plan.index.copy()))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("admitted categorical min/max dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc={"value": "min", "flag": "max"},
        as_index=True,
        level=None,
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    reduced_columns, reducers, output_index = reducer_calls[0]
    assert reduced_columns == ("value", "flag")
    assert reducers == {"value": "min", "flag": "max"}
    assert output_index.tolist() == ["b", "a", "c"]
    assert result.index.tolist() == ["b", "a", "c"]
    assert result["value"].tolist()[:2] == [1.0, 2.0]
    assert result["flag"].tolist()[:2] == [1.0, 1.0]
    assert np.isnan(result["value"].iloc[2])
    assert np.isnan(result["flag"].iloc[2])
    assert_geodataframe_equal(result, expected)


def test_evaluate_geopandas_dissolve_default_first_uses_native_grouped_reducer(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(["b", "a", "b", "a"], categories=["a", "b", "c"]),
            "value": [1, 2, 3, 4],
            "flag": [True, False, True, True],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
        }
    )
    expected = frame.dissolve(
        by="group",
        aggfunc="first",
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((tuple(columns), dict(reducers), self.output_index_plan.index.copy()))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("admitted default-first dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    reduced_columns, reducers, output_index = reducer_calls[0]
    assert reduced_columns == ("value", "flag")
    assert reducers == {"value": "first", "flag": "first"}
    assert output_index.tolist() == ["b", "a", "c"]
    assert result["value"].tolist()[:2] == [1.0, 2.0]
    assert result["flag"].tolist()[:2] == [1.0, 0.0]
    assert np.isnan(result["value"].iloc[2])
    assert np.isnan(result["flag"].iloc[2])
    assert_geodataframe_equal(result, expected)


def test_evaluate_geopandas_dissolve_default_first_uses_native_take_reducers(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(["b", "a", "b", "a"], categories=["a", "b", "c"]),
            "value": [1, 2, 3, 4],
            "label": pd.Series(
                [None, "right", "last-left", "last-right"],
                dtype=object,
            ),
            "category": pd.Categorical(
                [None, "y", "z", "w"],
                categories=["w", "x", "y", "z"],
            ),
            "text": pd.array([pd.NA, "bb", "cc", "dd"], dtype="string"),
            "nullable": pd.array([pd.NA, 2, 3, pd.NA], dtype="Int64"),
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
        }
    )
    expected = frame.dissolve(
        by="group",
        aggfunc="first",
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
    )

    real_take = NativeGrouped.reduce_take
    take_calls = []

    def _record_take(self, values, reducer):
        take_calls.append((getattr(values, "name", None), reducer))
        return real_take(self, values, reducer)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("admitted label-column dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_take", _record_take)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert take_calls == [
        ("label", "first"),
        ("category", "first"),
        ("text", "first"),
        ("nullable", "first"),
    ]
    assert result["label"].tolist() == ["last-left", "right", None]
    assert result["category"].tolist() == ["z", "y", np.nan]
    assert isinstance(result["category"].dtype, pd.CategoricalDtype)
    assert result["text"].tolist()[:2] == ["cc", "bb"]
    assert pd.isna(result["text"].iloc[2])
    assert result["nullable"].tolist()[:2] == [3, 2]
    assert pd.isna(result["nullable"].iloc[2])
    assert_geodataframe_equal(result, expected)


def test_evaluate_geopandas_dissolve_categorical_observed_true_uses_observed_codes(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(["b", None, "a", "b"], categories=["a", "b", "c"]),
            "value": [1, 2, 3, 4],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
        }
    )
    expected = frame.dissolve(
        by="group",
        aggfunc={"value": "count"},
        sort=True,
        observed=True,
        dropna=True,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes.copy()))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("observed categorical dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc={"value": "count"},
        as_index=True,
        level=None,
        sort=True,
        observed=True,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    assert output_index.tolist() == ["a", "b"]
    assert group_codes.tolist() == [1, -1, 0, 1]
    assert_geodataframe_equal(result, expected)


@pytest.mark.parametrize(
    ("observed", "sort", "expected_index", "expected_values", "expected_codes"),
    [
        (False, False, ["b", np.nan, "a", "c"], [5, 2, 3, 0], [0, 1, 2, 0]),
        (False, True, ["a", "b", "c", np.nan], [3, 5, 0, 2], [1, 3, 0, 1]),
        (True, False, ["b", np.nan, "a"], [5, 2, 3], [0, 1, 2, 0]),
        (True, True, ["a", "b", np.nan], [3, 5, 2], [1, 2, 0, 1]),
    ],
)
def test_evaluate_geopandas_dissolve_categorical_dropna_false_uses_native_null_group(
    monkeypatch,
    observed: bool,
    sort: bool,
    expected_index: list[object],
    expected_values: list[int],
    expected_codes: list[int],
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(["b", None, "a", "b"], categories=["a", "b", "c"]),
            "value": [1, 2, 3, 4],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
        }
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes.copy()))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("categorical null-group dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=sort,
        observed=observed,
        dropna=False,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    assert output_index.tolist() == expected_index
    assert group_codes.tolist() == expected_codes
    assert result.index.tolist() == expected_index
    assert result["value"].tolist() == expected_values


@pytest.mark.parametrize(
    ("group_key", "sort", "dropna", "expected_codes"),
    [
        (
            pd.Series(["b", None, "a", "b"], dtype=object),
            False,
            False,
            [0, 1, 2, 0],
        ),
        (
            pd.Series(pd.array(["b", pd.NA, "a", "b"], dtype="string")),
            True,
            False,
            [1, 2, 0, 1],
        ),
        (
            pd.Series([2.0, np.nan, 1.0, 2.0], dtype="float64"),
            False,
            False,
            [0, 1, 2, 0],
        ),
        (
            pd.Series(pd.array([2, pd.NA, 1, 2], dtype="Int64")),
            True,
            False,
            [1, 2, 0, 1],
        ),
        (
            pd.Series(["b", None, "a", "b"], dtype=object),
            True,
            True,
            [1, -1, 0, 1],
        ),
    ],
)
def test_evaluate_geopandas_dissolve_plain_nullable_keys_use_native_group_codes(
    monkeypatch,
    group_key: pd.Series,
    sort: bool,
    dropna: bool,
    expected_codes: list[int],
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": group_key.reset_index(drop=True),
            "value": [1, 2, 3, 4],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
        }
    )
    expected = frame.dissolve(
        by="group",
        aggfunc={"value": "sum"},
        sort=sort,
        dropna=dropna,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes.copy()))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("plain nullable dissolve keys should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=sort,
        observed=False,
        dropna=dropna,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    pd.testing.assert_index_equal(output_index, expected.index)
    assert group_codes.tolist() == expected_codes
    assert_geodataframe_equal(result, expected)


@pytest.mark.parametrize(
    ("sort", "dropna", "expected_codes"),
    [
        (False, False, [0, 1, 2, 0, 3]),
        (True, False, [2, 3, 0, 2, 1]),
        (True, True, [1, -1, 0, 1, -1]),
    ],
)
def test_evaluate_geopandas_dissolve_multi_nullable_keys_use_native_group_codes(
    monkeypatch,
    sort: bool,
    dropna: bool,
    expected_codes: list[int],
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Series(["b", pd.NA, "a", "b", "a"], dtype="string"),
            "bucket": pd.Series([2, 1, 1, 2, pd.NA], dtype="Int64"),
            "value": [1, 2, 3, 4, 5],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                    "POINT (4 4)",
                ]
            ),
        }
    )
    expected = frame.dissolve(
        by=["group", "bucket"],
        aggfunc={"value": "sum"},
        sort=sort,
        dropna=dropna,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append(
            (
                tuple(columns),
                self.output_index_plan.index.copy(),
                self.group_codes.copy(),
            )
        )
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("multi-key dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by=["group", "bucket"],
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=sort,
        observed=False,
        dropna=dropna,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    reduced_columns, output_index, group_codes = reducer_calls[0]
    assert reduced_columns == ("value",)
    pd.testing.assert_index_equal(output_index, expected.index)
    assert group_codes.tolist() == expected_codes
    assert_geodataframe_equal(result, expected)


@pytest.mark.parametrize(
    ("sort", "dropna", "expected_codes"),
    [
        (False, False, [0, 1, 2, 0, 2]),
        (True, False, [1, 2, 0, 1, 0]),
        (True, True, [1, -1, 0, 1, 0]),
    ],
)
def test_evaluate_geopandas_dissolve_object_string_multi_keys_use_native_group_codes(
    monkeypatch,
    sort: bool,
    dropna: bool,
    expected_codes: list[int],
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Series(["b", None, "a", "b", "a"], dtype=object),
            "bucket": pd.Series(["x", None, "y", "x", "y"], dtype=object),
            "value": [1, 2, 3, 4, 5],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                    "POINT (4 4)",
                ]
            ),
        }
    )
    expected = frame.dissolve(
        by=["group", "bucket"],
        aggfunc={"value": "sum"},
        sort=sort,
        dropna=dropna,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes.copy()))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("object-string multi-key dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by=["group", "bucket"],
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=sort,
        observed=False,
        dropna=dropna,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    pd.testing.assert_index_equal(output_index, expected.index)
    assert group_codes.tolist() == expected_codes
    assert_geodataframe_equal(result, expected)


@pytest.mark.parametrize(
    ("sort", "dropna", "expected_codes"),
    [
        (False, False, [0, 1, 2, 0, 2]),
        (False, True, [0, -1, 1, 0, 1]),
        (True, True, [1, -1, 0, 1, 0]),
    ],
)
def test_evaluate_geopandas_dissolve_object_numeric_multi_keys_use_native_group_codes(
    monkeypatch,
    sort: bool,
    dropna: bool,
    expected_codes: list[int],
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Series([2, None, 1, 2, 1], dtype=object),
            "bucket": pd.Series([10, None, 20, 10, 20], dtype=object),
            "value": [1, 2, 3, 4, 5],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                    "POINT (4 4)",
                ]
            ),
        }
    )
    expected = frame.dissolve(
        by=["group", "bucket"],
        aggfunc={"value": "sum"},
        sort=sort,
        dropna=dropna,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes.copy()))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("object-numeric multi-key dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by=["group", "bucket"],
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=sort,
        observed=False,
        dropna=dropna,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    pd.testing.assert_index_equal(output_index, expected.index)
    assert group_codes.tolist() == expected_codes
    assert_geodataframe_equal(result, expected)


def test_evaluate_geopandas_dissolve_mixed_object_multi_keys_use_native_group_codes(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Series([2, None, "a", 2, "a"], dtype=object),
            "bucket": pd.Series(["x", None, "y", "x", "y"], dtype=object),
            "value": [1, 2, 3, 4, 5],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                    "POINT (4 4)",
                ]
            ),
        }
    )
    expected = frame.dissolve(
        by=["group", "bucket"],
        aggfunc={"value": "sum"},
        sort=False,
        dropna=True,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes.copy()))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("mixed object multi-key dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by=["group", "bucket"],
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    pd.testing.assert_index_equal(output_index, expected.index)
    assert group_codes.tolist() == [0, -1, 1, 0, 1]
    assert_geodataframe_equal(result, expected)


def test_evaluate_geopandas_dissolve_custom_object_key_uses_pandas_policy(
    monkeypatch,
) -> None:
    class CustomKey:
        def __init__(self, value: int) -> None:
            self.value = value

        def __eq__(self, other) -> bool:
            return isinstance(other, CustomKey) and self.value == other.value

        def __hash__(self) -> int:
            return hash(self.value)

        def __repr__(self) -> str:
            return f"CustomKey({self.value})"

    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Series(
                [CustomKey(1), CustomKey(2), CustomKey(1)],
                dtype=object,
            ),
            "value": [1, 2, 3],
            "geometry": geopandas.array.from_wkt(["POINT (0 0)", "POINT (1 1)", "POINT (2 2)"]),
        }
    )
    expected = frame.dissolve(
        by="group",
        aggfunc={"value": "sum"},
        sort=False,
        method="unary",
    )

    groupby_calls = 0
    real_groupby = pd.DataFrame.groupby

    def _record_groupby(self, *args, **kwargs):
        nonlocal groupby_calls
        groupby_calls += 1
        return real_groupby(self, *args, **kwargs)

    def _fail_reduce(*_args, **_kwargs):
        raise AssertionError("custom object keys are not a native grouping contract")

    monkeypatch.setattr(pd.DataFrame, "groupby", _record_groupby)
    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _fail_reduce)

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert groupby_calls >= 1
    assert_geodataframe_equal(result, expected)


def test_evaluate_geopandas_dissolve_unhashable_object_key_uses_pandas_policy(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Series([[1], [1]], dtype=object),
            "value": [1, 2],
            "geometry": geopandas.array.from_wkt(["POINT (0 0)", "POINT (1 1)"]),
        }
    )

    def _fail_reduce(*_args, **_kwargs):
        raise AssertionError("unhashable object keys are not a native grouping contract")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _fail_reduce)

    with pytest.raises(TypeError, match="unhashable"):
        evaluate_geopandas_dissolve(
            frame,
            by="group",
            aggfunc={"value": "sum"},
            as_index=True,
            level=None,
            sort=False,
            observed=False,
            dropna=True,
            method="unary",
            grid_size=None,
            agg_kwargs={},
        )


@pytest.mark.gpu
def test_evaluate_geopandas_dissolve_device_integer_key_uses_device_group_codes(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device key encoding")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")

    geoms = [
        box(0, 1, 1, 2),
        box(0, 0, 1, 1),
        box(1, 1, 2, 2),
        box(1, 0, 2, 1),
    ]
    reference = geopandas.GeoDataFrame(
        {
            "group": [1, 0, 1, 0],
            "value": [10, 1, 20, 2],
            "geometry": geoms,
        }
    )
    expected = reference.dissolve(
        by="group",
        aggfunc={"value": "sum"},
        sort=True,
        method="coverage",
    )
    frame = reference.copy(deep=True)
    attribute_arrow = pa.table(
        {
            "group": pa.array([1, 0, 1, 0], type=pa.int64()),
            "value": pa.array([10, 1, 20, 2], type=pa.int64()),
        }
    )
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(attribute_arrow),
        index_override=frame.index,
        column_override=tuple(attribute_arrow.column_names),
        schema_override=attribute_arrow.schema,
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=attributes,
            geometry=GeometryNativeResult.from_owned(owned, crs=frame.crs),
            geometry_name="geometry",
            column_order=tuple(frame.columns),
        )
    )
    attach_native_state(frame, state)

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("device-key dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)
    clear_materialization_events()

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="coverage",
        grid_size=None,
        agg_kwargs={},
    )
    events = get_materialization_events(clear=True)

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    pd.testing.assert_index_equal(output_index, expected.index)
    assert hasattr(group_codes, "__cuda_array_interface__")
    assert cp.asnumpy(group_codes).tolist() == [1, 0, 1, 0]
    assert any(
        event.operation == "device_group_key_labels_to_host"
        and event.detail == "groups=2, bytes=16"
        and event.strict_disallowed is False
        for event in events
    )
    assert result.index.equals(expected.index)
    assert result["value"].tolist() == expected["value"].tolist()
    assert bool(
        np.all(
            shapely.equals(
                np.asarray(result.geometry, dtype=object),
                np.asarray(expected.geometry, dtype=object),
            )
        )
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("dropna", "expected_codes", "expected_event_detail"),
    [
        (True, [1, -1, 0, 1], "groups=2, bytes=16"),
        (False, [1, 2, 0, 1], "groups=2, bytes=16"),
    ],
)
def test_evaluate_geopandas_dissolve_device_nullable_integer_key_uses_device_group_codes(
    monkeypatch,
    dropna: bool,
    expected_codes: list[int],
    expected_event_detail: str,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for nullable device key encoding")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")

    geoms = [
        box(0, 1, 1, 2),
        box(2, 0, 3, 1),
        box(0, 0, 1, 1),
        box(1, 1, 2, 2),
    ]
    reference = geopandas.GeoDataFrame(
        {
            "group": pd.Series(pd.array([1, pd.NA, 0, 1], dtype="Int64")),
            "value": [10, 3, 1, 20],
            "geometry": geoms,
        }
    )
    expected = reference.dissolve(
        by="group",
        aggfunc={"value": "sum"},
        sort=True,
        dropna=dropna,
        method="coverage",
    )
    frame = reference.copy(deep=True)
    attribute_arrow = pa.table(
        {
            "group": pa.array([1, None, 0, 1], type=pa.int64()),
            "value": pa.array([10, 3, 1, 20], type=pa.int64()),
        }
    )
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(attribute_arrow),
        index_override=frame.index,
        column_override=tuple(attribute_arrow.column_names),
        schema_override=attribute_arrow.schema,
    )
    attach_native_state(
        frame,
        NativeFrameState.from_native_tabular_result(
            NativeTabularResult(
                attributes=attributes,
                geometry=GeometryNativeResult.from_owned(owned, crs=frame.crs),
                geometry_name="geometry",
                column_order=tuple(frame.columns),
            )
        ),
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("nullable device-key dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)
    clear_materialization_events()

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=dropna,
        method="coverage",
        grid_size=None,
        agg_kwargs={},
    )
    events = get_materialization_events(clear=True)

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    pd.testing.assert_index_equal(output_index, expected.index)
    assert hasattr(group_codes, "__cuda_array_interface__")
    assert cp.asnumpy(group_codes).tolist() == expected_codes
    assert any(
        event.operation == "device_group_key_labels_to_host"
        and event.detail == expected_event_detail
        and event.strict_disallowed is False
        for event in events
    )
    assert result.index.equals(expected.index)
    assert result["value"].tolist() == expected["value"].tolist()
    assert bool(
        np.all(
            shapely.equals(
                np.asarray(result.geometry, dtype=object),
                np.asarray(expected.geometry, dtype=object),
            )
        )
    )


@pytest.mark.gpu
def test_evaluate_geopandas_dissolve_device_nullable_boolean_key_uses_device_group_codes(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for nullable device bool key encoding")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")

    geoms = [
        box(0, 1, 1, 2),
        box(2, 0, 3, 1),
        box(0, 0, 1, 1),
        box(1, 1, 2, 2),
    ]
    reference = geopandas.GeoDataFrame(
        {
            "group": pd.Series(pd.array([True, pd.NA, False, True], dtype="boolean")),
            "value": [10, 3, 1, 20],
            "geometry": geoms,
        }
    )
    expected = reference.dissolve(
        by="group",
        aggfunc={"value": "sum"},
        sort=True,
        dropna=False,
        method="coverage",
    )
    frame = reference.copy(deep=True)
    attribute_arrow = pa.table(
        {
            "group": pa.array([True, None, False, True], type=pa.bool_()),
            "value": pa.array([10, 3, 1, 20], type=pa.int64()),
        }
    )
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(attribute_arrow),
        index_override=frame.index,
        column_override=tuple(attribute_arrow.column_names),
        schema_override=attribute_arrow.schema,
    )
    attach_native_state(
        frame,
        NativeFrameState.from_native_tabular_result(
            NativeTabularResult(
                attributes=attributes,
                geometry=GeometryNativeResult.from_owned(owned, crs=frame.crs),
                geometry_name="geometry",
                column_order=tuple(frame.columns),
            )
        ),
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("nullable bool device-key dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)
    clear_materialization_events()

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=False,
        method="coverage",
        grid_size=None,
        agg_kwargs={},
    )
    events = get_materialization_events(clear=True)

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    pd.testing.assert_index_equal(output_index, expected.index)
    assert hasattr(group_codes, "__cuda_array_interface__")
    assert cp.asnumpy(group_codes).tolist() == [1, 2, 0, 1]
    assert any(
        event.operation == "device_group_key_labels_to_host"
        and event.detail == "groups=2, bytes=2"
        and event.strict_disallowed is False
        for event in events
    )
    assert result.index.equals(expected.index)
    assert result["value"].tolist() == expected["value"].tolist()
    assert bool(
        np.all(
            shapely.equals(
                np.asarray(result.geometry, dtype=object),
                np.asarray(expected.geometry, dtype=object),
            )
        )
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("observed", "dropna", "expected_codes"),
    [
        (False, True, [1, -1, 0, 1]),
        (False, False, [1, 3, 0, 1]),
        (True, True, [1, -1, 0, 1]),
        (True, False, [1, 2, 0, 1]),
    ],
)
def test_evaluate_geopandas_dissolve_device_categorical_key_uses_device_group_codes(
    monkeypatch,
    observed: bool,
    dropna: bool,
    expected_codes: list[int],
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for categorical device key encoding")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")

    geoms = [
        box(0, 1, 1, 2),
        box(2, 0, 3, 1),
        box(0, 0, 1, 1),
        box(1, 1, 2, 2),
    ]
    reference = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(
                ["b", None, "a", "b"],
                categories=["a", "b", "c"],
            ),
            "value": [10, 3, 1, 20],
            "geometry": geoms,
        }
    )
    expected = reference.dissolve(
        by="group",
        aggfunc={"value": "sum"},
        sort=True,
        observed=observed,
        dropna=dropna,
        method="coverage",
    )
    frame = reference.copy(deep=True)
    attribute_arrow = pa.table(
        {
            "group": pa.DictionaryArray.from_arrays(
                pa.array([1, None, 0, 1], type=pa.int8()),
                pa.array(["a", "b", "c"]),
            ),
            "value": pa.array([10, 3, 1, 20], type=pa.int64()),
        }
    )
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(attribute_arrow),
        index_override=frame.index,
        column_override=tuple(attribute_arrow.column_names),
        schema_override=attribute_arrow.schema,
    )
    attach_native_state(
        frame,
        NativeFrameState.from_native_tabular_result(
            NativeTabularResult(
                attributes=attributes,
                geometry=GeometryNativeResult.from_owned(owned, crs=frame.crs),
                geometry_name="geometry",
                column_order=tuple(frame.columns),
            )
        ),
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("categorical device-key dissolve should not call pandas groupby")

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)
    clear_materialization_events()
    reset_d2h_transfer_count()

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=True,
        observed=observed,
        dropna=dropna,
        method="coverage",
        grid_size=None,
        agg_kwargs={},
    )
    events = get_materialization_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    pd.testing.assert_index_equal(output_index, expected.index)
    assert hasattr(group_codes, "__cuda_array_interface__")
    assert cp.asnumpy(group_codes).tolist() == expected_codes
    assert "owned geometry device-take nested slice-size allocation fence" not in runtime_reasons
    if observed:
        assert any(
            event.operation == "device_categorical_group_key_codes_to_host"
            and event.detail == "groups=2, bytes=8"
            and event.strict_disallowed is False
            for event in events
        )
    assert result.index.equals(expected.index)
    assert result["value"].tolist() == expected["value"].tolist()
    assert bool(
        np.all(
            shapely.equals(
                np.asarray(result.geometry, dtype=object),
                np.asarray(expected.geometry, dtype=object),
            )
        )
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("observed", "dropna", "expected_codes"),
    [
        (False, True, [3, -1, 0, 3]),
        (False, False, [3, 6, 0, 3]),
        (True, True, [1, -1, 0, 1]),
        (True, False, [1, 2, 0, 1]),
    ],
)
def test_evaluate_geopandas_dissolve_device_categorical_multi_key_uses_device_group_codes(
    monkeypatch,
    observed: bool,
    dropna: bool,
    expected_codes: list[int],
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for categorical device multi-key encoding")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")

    geoms = [
        box(0, 1, 1, 2),
        box(2, 0, 3, 1),
        box(0, 0, 1, 1),
        box(1, 1, 2, 2),
    ]
    reference = geopandas.GeoDataFrame(
        {
            "cat": pd.Categorical(
                ["b", None, "a", "b"],
                categories=["a", "b", "c"],
            ),
            "zone": [2, 1, 1, 2],
            "value": [10, 3, 1, 20],
            "geometry": geoms,
        }
    )
    expected = reference.dissolve(
        by=["cat", "zone"],
        aggfunc={"value": "sum"},
        sort=True,
        observed=observed,
        dropna=dropna,
        method="coverage",
    )
    frame = reference.copy(deep=True)
    attribute_arrow = pa.table(
        {
            "cat": pa.DictionaryArray.from_arrays(
                pa.array([1, None, 0, 1], type=pa.int8()),
                pa.array(["a", "b", "c"]),
            ),
            "zone": pa.array([2, 1, 1, 2], type=pa.int64()),
            "value": pa.array([10, 3, 1, 20], type=pa.int64()),
        }
    )
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(attribute_arrow),
        index_override=frame.index,
        column_override=tuple(attribute_arrow.column_names),
        schema_override=attribute_arrow.schema,
    )
    attach_native_state(
        frame,
        NativeFrameState.from_native_tabular_result(
            NativeTabularResult(
                attributes=attributes,
                geometry=GeometryNativeResult.from_owned(owned, crs=frame.crs),
                geometry_name="geometry",
                column_order=tuple(frame.columns),
            )
        ),
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("categorical device multi-key dissolve should not call pandas groupby")

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)
    reset_d2h_transfer_count()

    result = evaluate_geopandas_dissolve(
        frame,
        by=["cat", "zone"],
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=True,
        observed=observed,
        dropna=dropna,
        method="coverage",
        grid_size=None,
        agg_kwargs={},
    )
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    pd.testing.assert_index_equal(output_index, expected.index)
    assert hasattr(group_codes, "__cuda_array_interface__")
    assert cp.asnumpy(group_codes).tolist() == expected_codes
    assert "owned geometry device-take nested slice-size allocation fence" not in runtime_reasons
    assert result.index.equals(expected.index)
    assert result["value"].tolist() == expected["value"].tolist()
    assert bool(
        np.all(
            shapely.equals(
                np.asarray(result.geometry, dtype=object),
                np.asarray(expected.geometry, dtype=object),
            )
        )
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("dropna", "expected_codes"),
    [
        (True, [0, -1, -1, 0]),
        (False, [1, 2, 0, 1]),
    ],
)
def test_evaluate_geopandas_dissolve_device_nullable_integer_multi_key_uses_device_group_codes(
    monkeypatch,
    dropna: bool,
    expected_codes: list[int],
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for nullable device multi-key encoding")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")

    geoms = [
        box(0, 1, 1, 2),
        box(2, 0, 3, 1),
        box(0, 0, 1, 1),
        box(1, 1, 2, 2),
    ]
    reference = geopandas.GeoDataFrame(
        {
            "zone": pd.Series(pd.array([1, pd.NA, 0, 1], dtype="Int64")),
            "kind": pd.Series(pd.array([1, 0, pd.NA, 1], dtype="Int64")),
            "value": [10, 3, 1, 20],
            "geometry": geoms,
        }
    )
    expected = reference.dissolve(
        by=["zone", "kind"],
        aggfunc={"value": "sum"},
        sort=True,
        dropna=dropna,
        method="coverage",
    )
    frame = reference.copy(deep=True)
    attribute_arrow = pa.table(
        {
            "zone": pa.array([1, None, 0, 1], type=pa.int64()),
            "kind": pa.array([1, 0, None, 1], type=pa.int64()),
            "value": pa.array([10, 3, 1, 20], type=pa.int64()),
        }
    )
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(attribute_arrow),
        index_override=frame.index,
        column_override=tuple(attribute_arrow.column_names),
        schema_override=attribute_arrow.schema,
    )
    attach_native_state(
        frame,
        NativeFrameState.from_native_tabular_result(
            NativeTabularResult(
                attributes=attributes,
                geometry=GeometryNativeResult.from_owned(owned, crs=frame.crs),
                geometry_name="geometry",
                column_order=tuple(frame.columns),
            )
        ),
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("nullable device multi-key dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)
    clear_materialization_events()

    result = evaluate_geopandas_dissolve(
        frame,
        by=["zone", "kind"],
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=dropna,
        method="coverage",
        grid_size=None,
        agg_kwargs={},
    )
    events = get_materialization_events(clear=True)

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    pd.testing.assert_index_equal(output_index, expected.index)
    assert hasattr(group_codes, "__cuda_array_interface__")
    assert cp.asnumpy(group_codes).tolist() == expected_codes
    assert any(
        event.operation == "device_multi_group_key_labels_to_host"
        and event.strict_disallowed is False
        for event in events
    )
    assert result.index.equals(expected.index)
    assert result["value"].tolist() == expected["value"].tolist()
    assert bool(
        np.all(
            shapely.equals(
                np.asarray(result.geometry, dtype=object),
                np.asarray(expected.geometry, dtype=object),
            )
        )
    )


@pytest.mark.gpu
def test_evaluate_geopandas_dissolve_device_nullable_boolean_multi_key_uses_device_group_codes(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for nullable device bool multi-key encoding")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")

    geoms = [
        box(0, 1, 1, 2),
        box(2, 0, 3, 1),
        box(0, 0, 1, 1),
        box(1, 1, 2, 2),
    ]
    reference = geopandas.GeoDataFrame(
        {
            "zone": pd.Series(pd.array([True, pd.NA, False, True], dtype="boolean")),
            "kind": pd.Series(pd.array([False, True, False, pd.NA], dtype="boolean")),
            "value": [10, 3, 1, 20],
            "geometry": geoms,
        }
    )
    expected = reference.dissolve(
        by=["zone", "kind"],
        aggfunc={"value": "sum"},
        sort=True,
        dropna=False,
        method="coverage",
    )
    frame = reference.copy(deep=True)
    attribute_arrow = pa.table(
        {
            "zone": pa.array([True, None, False, True], type=pa.bool_()),
            "kind": pa.array([False, True, False, None], type=pa.bool_()),
            "value": pa.array([10, 3, 1, 20], type=pa.int64()),
        }
    )
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(attribute_arrow),
        index_override=frame.index,
        column_override=tuple(attribute_arrow.column_names),
        schema_override=attribute_arrow.schema,
    )
    attach_native_state(
        frame,
        NativeFrameState.from_native_tabular_result(
            NativeTabularResult(
                attributes=attributes,
                geometry=GeometryNativeResult.from_owned(owned, crs=frame.crs),
                geometry_name="geometry",
                column_order=tuple(frame.columns),
            )
        ),
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError(
            "nullable bool device multi-key dissolve should not call pandas groupby"
        )

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)
    clear_materialization_events()

    result = evaluate_geopandas_dissolve(
        frame,
        by=["zone", "kind"],
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=False,
        method="coverage",
        grid_size=None,
        agg_kwargs={},
    )
    events = get_materialization_events(clear=True)

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    pd.testing.assert_index_equal(output_index, expected.index)
    assert hasattr(group_codes, "__cuda_array_interface__")
    assert cp.asnumpy(group_codes).tolist() == [1, 3, 0, 2]
    assert any(
        event.operation == "device_multi_group_key_labels_to_host"
        and event.strict_disallowed is False
        for event in events
    )
    assert result.index.equals(expected.index)
    assert result["value"].tolist() == expected["value"].tolist()
    assert bool(
        np.all(
            shapely.equals(
                np.asarray(result.geometry, dtype=object),
                np.asarray(expected.geometry, dtype=object),
            )
        )
    )


@pytest.mark.gpu
def test_evaluate_geopandas_dissolve_device_integer_multi_key_uses_device_group_codes(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device multi-key encoding")
    cp = pytest.importorskip("cupy")
    plc = pytest.importorskip("pylibcudf")

    geoms = [
        box(0, 0, 1, 1),
        box(0, 2, 1, 3),
        box(0, 4, 1, 5),
        box(1, 2, 2, 3),
        box(1, 0, 2, 1),
    ]
    reference = geopandas.GeoDataFrame(
        {
            "zone": [1, 0, 1, 0, 1],
            "kind": [0, 1, 1, 1, 0],
            "value": [10, 1, 20, 2, 30],
            "geometry": geoms,
        }
    )
    expected = reference.dissolve(
        by=["zone", "kind"],
        aggfunc={"value": "sum"},
        sort=True,
        method="coverage",
    )
    frame = reference.copy(deep=True)
    attribute_arrow = pa.table(
        {
            "zone": pa.array([1, 0, 1, 0, 1], type=pa.int64()),
            "kind": pa.array([0, 1, 1, 1, 0], type=pa.int64()),
            "value": pa.array([10, 1, 20, 2, 30], type=pa.int64()),
        }
    )
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    attributes = NativeAttributeTable(
        device_table=plc.Table.from_arrow(attribute_arrow),
        index_override=frame.index,
        column_override=tuple(attribute_arrow.column_names),
        schema_override=attribute_arrow.schema,
    )
    state = NativeFrameState.from_native_tabular_result(
        NativeTabularResult(
            attributes=attributes,
            geometry=GeometryNativeResult.from_owned(owned, crs=frame.crs),
            geometry_name="geometry",
            column_order=tuple(frame.columns),
        )
    )
    attach_native_state(frame, state)

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("device multi-key dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)
    clear_materialization_events()

    result = evaluate_geopandas_dissolve(
        frame,
        by=["zone", "kind"],
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="coverage",
        grid_size=None,
        agg_kwargs={},
    )
    events = get_materialization_events(clear=True)

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    pd.testing.assert_index_equal(output_index, expected.index)
    assert hasattr(group_codes, "__cuda_array_interface__")
    assert cp.asnumpy(group_codes).tolist() == [1, 0, 2, 0, 1]
    assert any(
        event.operation == "device_multi_group_key_labels_to_host"
        and event.detail == "groups=3, columns=2, bytes=48"
        and event.strict_disallowed is False
        for event in events
    )
    assert result.index.equals(expected.index)
    assert result["value"].tolist() == expected["value"].tolist()
    assert bool(
        np.all(
            shapely.equals(
                np.asarray(result.geometry, dtype=object),
                np.asarray(expected.geometry, dtype=object),
            )
        )
    )


@pytest.mark.parametrize(
    ("observed", "sort", "expected_codes"),
    [
        (False, True, [0, 0, 2, 3]),
        (True, True, [0, 0, 1, 2]),
        (False, False, [0, 0, 1, 2]),
    ],
)
def test_evaluate_geopandas_dissolve_categorical_multi_keys_use_native_group_codes(
    monkeypatch,
    observed: bool,
    sort: bool,
    expected_codes: list[int],
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "cat": pd.Categorical(["a", "a", "b", "b"]),
            "noncat": [1, 1, 1, 2],
            "value": [1, 2, 3, 4],
            "geometry": geopandas.array.from_wkt(
                ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)"]
            ),
        }
    )
    expected = frame.dissolve(
        by=["cat", "noncat"],
        aggfunc={"value": "first"},
        sort=sort,
        observed=observed,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes.copy()))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("categorical multi-key dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by=["cat", "noncat"],
        aggfunc={"value": "first"},
        as_index=True,
        level=None,
        sort=sort,
        observed=observed,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    pd.testing.assert_index_equal(output_index, expected.index)
    assert group_codes.tolist() == expected_codes
    assert_geodataframe_equal(result, expected)


@pytest.mark.parametrize(
    ("sort", "expected_codes"),
    [
        (False, [0, 1, 2, 0]),
        (True, [1, 2, 0, 1]),
    ],
)
def test_evaluate_geopandas_dissolve_categorical_null_multi_keys_use_native_group_codes(
    monkeypatch,
    sort: bool,
    expected_codes: list[int],
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "cat": pd.Categorical(["b", None, "a", "b"], categories=["a", "b", "c"]),
            "noncat": [2, 1, 1, 2],
            "value": [1, 2, 3, 4],
            "geometry": geopandas.array.from_wkt(
                ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)"]
            ),
        }
    )
    expected = frame.dissolve(
        by=["cat", "noncat"],
        aggfunc={"value": "sum"},
        sort=sort,
        observed=True,
        dropna=False,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes.copy()))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("categorical-null multi-key dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by=["cat", "noncat"],
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=sort,
        observed=True,
        dropna=False,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    pd.testing.assert_index_equal(output_index, expected.index)
    assert group_codes.tolist() == expected_codes
    assert_geodataframe_equal(result, expected)


@pytest.mark.parametrize(
    ("sort", "expected_codes"),
    [
        (False, [0, 1, 2, 0]),
        (True, [3, 6, 0, 3]),
    ],
)
def test_evaluate_geopandas_dissolve_unobserved_categorical_null_product_uses_native_group_codes(
    monkeypatch,
    sort: bool,
    expected_codes: list[int],
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "cat": pd.Categorical(["b", None, "a", "b"], categories=["a", "b", "c"]),
            "noncat": [2, 1, 1, 2],
            "value": [1, 2, 3, 4],
            "geometry": geopandas.array.from_wkt(
                ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)"]
            ),
        }
    )
    expected = frame.dissolve(
        by=["cat", "noncat"],
        aggfunc={"value": "sum"},
        sort=sort,
        observed=False,
        dropna=False,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes.copy()))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("unobserved categorical-null dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by=["cat", "noncat"],
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=sort,
        observed=False,
        dropna=False,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    pd.testing.assert_index_equal(output_index, expected.index)
    assert group_codes.tolist() == expected_codes
    assert_geodataframe_equal(result, expected)


def test_evaluate_geopandas_dissolve_native_as_index_false_defers_attribute_export(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": ["b", "a", "b", "a"],
            "value": [1, 2, 3, 4],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
        }
    )

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("admitted as_index=False dissolve should not call pandas groupby")

    def _fail_to_pandas(*_args, **_kwargs):
        raise AssertionError("grouped reductions should not export before terminal materialization")

    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)
    monkeypatch.setattr(NativeGroupedAttributeReduction, "to_pandas", _fail_to_pandas)

    native_result = evaluate_geopandas_dissolve_native(
        frame,
        by="group",
        aggfunc={"value": "sum"},
        as_index=False,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert native_result.attributes.loader is not None
    assert tuple(native_result.attributes.columns) == ("group", "value")
    assert native_result.column_order == ("group", "geometry", "value")


@pytest.mark.gpu
def test_unary_dissolve_keeps_small_device_coverage_on_exact_gpu_path() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    geoms = [
        box(float(i), float(group) * 10.0, float(i) + 1.0, float(group) * 10.0 + 1.0)
        for group in range(3)
        for i in range(20)
    ]
    groups = np.repeat(np.arange(3, dtype=np.int32), 20)
    frame = geopandas.GeoDataFrame(
        {"group": groups, "value": np.arange(len(geoms), dtype=np.int32)},
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(geoms, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )

    clear_rewrite_events()
    actual = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)

    expected = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="coverage",
        grid_size=None,
        agg_kwargs={},
    )
    assert not any(
        event.rule_name == "R11_dissolve_unary_polygon_coverage_to_coverage"
        for event in rewrite_events
    )
    assert actual.geometry.dtype.name == "device_geometry"
    assert type(actual.geometry.values).__name__ == "DeviceGeometryArray"
    assert actual["value"].tolist() == expected["value"].tolist()
    actual_geoms = np.asarray(actual.geometry.array, dtype=object)
    expected_geoms = np.asarray(expected.geometry.array, dtype=object)
    assert all(
        bool(shapely.equals(actual_geom, expected_geom))
        for actual_geom, expected_geom in zip(actual_geoms, expected_geoms, strict=True)
    )


@pytest.mark.gpu
def test_unary_dissolve_does_not_rewrite_overlapping_polygon_groups() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    geoms = [box(float(i) * 0.5, 0.0, float(i) * 0.5 + 1.0, 1.0) for i in range(50)]
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(len(geoms), dtype=np.int32),
            "value": np.arange(len(geoms), dtype=np.int32),
        },
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(geoms, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )

    clear_rewrite_events()
    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)

    assert not any(
        event.rule_name == "R11_dissolve_unary_polygon_coverage_to_coverage"
        for event in rewrite_events
    )
    actual_geom = np.asarray(result.geometry.array, dtype=object)[0]
    expected_geom = shapely.union_all(np.asarray(frame.geometry.array, dtype=object))
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0


def test_evaluate_geopandas_dissolve_preserves_owned_backing_through_reset_index() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(16, dtype=np.int32),
            "value": np.arange(16, dtype=np.int32),
        },
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(
                    [Point(float(i), 0.0) for i in range(16)],
                    residency=Residency.DEVICE,
                ),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )
    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(2.0)
    assert getattr(buffered.geometry.values, "_owned", None) is not None

    result = evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    owned = getattr(result.geometry.values, "_owned", None)
    assert owned is not None
    assert owned.residency is Residency.DEVICE

    reset = result.reset_index(drop=True)
    reset_owned = getattr(reset.geometry.values, "_owned", None)
    assert reset_owned is not None
    assert reset_owned.residency is Residency.DEVICE


def test_benchmark_dissolve_pipeline_reports_group_count() -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": [0, 0, 1, 1],
            "value": [1, 2, 3, 4],
            "geometry": [Point(0, 0), Point(1, 1), Point(10, 10), Point(11, 11)],
        }
    )

    benchmark = benchmark_dissolve_pipeline(
        frame,
        by="group",
        dataset="points",
        iterations=1,
        warmup=0,
    )

    assert benchmark.rows == 4
    assert benchmark.groups == 2
    assert benchmark.iterations == 1
    assert benchmark.pipeline_elapsed_seconds >= 0.0
    assert benchmark.baseline_elapsed_seconds >= 0.0


def test_execute_grouped_union_codes_avoids_geometry_array_materialization_for_owned_unary(
    monkeypatch,
) -> None:
    geometry_array = GeometryArray.from_owned(
        from_shapely_geometries(
            [
                box(0, 0, 1, 1),
                box(1, 1, 2, 2),
            ]
        )
    )
    owned = getattr(geometry_array, "_owned", None)
    assert owned is not None

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "owned unary grouped union should not materialize the full GeometryArray"
        )

    monkeypatch.setattr(GeometryArray, "__array__", _fail, raising=False)

    grouped = dissolve_module.execute_grouped_union_codes(
        geometry_array,
        pd.Index([0, 0], dtype="int32").to_numpy(),
        group_count=1,
        method="unary",
        owned=owned,
    )

    assert grouped is not None
    assert grouped.group_count == 1
    assert grouped.non_empty_groups == 1
    assert grouped.owned is not None
    assert grouped.geometries is None


def test_execute_native_grouped_union_consumes_native_grouped_offsets_for_owned_unary(
    monkeypatch,
) -> None:
    geometry_array = GeometryArray.from_owned(
        from_shapely_geometries(
            [
                box(0, 0, 1, 1),
                box(1, 1, 2, 2),
                box(5, 0, 6, 1),
            ]
        )
    )
    owned = getattr(geometry_array, "_owned", None)
    assert owned is not None

    def _fail(*_args, **_kwargs):
        raise AssertionError("native grouped union should not materialize geometries")

    monkeypatch.setattr(GeometryArray, "__array__", _fail, raising=False)
    grouped = NativeGrouped.from_dense_codes(
        pd.Index([0, 0, 1], dtype="int32").to_numpy(),
        group_count=2,
    )

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=geometry_array,
        method="unary",
        owned=owned,
    )

    assert result is not None
    assert result.group_count == 2
    assert result.non_empty_groups == 2
    assert result.empty_groups == 0
    assert result.owned is not None
    assert result.geometries is None


@pytest.mark.gpu
def test_native_grouped_union_dense_host_codes_feed_disjoint_pack_metadata() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(10.0, 0.0, 11.0, 1.0),
            box(20.0, 0.0, 21.0, 1.0),
            box(30.0, 0.0, 31.0, 1.0),
            box(40.0, 0.0, 41.0, 1.0),
        ],
        residency=Residency.DEVICE,
    )
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_dense_codes(
        np.asarray([0, 0, 1, 2, 1], dtype=np.int32),
        group_count=3,
    )

    assert grouped.group_size_max == 2

    geopandas.clear_dispatch_events()
    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=owned,
    )
    events = geopandas.get_dispatch_events(clear=True)

    assert result is not None
    assert result.owned is not None
    assert result.geometries is None
    assert any(event.implementation == "native_grouped_disjoint_pack_union" for event in events)


@pytest.mark.gpu
def test_native_grouped_union_preserves_disjoint_near_degenerate_polygon() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    tiny = Polygon(
        [
            (673.205080756887, 400.0),
            (673.205080756888, 400.0),
            (673.205080756888, 400.000000000001),
            (673.205080756887, 400.0),
        ]
    )
    geoms = [box(540.0, 370.0, 640.0, 450.0), tiny]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_dense_codes(
        np.asarray([0, 0], dtype=np.int32),
        group_count=1,
    )

    geopandas.clear_fallback_events()
    if hasattr(geopandas, "clear_dispatch_events"):
        geopandas.clear_dispatch_events()

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=None,
        method="unary",
        owned=owned,
    )

    assert geopandas.get_fallback_events(clear=True) == []
    assert result is not None
    assert result.owned is not None
    out = result.owned.to_shapely()[0]
    expected = shapely.union_all(np.asarray(geoms, dtype=object))
    assert out.bounds[2] == pytest.approx(expected.bounds[2])
    assert out.convex_hull.area == pytest.approx(expected.convex_hull.area)
    assert any(
        event.implementation == "native_grouped_disjoint_pack_union"
        for event in geopandas.get_dispatch_events(clear=True)
    )


@pytest.mark.gpu
def test_native_grouped_union_area_proof_preserves_subresolution_remote_member() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    large = MultiPolygon([box(0.0, 0.0, 2.0, 2.0), box(4.0, 0.0, 5.0, 1.0)])
    overlap = box(1.0, 0.0, 3.0, 2.0)
    tiny = Polygon(
        [
            (673.2050807568876, 400.0),
            (673.2050807568874, 400.0),
            (673.2050807568877, 399.9999999999999),
            (673.2050807568876, 400.0),
        ]
    )
    geometries = [large, overlap, tiny]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_dense_codes(
        np.asarray([0, 0, 0], dtype=np.int32),
        group_count=1,
    )

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=owned,
    )

    assert result is not None
    assert result.owned is not None
    actual = result.owned.to_shapely()[0]
    expected = shapely.union_all(np.asarray(geometries, dtype=object))
    assert actual.bounds == pytest.approx(expected.bounds)
    assert shapely.area(shapely.convex_hull(actual)) == pytest.approx(
        shapely.area(shapely.convex_hull(expected))
    )
    assert (
        getattr(result.owned, "_native_grouped_union_implementation", None)
        != "native_grouped_noded_coverage_area_partition_union"
    )


@pytest.mark.gpu
def test_grouped_pairwise_tree_preserves_compacted_singleton_carry() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.geometry.owned import (
        build_empty_polygon_rows_device,
        device_take_owned_capacity_selection,
    )
    from vibespatial.kernels.constructive.segmented_union import (
        _segmented_union_device_grouped_pairwise_tree,
    )
    from vibespatial.runtime.precision import (
        CompensationMode,
        KernelClass,
        PrecisionMode,
        PrecisionPlan,
        RefinementMode,
    )

    tiny = Polygon(
        [
            (673.2050807568876, 400.0),
            (673.2050807568874, 400.0),
            (673.2050807568877, 399.9999999999999),
            (673.2050807568876, 400.0),
        ]
    )
    sparse = from_shapely_geometries(
        [None, tiny, None],
        residency=Residency.DEVICE,
    )
    selection = NativeDeviceSelection.from_mask(
        cp.asarray([False, True, False], dtype=cp.bool_)
    )
    compacted = device_take_owned_capacity_selection(sparse, selection)
    precision_plan = PrecisionPlan(
        storage_precision=PrecisionMode.FP64,
        compute_precision=PrecisionMode.FP64,
        kernel_class=KernelClass.CONSTRUCTIVE,
        compensation=CompensationMode.NONE,
        refinement=RefinementMode.NONE,
        center_coordinates=False,
        reason="grouped singleton carry regression",
    )

    result = _segmented_union_device_grouped_pairwise_tree(
        compacted,
        cp.asarray([0, 1], dtype=cp.int64),
        cp.asarray([0], dtype=cp.int64),
        output_row_count=1,
        precision_plan=precision_plan,
        empty_output=build_empty_polygon_rows_device(1),
        all_groups_observed=False,
        group_size_max=3,
    )

    assert result is not None
    actual = result.to_shapely()[0]
    assert actual is not None
    assert actual.wkb == tiny.wkb


@pytest.mark.gpu
def test_native_grouped_union_positive_area_sliver_uses_grouped_disjoint_pack() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    sliver = Polygon(
        [
            (4.0, 0.0),
            (4.000001, 0.0),
            (4.0, 0.000001),
            (4.0, 0.0),
        ]
    )
    assert 0.0 < shapely.area(sliver) <= dissolve_module.SPATIAL_EPSILON
    values = [box(0.0, 0.0, 1.0, 1.0), box(2.0, 0.0, 3.0, 1.0), sliver]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_dense_codes(
        cp.zeros(len(values), dtype=cp.int32),
        group_count=1,
        all_groups_observed=True,
        group_size_min=len(values),
        group_size_max=len(values),
    )

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=owned,
    )

    assert result is not None
    assert result.owned is not None
    assert (
        getattr(result.owned, "_native_grouped_union_implementation", None)
        == "native_grouped_disjoint_pack_partition_union"
    )
    actual = result.owned.to_shapely()[0]
    expected = shapely.union_all(np.asarray(values, dtype=object))
    assert shapely.area(shapely.symmetric_difference(actual, expected)) == pytest.approx(0.0)
    assert shapely.area(shapely.convex_hull(actual)) == pytest.approx(
        shapely.area(shapely.convex_hull(expected)),
    )


@pytest.mark.gpu
def test_execute_native_grouped_union_sparse_host_codes_scatter_empty_groups_on_device(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    geoms = [
        box(0, 0, 1, 1),
        box(1, 0, 2, 1),
        box(10, 0, 11, 1),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_dense_codes(
        np.asarray([0, 0, 2], dtype=np.int32),
        group_count=4,
    )

    def _fail_to_shapely(*_args, **_kwargs):
        raise AssertionError("sparse native grouped union should not export Shapely rows")

    clear_materialization_events()
    with monkeypatch.context() as patch:
        patch.setattr(OwnedGeometryArray, "to_shapely", _fail_to_shapely)
        result = dissolve_module.execute_native_grouped_union(
            grouped,
            _geometries=(),
            method="unary",
            owned=owned,
        )
    events = get_materialization_events(clear=True)

    assert result is not None
    assert result.group_count == 4
    assert result.non_empty_groups == 2
    assert result.empty_groups == 2
    assert result.geometries is None
    assert result.owned is not None
    assert result.owned.residency is Residency.DEVICE
    assert result.owned.row_count == 4
    assert events == []

    actual = np.asarray(result.owned.to_shapely(), dtype=object)
    assert shapely.equals(actual[0], shapely.union_all(np.asarray(geoms[:2], dtype=object)))
    assert actual[1] is not None and actual[1].is_empty
    assert shapely.equals(actual[2], geoms[2])
    assert actual[3] is not None and actual[3].is_empty


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("executor", "method"),
    [
        (dissolve_module.execute_grouped_coverage_union_gpu, DissolveUnionMethod.COVERAGE),
        (
            dissolve_module.execute_grouped_disjoint_subset_union_gpu,
            DissolveUnionMethod.DISJOINT_SUBSET,
        ),
    ],
)
def test_grouped_owned_union_group_positions_sparse_scatter_empty_groups_on_device(
    monkeypatch,
    executor,
    method,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    geoms = [
        box(0, 0, 1, 1),
        box(2, 0, 3, 1),
        box(10, 0, 11, 1),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    group_positions = [
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([], dtype=np.int64),
        np.asarray([2], dtype=np.int64),
        np.asarray([], dtype=np.int64),
    ]

    def _fail_to_shapely(*_args, **_kwargs):
        raise AssertionError("sparse grouped owned union should not export Shapely rows")

    clear_materialization_events()
    with monkeypatch.context() as patch:
        patch.setattr(OwnedGeometryArray, "to_shapely", _fail_to_shapely)
        result = executor(None, group_positions, owned=owned)
    events = get_materialization_events(clear=True)

    assert result is not None
    assert result.method is method
    assert result.group_count == 4
    assert result.non_empty_groups == 2
    assert result.empty_groups == 2
    assert result.geometries is None
    assert result.owned is not None
    assert result.owned.residency is Residency.DEVICE
    assert result.owned.row_count == 4
    assert events == []

    actual = np.asarray(result.owned.to_shapely(), dtype=object)
    expected = (
        shapely.coverage_union_all(np.asarray(geoms[:2], dtype=object))
        if method is DissolveUnionMethod.COVERAGE
        else shapely.disjoint_subset_union_all(np.asarray(geoms[:2], dtype=object))
    )
    assert shapely.equals(actual[0], expected)
    assert actual[1] is not None and actual[1].is_empty
    assert shapely.equals(actual[2], geoms[2])
    assert actual[3] is not None and actual[3].is_empty


def test_execute_native_grouped_union_consumes_device_codes_for_owned_coverage(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for NativeGrouped coverage box reducer")
    cp = pytest.importorskip("cupy")

    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
            box(10, 10, 11, 11),
            box(11, 10, 12, 11),
        ],
        residency=Residency.DEVICE,
    )
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 0, 1, 1], dtype=cp.int32),
        group_count=2,
    )

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "device NativeGrouped coverage union should not host-normalize group codes"
        )

    monkeypatch.setattr(dissolve_module, "_group_non_empty_counts", _fail)

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="coverage",
        owned=owned,
    )

    assert result is not None
    assert result.group_count == 2
    assert result.non_empty_groups == 2
    assert result.empty_groups == 0
    assert result.owned is not None
    assert result.geometries is None
    assert result.owned.residency is Residency.DEVICE
    actual = np.asarray(result.owned.to_shapely(), dtype=object)
    expected = np.asarray([box(0, 0, 2, 1), box(10, 10, 12, 11)], dtype=object)
    assert shapely.equals(actual, expected).tolist() == [
        True,
        True,
    ]


@pytest.mark.gpu
def test_execute_native_grouped_union_all_valid_cache_avoids_group_scalar_fences() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    values = [
        box(0.0, 0.0, 1.0, 1.0),
        box(0.5, 0.0, 1.5, 1.0),
        box(10.0, 0.0, 11.0, 1.0),
        box(10.5, 0.0, 11.5, 1.0),
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 0, 1, 1], dtype=cp.int32),
        group_count=2,
    )
    clear_materialization_events()
    reset_d2h_transfer_count()

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=owned,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result is not None
    assert result.group_count == 2
    assert result.non_empty_groups == 2
    assert result.empty_groups == 0
    assert result.owned is not None
    assert result.owned.residency is Residency.DEVICE
    assert get_materialization_events(clear=True) == []
    assert "overlay dissolve native grouped-union valid-row count fence" not in reasons
    assert "overlay dissolve native grouped-union nonempty-group count fence" not in reasons

    actual = np.asarray(result.owned.to_shapely(), dtype=object)
    expected = [
        shapely.union_all(np.asarray(values[:2], dtype=object)),
        shapely.union_all(np.asarray(values[2:], dtype=object)),
    ]
    assert bool(shapely.equals(actual[0], expected[0]))
    assert bool(shapely.equals(actual[1], expected[1]))


@pytest.mark.gpu
def test_execute_native_grouped_union_all_invalid_capacity_stays_device_native() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.geometry.owned import (
        device_mask_owned_capacity,
        device_valid_nonempty_mask,
    )

    owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(10.0, 0.0, 11.0, 1.0),
            box(10.5, 0.0, 11.5, 1.0),
        ],
        residency=Residency.DEVICE,
    )
    masked = device_mask_owned_capacity(owned, cp.zeros(owned.row_count, dtype=cp.bool_))
    grouped = NativeGrouped.from_sorted_offsets(
        cp.asarray([0, 1, 3], dtype=cp.int64),
        row_count=owned.row_count,
        all_groups_observed=True,
        group_size_min=1,
        group_size_max=2,
    )
    clear_materialization_events()
    reset_d2h_transfer_count()

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=masked,
    )
    d2h_events = get_d2h_transfer_events(clear=True)

    assert result is not None
    assert result.group_count == 2
    assert result.owned is not None
    assert result.owned.residency is Residency.DEVICE
    state = result.owned._ensure_device_state(preserve_indexed_view=True)
    assert bool(cp.all(state.validity))
    assert not bool(cp.any(device_valid_nonempty_mask(result.owned)))
    assert get_materialization_events(clear=True) == []
    assert d2h_events == []
    assert sum(event.bytes_transferred for event in d2h_events) <= 2040


@pytest.mark.gpu
def test_execute_native_grouped_union_host_validity_proof_avoids_valid_count_fence() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    values = [
        box(0.0, 0.0, 1.0, 1.0),
        box(0.5, 0.0, 1.5, 1.0),
        box(10.0, 0.0, 11.0, 1.0),
        box(10.5, 0.0, 11.5, 1.0),
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 0, 1, 1], dtype=cp.int32),
        group_count=2,
    )
    reset_d2h_transfer_count()

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=owned,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result is not None
    assert result.group_count == 2
    assert result.non_empty_groups == 2
    assert result.empty_groups == 0
    assert result.owned is not None
    assert result.owned.residency is Residency.DEVICE
    assert "overlay dissolve native grouped-union valid-row count fence" not in reasons
    assert "overlay dissolve native grouped-union nonempty-group count fence" not in reasons


@pytest.mark.gpu
def test_device_take_preserves_all_valid_proof_for_native_grouped_union() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    values = [
        box(0.0, 0.0, 1.0, 1.0),
        box(0.5, 0.0, 1.5, 1.0),
        box(10.0, 0.0, 11.0, 1.0),
        box(10.5, 0.0, 11.5, 1.0),
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)
    subset = owned.device_take(cp.asarray([0, 1, 2, 3], dtype=cp.int64))
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 0, 1, 1], dtype=cp.int32),
        group_count=2,
    )
    reset_d2h_transfer_count()

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=subset,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result is not None
    assert result.group_count == 2
    assert result.non_empty_groups == 2
    assert result.empty_groups == 0
    assert result.owned is not None
    assert result.owned.residency is Residency.DEVICE
    assert "overlay dissolve native grouped-union valid-row count fence" not in reasons
    assert "overlay dissolve native grouped-union nonempty-group count fence" not in reasons


@pytest.mark.gpu
def test_native_grouped_union_singleton_identity_preserves_row_indirection() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    values = [
        Polygon([(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 0.0)]),
        Polygon(
            [(10.0, 0.0), (14.0, 0.0), (14.0, 4.0), (10.0, 0.0)],
            [[(11.0, 1.0), (12.0, 1.0), (12.0, 2.0), (11.0, 1.0)]],
        ),
        MultiPolygon(
            [
                Polygon([(20.0, 0.0), (21.0, 0.0), (21.0, 1.0), (20.0, 0.0)]),
                Polygon([(22.0, 0.0), (23.0, 0.0), (23.0, 1.0), (22.0, 0.0)]),
            ]
        ),
        Polygon([(30.0, 0.0), (31.0, 0.0), (31.0, 1.0), (30.0, 0.0)]),
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)
    reordered = owned.device_take(cp.asarray([1, 2, 0, 3], dtype=cp.int64))
    assert reordered.is_indexed_view
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 1, 2, 3], dtype=cp.int32),
        group_count=4,
        all_groups_observed=True,
        group_size_min=1,
        group_size_max=1,
    )
    reset_d2h_transfer_count()

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=reordered,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result is not None
    assert result.owned is not None
    assert result.owned.is_indexed_view
    assert (
        getattr(result.owned, "_native_grouped_union_implementation", None)
        == "native_grouped_singleton_identity_union"
    )
    assert "owned geometry device-take nested slice-size allocation fence" not in reasons
    assert "owned geometry device-take slice-size allocation fence" not in reasons


@pytest.mark.gpu
def test_execute_native_grouped_union_host_codes_device_owned_uses_native_carrier(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import vibespatial.kernels.constructive.segmented_union as segmented_union_module

    values = [
        box(0.0, 0.0, 1.0, 1.0),
        box(0.5, 0.0, 1.5, 1.0),
        box(10.0, 0.0, 11.0, 1.0),
        box(10.5, 0.0, 11.5, 1.0),
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_dense_codes(
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        group_count=2,
    )

    def _fail_segmented_union_all(*_args, **_kwargs):
        raise AssertionError("host-code device grouped union must use native carrier")

    monkeypatch.setattr(
        segmented_union_module,
        "segmented_union_all",
        _fail_segmented_union_all,
    )
    geopandas.clear_dispatch_events()

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=owned,
    )
    dispatch_events = geopandas.get_dispatch_events(clear=True)

    assert result is not None
    assert result.owned is not None
    assert result.owned.residency is Residency.DEVICE
    assert any(
        event.implementation
        in {
            "native_grouped_overlay_union_plan",
            "native_grouped_device_pairwise_tree_union",
            "native_grouped_rectangle_strip_union",
            "native_grouped_rectangle_strip_partition_union",
            "native_grouped_disjoint_pack_partition_union",
        }
        for event in dispatch_events
    )
    actual = np.asarray(result.owned.to_shapely(), dtype=object)
    expected = [
        shapely.union_all(np.asarray(values[:2], dtype=object)),
        shapely.union_all(np.asarray(values[2:], dtype=object)),
    ]
    assert bool(shapely.equals(actual[0], expected[0]))
    assert bool(shapely.equals(actual[1], expected[1]))


@pytest.mark.gpu
def test_execute_native_grouped_union_prefers_grouped_overlay_plan_for_overlaps() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    values = [
        Polygon([(0.0, 0.0), (3.0, 0.0), (1.5, 2.0), (0.0, 0.0)]),
        Polygon([(1.0, -0.2), (4.0, -0.2), (2.5, 2.2), (1.0, -0.2)]),
        Polygon([(2.0, 0.1), (5.0, 0.1), (3.5, 2.1), (2.0, 0.1)]),
        Polygon([(10.0, 0.0), (13.0, 0.0), (11.5, 2.0), (10.0, 0.0)]),
        Polygon([(11.0, -0.2), (14.0, -0.2), (12.5, 2.2), (11.0, -0.2)]),
        Polygon([(12.0, 0.1), (15.0, 0.1), (13.5, 2.1), (12.0, 0.1)]),
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 0, 0, 1, 1, 1], dtype=cp.int32),
        group_count=2,
    )

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=owned,
    )

    assert result is not None
    assert result.owned is not None
    assert getattr(result.owned, "_native_grouped_union_implementation", None) in {
        "native_grouped_overlay_union_plan",
        "native_grouped_overlay_union_plan_mixed_degenerate_pairwise",
    }
    actual = np.asarray(result.owned.to_shapely(), dtype=object)
    expected = [
        shapely.union_all(np.asarray(values[:3], dtype=object)),
        shapely.union_all(np.asarray(values[3:], dtype=object)),
    ]
    assert bool(shapely.is_valid(actual).all())
    assert shapely.area(shapely.symmetric_difference(actual[0], expected[0])) == pytest.approx(0.0)
    assert shapely.area(shapely.symmetric_difference(actual[1], expected[1])) == pytest.approx(0.0)


@pytest.mark.gpu
def test_native_grouped_union_splits_overlapping_nonseed_members() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    values = [
        box(0.0, 0.0, 1.0, 1.0),
        box(10.0, 0.0, 12.0, 2.0),
        box(11.0, 0.0, 13.0, 2.0),
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_dense_codes(
        cp.zeros(len(values), dtype=cp.int32),
        group_count=1,
    )

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=owned,
    )

    assert result is not None
    assert result.owned is not None
    assert getattr(result.owned, "_native_grouped_union_implementation", None) in {
        "native_grouped_overlay_union_plan",
        "native_grouped_overlay_union_plan_mixed_degenerate_pairwise",
    }
    actual = result.owned.to_shapely()[0]
    expected = shapely.union_all(np.asarray(values, dtype=object))
    assert shapely.is_valid(actual)
    assert shapely.equals(actual, expected)


@pytest.mark.gpu
def test_native_grouped_union_resolves_collapsed_triangle_face_coverage() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    values = [
        MultiPolygon(
            [
                Polygon(
                    [
                        (780.0, 590.0),
                        (780.0, 583.2685564925192),
                        (782.5116250684673, 583.6695836409644),
                        (784.4604284565498, 587.6245920319477),
                        (786.5271606630206, 582.6688642367599),
                        (786.7692540869118, 582.9489629706218),
                        (786.879087668319, 584.0164294642864),
                        (786.5876592030364, 584.0080913740769),
                        (786.299321872665, 590.0),
                        (780.0, 590.0),
                    ]
                ),
                Polygon(
                    [
                        (787.0222109887794, 584.0205243805326),
                        (787.1878569706536, 583.4332807015311),
                        (787.7124901638854, 584.0402740299919),
                        (787.0222109887794, 584.0205243805326),
                    ]
                ),
            ]
        ),
        MultiPolygon(
            [
                Polygon(
                    [
                        (786.1651378875935, 590.0),
                        (784.9364055555279, 586.4832671269147),
                        (786.6982284043847, 582.2586682964808),
                        (786.879087668319, 584.0164294642864),
                        (786.5876592030364, 584.0080913740769),
                        (786.299321872665, 590.0),
                        (786.1651378875935, 590.0),
                    ]
                ),
                Polygon(
                    [
                        (787.0222109887794, 584.0205243805326),
                        (788.156295070867, 580.0),
                        (790.0, 580.0),
                        (790.0, 584.1057222124479),
                        (787.0222109887794, 584.0205243805326),
                    ]
                ),
            ]
        ),
        MultiPolygon(
            [
                Polygon(
                    [
                        (784.665512077896, 590.0),
                        (784.5224222293724, 587.4759398446234),
                        (786.6982284043847, 582.2586682964808),
                        (786.879087668319, 584.0164294642864),
                        (786.5876592030364, 584.0080913740769),
                        (786.299321872665, 590.0),
                        (784.665512077896, 590.0),
                    ]
                ),
                Polygon(
                    [
                        (787.0222109887794, 584.0205243805326),
                        (788.156295070867, 580.0),
                        (790.0, 580.0),
                        (790.0, 584.1057222124479),
                        (787.0222109887794, 584.0205243805326),
                    ]
                ),
            ]
        ),
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_dense_codes(
        cp.zeros(len(values), dtype=cp.int32),
        group_count=1,
        all_groups_observed=True,
        group_size_min=len(values),
        group_size_max=len(values),
    )

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=owned,
    )

    assert result is not None
    assert result.owned is not None
    actual = result.owned.to_shapely()[0]
    expected = shapely.union_all(np.asarray(values, dtype=object))
    assert shapely.is_valid(actual)
    assert shapely.area(shapely.symmetric_difference(actual, expected)) == pytest.approx(0.0)


@pytest.mark.gpu
def test_grouped_union_residual_repair_uses_coverage_merge_carrier() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.kernels.constructive import segmented_union as segmented_union_module
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    inputs = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), box(1.0, 0.0, 2.0, 1.0)],
        residency=Residency.DEVICE,
    )
    candidate = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0)],
        residency=Residency.DEVICE,
    )
    seed_all_validity_cache(inputs)
    seed_all_validity_cache(candidate)
    failure = segmented_union_module._grouped_union_failure_from_mask(
        cp.asarray([False, True], dtype=cp.bool_),
        cp.asarray([0, 0], dtype=cp.int64),
        output_row_count=1,
    )

    clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    repaired = segmented_union_module._repair_grouped_union_uncovered_rows_device(
        candidate,
        inputs,
        failure,
        output_row_count=1,
        empty_output=dissolve_module._empty_polygon_rows_device(1),
        stage="test_residual_repair",
    )
    events = get_dispatch_events(clear=True)
    runtime_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert repaired is not None
    assert repaired.residency is Residency.DEVICE
    assert repaired.row_count == 1
    assert any(
        event.implementation == "native_grouped_overlay_union_residual_coverage_merge"
        for event in events
    )
    assert not any(
        event.implementation == "single_row_direct_union_residual_proof_gpu" for event in events
    )
    assert not any("area-equivalence scalar fence" in reason for reason in runtime_reasons)
    assert "grouped polygon coverage-union segment-count fence" not in runtime_reasons
    actual = repaired.to_shapely()[0]
    expected = box(0.0, 0.0, 2.0, 1.0)
    assert shapely.area(shapely.symmetric_difference(actual, expected)) == pytest.approx(0.0)


@pytest.mark.gpu
def test_grouped_union_constructive_coverage_keeps_positive_area_slivers() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.kernels.constructive import segmented_union as segmented_union_module

    sliver_width = 1.0e-7
    inputs = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(1.0 - sliver_width, 0.0, 1.0 + sliver_width, 1.0),
        ],
        residency=Residency.DEVICE,
    )
    candidate = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0)],
        residency=Residency.DEVICE,
    )
    seed_all_validity_cache(inputs)
    seed_all_validity_cache(candidate)

    failure = segmented_union_module._grouped_union_coverage_failure_device(
        candidate,
        inputs,
        cp.asarray([0, 2], dtype=cp.int64),
        cp.asarray([0], dtype=cp.int64),
        output_row_count=1,
        stage="test_positive_area_sliver",
        group_size_max=2,
        source_segment_span_max=4,
    )

    assert failure is not None
    assert failure.residuals is not None
    assert int(cp.asnumpy(failure.failed_selection.logical_count)[0]) == 1


@pytest.mark.gpu
def test_grouped_overlay_union_plan_keeps_repair_gate_live(monkeypatch) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    values = [
        Polygon([(0.0, 0.0), (3.0, 0.0), (1.5, 2.0), (0.0, 0.0)]),
        Polygon([(1.0, -0.2), (4.0, -0.2), (2.5, 2.2), (1.0, -0.2)]),
        Polygon([(2.0, 0.1), (5.0, 0.1), (3.5, 2.1), (2.0, 0.1)]),
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_dense_codes(
        cp.zeros(len(values), dtype=cp.int32),
        group_count=1,
    )
    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=owned,
    )
    assert result is not None
    assert result.owned is not None
    assert getattr(result.owned, "_native_grouped_union_implementation", None) in {
        "native_grouped_overlay_union_plan",
        "native_grouped_overlay_union_plan_mixed_degenerate_pairwise",
    }

    calls = 0

    def _repair_probe(_owned, *, exact_collinearity=False):
        nonlocal calls
        assert exact_collinearity is True
        calls += 1
        return cp.empty(0, dtype=cp.int64)

    monkeypatch.setattr(
        dissolve_module,
        "_device_grouped_union_invalid_rows",
        _repair_probe,
    )
    payload = dissolve_module._grouped_union_geometry_payload(
        result,
        geometry_name="geometry",
        crs="EPSG:3857",
    )

    assert payload.geometry.owned is result.owned
    assert calls == 1


@pytest.mark.gpu
def test_native_grouped_union_degenerate_area_declines_grouped_overlay_plan() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    values = [
        box(0.0, 0.0, 1.0, 1.0),
        box(0.5, 0.0, 1.5, 1.0),
        Polygon([(10.0, 0.0), (11.0, 0.0), (10.5, 1.0e-14), (10.0, 0.0)]),
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_dense_codes(
        cp.zeros(len(values), dtype=cp.int32),
        group_count=1,
    )

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=owned,
    )

    assert result is not None
    assert result.owned is not None
    assert (
        getattr(result.owned, "_native_grouped_union_implementation", None)
        != "native_grouped_overlay_union_plan"
    )
    actual = result.owned.to_shapely()[0]
    expected = shapely.union_all(np.asarray(values, dtype=object))
    assert shapely.area(shapely.symmetric_difference(actual, expected)) == pytest.approx(0.0)
    assert shapely.area(shapely.convex_hull(actual)) == pytest.approx(
        shapely.area(shapely.convex_hull(expected)),
    )


@pytest.mark.gpu
def test_native_grouped_union_splits_degenerate_groups_without_declining_overlay_plan() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    values = [
        Polygon([(0.0, 0.0), (3.0, 0.0), (1.5, 2.0), (0.0, 0.0)]),
        Polygon([(1.0, -0.2), (4.0, -0.2), (2.5, 2.2), (1.0, -0.2)]),
        Polygon([(2.0, 0.1), (5.0, 0.1), (3.5, 2.1), (2.0, 0.1)]),
        box(10.0, 0.0, 11.0, 1.0),
        box(10.5, 0.0, 11.5, 1.0),
        Polygon([(20.0, 0.0), (21.0, 0.0), (20.5, 1.0e-14), (20.0, 0.0)]),
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 0, 0, 1, 1, 1], dtype=cp.int32),
        group_count=2,
    )

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=owned,
    )

    assert result is not None
    assert result.owned is not None
    assert (
        getattr(result.owned, "_native_grouped_union_implementation", None)
        == "native_grouped_overlay_union_plan_mixed_degenerate_pairwise"
    )
    actual = np.asarray(result.owned.to_shapely(), dtype=object)
    expected = [
        shapely.union_all(np.asarray(values[:3], dtype=object)),
        shapely.union_all(np.asarray(values[3:], dtype=object)),
    ]
    for got, want in zip(actual, expected, strict=True):
        assert shapely.area(shapely.symmetric_difference(got, want)) == pytest.approx(0.0)
        assert shapely.area(shapely.convex_hull(got)) == pytest.approx(
            shapely.area(shapely.convex_hull(want)),
        )


@pytest.mark.gpu
def test_execute_grouped_union_codes_batches_multi_group_unary_on_gpu(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import vibespatial.constructive.union_all as union_all_module

    def _fail_serial_union_all(*_args, **_kwargs):
        raise AssertionError("multi-group unary dissolve should not run serial union_all per group")

    monkeypatch.setattr(union_all_module, "union_all_gpu_owned", _fail_serial_union_all)

    values = [
        *[box(float(i) * 0.5, 0, float(i) * 0.5 + 1.0, 1.0) for i in range(12)],
        *[box(20.0 + float(i) * 0.5, 0, 21.0 + float(i) * 0.5, 1.0) for i in range(12)],
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)

    class ExplodingGeometries:
        def __array__(self, dtype=None):
            raise AssertionError(
                "owned unary grouped union should not materialize geometry objects"
            )

    grouped = dissolve_module.execute_grouped_union_codes(
        ExplodingGeometries(),
        np.repeat(np.asarray([0, 1], dtype=np.int32), 12),
        group_count=2,
        method="unary",
        owned=owned,
    )

    assert grouped is not None
    assert grouped.geometries is None
    assert grouped.owned is not None
    assert grouped.owned.row_count == 2

    actual = np.asarray(grouped.owned.to_shapely(), dtype=object)
    expected = [
        shapely.union_all(np.asarray(values[:12], dtype=object)),
        shapely.union_all(np.asarray(values[12:], dtype=object)),
    ]
    assert bool(shapely.equals(actual[0], expected[0]))
    assert bool(shapely.equals(actual[1], expected[1]))


@pytest.mark.gpu
def test_execute_native_grouped_union_rectangle_strips_use_direct_carrier() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    group_sizes = np.asarray([2, 3, 4, 5, 6, 7, 8, 2], dtype=np.int32)
    group_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int32), np.cumsum(group_sizes, dtype=np.int32)]
    )
    values = []
    for group_index, group_size in enumerate(group_sizes):
        x0 = float(group_index) * 20.0
        for row in range(int(group_size)):
            left = x0 + float(row) * 0.45
            values.append(box(left, 0.0, left + 1.0, 1.0))

    owned = from_shapely_geometries(values, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_sorted_offsets(
        cp.asarray(group_offsets, dtype=cp.int64),
        row_count=owned.row_count,
        source_token="test-rectangle-strip-groups",
        all_groups_observed=True,
        group_size_min=int(group_sizes.min()),
        group_size_max=int(group_sizes.max()),
    )

    geopandas.clear_dispatch_events()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=owned,
    )
    d2h_events = get_d2h_transfer_events(clear=True)
    events = geopandas.get_dispatch_events(clear=True)

    assert result is not None
    assert result.owned is not None
    assert result.owned.residency is Residency.DEVICE
    assert d2h_events == []
    assert sum(event.bytes_transferred for event in d2h_events) <= 2040
    assert any(
        event.implementation
        in {
            "native_grouped_rectangle_strip_union",
            "native_grouped_rectangle_strip_partition_union",
        }
        for event in events
    )

    actual = np.asarray(result.owned.to_shapely(), dtype=object)
    expected = [
        shapely.union_all(np.asarray(values[int(start) : int(end)], dtype=object))
        for start, end in zip(group_offsets[:-1], group_offsets[1:], strict=True)
    ]
    for got, want in zip(actual, expected, strict=True):
        assert bool(
            shapely.normalize(got).equals_exact(
                shapely.normalize(want),
                tolerance=1.0e-9,
            )
        )
        assert shapely.area(shapely.symmetric_difference(got, want)) == pytest.approx(
            0.0,
            abs=1.0e-9,
        )


@pytest.mark.gpu
def test_grouped_union_partitions_rectangle_strips_from_exact_groups() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    values = [
        box(0.0, 0.0, 1.0, 1.0),
        box(0.5, 0.0, 1.5, 1.0),
        box(10.0, 0.0, 12.0, 1.0),
        box(11.0, 0.5, 13.0, 2.0),
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_sorted_offsets(
        cp.asarray([0, 2, 4], dtype=cp.int64),
        row_count=owned.row_count,
        source_token="mixed-rectangle-strip-partition-test",
        all_groups_observed=True,
        group_size_min=2,
        group_size_max=2,
    )

    clear_dispatch_events()
    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=owned,
    )
    events = get_dispatch_events(clear=True)

    assert result is not None
    assert result.owned is not None
    assert (
        getattr(result.owned, "_device_scatter_implementation", None)
        == "device_capacity_partition_selection"
    )
    assert any(
        event.implementation == "native_grouped_rectangle_strip_partition_union" for event in events
    )
    actual = np.asarray(result.owned.to_shapely(), dtype=object)
    expected = [
        shapely.union_all(np.asarray(values[:2], dtype=object)),
        shapely.union_all(np.asarray(values[2:], dtype=object)),
    ]
    for got, want in zip(actual, expected, strict=True):
        assert got.is_valid
        assert shapely.normalize(got).equals_exact(
            shapely.normalize(want),
            tolerance=1.0e-9,
        )


@pytest.mark.gpu
def test_rectangle_strip_union_uses_active_polygon_rows_from_mixed_indexed_view() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    strip_values = [
        box(0.0, 0.0, 1.0, 1.0),
        box(0.5, 0.0, 1.5, 1.0),
        box(10.0, 0.0, 11.0, 1.0),
        box(10.5, 0.0, 11.5, 1.0),
    ]
    source = from_shapely_geometries(
        [
            strip_values[0],
            strip_values[1],
            MultiPolygon([box(50.0, 0.0, 51.0, 1.0)]),
            strip_values[2],
            strip_values[3],
        ],
        residency=Residency.DEVICE,
    )
    owned = source.device_take(cp.asarray([0, 1, 3, 4], dtype=cp.int64))
    seed_all_validity_cache(owned)
    grouped = NativeGrouped.from_sorted_offsets(
        cp.asarray([0, 2, 4], dtype=cp.int64),
        row_count=owned.row_count,
        source_token="mixed-indexed-rectangle-strip-test",
        all_groups_observed=True,
        group_size_min=2,
        group_size_max=2,
    )
    clear_dispatch_events()

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="unary",
        owned=owned,
    )
    events = get_dispatch_events(clear=True)

    assert result is not None
    assert result.owned is not None
    assert any(
        event.implementation
        in {
            "native_grouped_rectangle_strip_union",
            "native_grouped_rectangle_strip_partition_union",
        }
        for event in events
    )
    assert not any(event.implementation == "native_grouped_overlay_union_plan" for event in events)
    actual = np.asarray(result.owned.to_shapely(), dtype=object)
    expected = [
        shapely.union_all(np.asarray(strip_values[:2], dtype=object)),
        shapely.union_all(np.asarray(strip_values[2:], dtype=object)),
    ]
    for got, want in zip(actual, expected, strict=True):
        assert shapely.normalize(got).equals_exact(
            shapely.normalize(want),
            tolerance=1.0e-9,
        )


@pytest.mark.gpu
def test_small_grouped_constructive_reduce_batches_many_tiny_groups_on_gpu() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    group_sizes = np.asarray([2, 3, 4, 5, 6, 7, 8, 2], dtype=np.int32)
    group_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int32), np.cumsum(group_sizes, dtype=np.int32)]
    )
    values = []
    for group_index, group_size in enumerate(group_sizes):
        x0 = float(group_index) * 20.0
        for row in range(int(group_size)):
            left = x0 + float(row) * 0.45
            values.append(box(left, 0.0, left + 1.0, 1.0))

    owned = from_shapely_geometries(values, residency=Residency.DEVICE)

    geopandas.clear_dispatch_events()
    result = segmented_union_all(
        owned,
        group_offsets,
        dispatch_mode=ExecutionMode.GPU,
    )
    events = geopandas.get_dispatch_events(clear=True)

    assert result.residency is Residency.DEVICE
    assert result.row_count == len(group_sizes)
    assert any(
        event.surface == "segmented_union_all"
        and event.operation == "segmented_union_strategy"
        and event.implementation == "gpu_native_grouped_constructive_carrier"
        for event in events
    )

    actual = np.asarray(result.to_shapely(), dtype=object)
    expected = [
        shapely.union_all(np.asarray(values[int(start) : int(end)], dtype=object))
        for start, end in zip(group_offsets[:-1], group_offsets[1:], strict=True)
    ]
    for got, want in zip(actual, expected, strict=True):
        got_norm = shapely.normalize(got)
        want_norm = shapely.normalize(want)
        assert bool(got_norm.equals_exact(want_norm, tolerance=1.0e-9))
        assert shapely.area(shapely.symmetric_difference(got, want)) == pytest.approx(
            0.0,
            abs=1.0e-9,
        )


def test_grouped_convex_hull_uses_polygon_part_capacity_carrier() -> None:
    convex_hull_module = importlib.import_module("vibespatial.constructive.convex_hull")
    source = Path(convex_hull_module.__file__).read_text()
    function_source = source[
        source.index("def _grouped_convex_hull_from_source_owned(") : source.index(
            "\n\n# ---------------------------------------------------------------------------\n# Public dispatch API"
        )
    ]

    assert "_explode_polygonal_rows_to_polygon_capacity_gpu" in function_source
    assert "NativeGroupedSelection" in function_source
    assert "allow_capacity_allocation=True" in function_source
    assert "polygon_parts.selection.active_capacity_mask()" in function_source
    assert "_explode_polygonal_rows_to_polygons_gpu" not in function_source
    assert "cp.argsort" not in function_source


def test_segmented_union_residual_relation_uses_polygon_part_capacity() -> None:
    segmented_union_module = importlib.import_module(
        "vibespatial.kernels.constructive.segmented_union"
    )
    source = Path(segmented_union_module.__file__).read_text()
    start = source.index("def _failed_input_residuals_against_candidate_parts_gpu(")
    end = source.index("\ndef ", start + 1)
    function_source = source[start:end]

    assert "_explode_polygonal_rows_to_polygon_capacity_gpu" in function_source
    assert "candidate_part_capacity.selection.active_capacity_mask()" in function_source
    assert "relation_capacity = failed_count * max_part_count" not in function_source
    assert "relation_tile_slots" in function_source
    assert "cp.nonzero(d_related)" in function_source
    assert "_physicalize_polygon_relation_rows_device" in function_source
    assert "grouped union residual repair admission scalar fence" not in function_source
    assert "copy_device_to_host" not in function_source
    assert "device_take_capacity(" in function_source
    assert "active_pairs=device-resident" in function_source
    assert "cp.unique(" not in function_source
    assert "cp.flatnonzero(" not in function_source
    assert "_explode_polygonal_rows_to_polygons_gpu" not in function_source


@pytest.mark.gpu
def test_public_dissolve_then_convex_hull_uses_grouped_hull_rewrite() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    values = [
        *[box(float(i) * 0.55, 0.0, float(i) * 0.55 + 1.0, 1.0) for i in range(12)],
        *[box(20.0 + float(i) * 0.55, 0.0, 21.0 + float(i) * 0.55, 1.0) for i in range(12)],
    ]
    groups = np.repeat(np.asarray([0, 1], dtype=np.int32), 12)
    frame = geopandas.GeoDataFrame(
        {
            "group": groups,
            "value": np.arange(len(values), dtype=np.int32),
        },
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(values, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )

    geopandas.clear_dispatch_events()
    clear_rewrite_events()
    dissolved = frame.dissolve(
        by="group",
        aggfunc={"value": "sum"},
        method="unary",
    ).reset_index()
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    hulls = dissolved.geometry.convex_hull
    runtime_reasons = {event.reason for event in get_d2h_transfer_events(clear=True)}
    actual = np.asarray(hulls.array, dtype=object)
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    rewrite_events = get_rewrite_events(clear=True)

    expected = [
        shapely.convex_hull(shapely.union_all(np.asarray(values[:12], dtype=object))),
        shapely.convex_hull(shapely.union_all(np.asarray(values[12:], dtype=object))),
    ]

    assert any(
        event.implementation == "grouped_dissolve_convex_hull_gpu" for event in dispatch_events
    )
    assert any(
        event.rule_name == "R12_dissolve_grouped_union_to_grouped_convex_hull"
        for event in rewrite_events
    )
    assert "grouped convex-hull nonempty-group scalar fence" not in runtime_reasons
    assert "grouped convex-hull source-row group-domain scalar fence" not in runtime_reasons
    assert shapely.area(shapely.symmetric_difference(actual[0], expected[0])) == 0.0
    assert shapely.area(shapely.symmetric_difference(actual[1], expected[1])) == 0.0


@pytest.mark.gpu
def test_public_dissolve_defers_grouped_union_until_exact_geometry_consumer() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    values = [
        *[box(float(i) * 0.45, 0.0, float(i) * 0.45 + 1.0, 1.0) for i in range(20)],
        *[box(20.0 + float(i) * 0.45, 0.0, 21.0 + float(i) * 0.45, 1.0) for i in range(20)],
    ]
    frame = geopandas.GeoDataFrame(
        {
            "group": np.repeat(np.asarray([0, 1], dtype=np.int32), 20),
            "value": np.arange(len(values), dtype=np.int32),
        },
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(values, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )

    geopandas.clear_dispatch_events()
    dissolved = frame.dissolve(by="group", aggfunc={"value": "sum"}, method="unary")
    dissolve_events = geopandas.get_dispatch_events(clear=True)

    assert any(
        event.implementation == "lazy_native_grouped_union_carrier" for event in dissolve_events
    )
    assert not any(
        event.surface == "vibespatial.overlay.dissolve.execute_native_grouped_union"
        and event.operation == "grouped_union"
        for event in dissolve_events
    )

    lazy_owned = dissolved.geometry.values.cached_owned()
    assert getattr(lazy_owned, "_is_lazy_grouped_union_owned", False)
    assert lazy_owned._materialized_owned is None
    valid_geometry = dissolved.geometry.make_valid()
    make_valid_events = geopandas.get_dispatch_events(clear=True)
    valid_owned = valid_geometry.values.cached_owned()
    assert getattr(valid_owned, "_is_lazy_grouped_union_owned", False)
    assert valid_owned._materialized_owned is None
    assert lazy_owned._materialized_owned is None
    assert any(
        event.surface == "geopandas.array.make_valid"
        and event.implementation == "trusted_deferred_ogc_valid_no_repair"
        for event in make_valid_events
    )
    assert not any(
        event.surface == "vibespatial.overlay.dissolve.LazyGroupedUnionOwned"
        and event.operation == "materialize_grouped_union"
        for event in make_valid_events
    )

    actual_area = np.asarray(valid_geometry.area, dtype=np.float64)
    materialize_events = geopandas.get_dispatch_events(clear=True)
    expected_area = np.asarray(
        [
            shapely.area(shapely.union_all(np.asarray(values[:20], dtype=object))),
            shapely.area(shapely.union_all(np.asarray(values[20:], dtype=object))),
        ],
        dtype=np.float64,
    )

    assert np.allclose(actual_area, expected_area)
    assert any(
        event.surface == "vibespatial.overlay.dissolve.LazyGroupedUnionOwned"
        and event.operation == "materialize_grouped_union"
        for event in materialize_events
    )


def test_public_dissolve_matches_public_union_all_component_order_for_nybb_fixture() -> None:
    path = Path("tests/upstream/geopandas/tests/data/nybb_16a.zip")
    frame = geopandas.read_file(path)
    frame = frame[["geometry", "BoroName", "BoroCode"]]
    frame = frame.rename(columns={"geometry": "myshapes"})
    frame = frame.set_geometry("myshapes")
    frame["manhattan_bronx"] = 5
    frame.loc[3:4, "manhattan_bronx"] = 6

    dissolved = frame.dissolve("manhattan_bronx")
    expected = frame.loc[0:2].geometry.union_all()

    # The dissolve exact path should preserve the same MultiPolygon component
    # ordering as the standalone public union_all() path for the grouped input.
    assert shapely.to_wkb(dissolved.loc[5, "myshapes"]) == shapely.to_wkb(expected)


def test_evaluate_geopandas_dissolve_routes_small_buffered_line_unary_to_grouped_union() -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(64, dtype=np.int32),
            "value": np.arange(64, dtype=np.int32),
            "geometry": [LineString([(float(i), 0.0), (float(i), 10.0)]) for i in range(64)],
        },
        crs="EPSG:3857",
    )
    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.5)

    clear_rewrite_events()
    geopandas.clear_fallback_events()
    actual = evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)

    expected = evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="disjoint_subset",
        grid_size=None,
        agg_kwargs={},
    )

    assert not any(
        event.rule_name == "R8_dissolve_buffered_lines_to_disjoint_subset"
        for event in rewrite_events
    )
    if has_gpu_runtime():
        assert any(
            event.rule_name == "R12_dissolve_buffered_lines_grouped_union"
            for event in rewrite_events
        )
        assert not any(
            event.surface == "geopandas.geodataframe.dissolve"
            and "declined the native grouped carrier" in event.reason
            for event in fallback_events
        )
    else:
        assert any(
            event.surface == "geopandas.geodataframe.dissolve"
            and "declined the native grouped carrier" in event.reason
            for event in fallback_events
        )
    actual_geom = np.asarray(actual.geometry.array, dtype=object)[0]
    expected_geom = np.asarray(expected.geometry.array, dtype=object)[0]
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0
    assert actual.iloc[0]["value"] == expected.iloc[0]["value"]


def test_duplicate_two_point_buffered_line_dissolve_promotes_to_grouped_union() -> None:
    lines = [
        LineString([(0.0, 0.0), (10.0, 0.0)]),
        LineString([(10.0, 0.0), (0.0, 0.0)]),
        LineString([(0.0, 5.0), (10.0, 5.0)]),
        LineString([(10.0, 5.0), (0.0, 5.0)]),
    ] * 32
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(len(lines), dtype=np.int32),
            "value": np.arange(len(lines), dtype=np.int32),
            "geometry": lines,
        },
        crs="EPSG:3857",
    )
    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.5)

    clear_rewrite_events()
    geopandas.clear_fallback_events()
    actual = evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)

    assert not any(
        event.rule_name == "R9_dissolve_buffered_two_point_lines_exact_union"
        for event in rewrite_events
    )
    if has_gpu_runtime():
        assert any(
            event.rule_name == "R12_dissolve_buffered_lines_grouped_union"
            for event in rewrite_events
        )
        assert not any(
            event.surface == "geopandas.geodataframe.dissolve"
            and "declined the native grouped carrier" in event.reason
            for event in fallback_events
        )
    else:
        assert any(
            event.surface == "geopandas.geodataframe.dissolve"
            and "declined the native grouped carrier" in event.reason
            for event in fallback_events
        )
    actual_geom = np.asarray(actual.geometry.array, dtype=object)[0]
    expected_geom = shapely.union_all(np.asarray(buffered.geometry.array, dtype=object))
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0


def test_evaluate_geopandas_dissolve_rewrites_duplicate_two_point_buffered_lines_to_exact_gpu_union(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        return

    lines = [
        LineString([(0.0, 0.0), (10.0, 0.0)]),
        LineString([(10.0, 0.0), (0.0, 0.0)]),
        LineString([(0.0, 5.0), (10.0, 5.0)]),
        LineString([(10.0, 5.0), (0.0, 5.0)]),
    ] * 32
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(len(lines), dtype=np.int32),
            "value": np.arange(len(lines), dtype=np.int32),
        },
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(lines, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )
    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.5)

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "duplicate two-point buffered-line dissolve should bypass the generic grouped union path"
        )

    monkeypatch.setattr(dissolve_module, "execute_grouped_union_codes", _fail)

    clear_rewrite_events()
    geopandas.clear_fallback_events()
    actual = evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)

    expected = geopandas.GeoDataFrame(
        {
            "geometry": [shapely.union_all(np.asarray(buffered.geometry.array, dtype=object))],
            "value": [0],
        },
        geometry="geometry",
        index=pd.Index([0], name="group"),
        crs=buffered.crs,
    )

    assert any(
        event.rule_name == "R9_dissolve_buffered_two_point_lines_exact_union"
        for event in rewrite_events
    )
    assert not any(
        event.surface == "geopandas.geodataframe.dissolve"
        and "declined the native grouped carrier" in event.reason
        for event in fallback_events
    )
    actual_geom = np.asarray(actual.geometry.array, dtype=object)[0]
    expected_geom = np.asarray(expected.geometry.array, dtype=object)[0]
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0
    assert actual.iloc[0]["value"] == expected.iloc[0]["value"]


def test_dedupe_two_point_linestring_rows_prefers_host_metadata_when_available(
    monkeypatch,
) -> None:
    lines = [
        LineString([(0.0, 0.0), (10.0, 0.0)]),
        LineString([(10.0, 0.0), (0.0, 0.0)]),
        LineString([(0.0, 5.0), (10.0, 5.0)]),
        LineString([(10.0, 5.0), (0.0, 5.0)]),
        LineString([(2.0, 2.0), (2.0, 8.0)]),
    ] * 8
    owned = from_shapely_geometries(lines)
    owned._ensure_host_state()

    if dissolve_module.cp is not None:

        def _fail(*_args, **_kwargs):
            raise AssertionError("host-materialized dedupe should not call device-side lexsort")

        monkeypatch.setattr(dissolve_module.cp, "lexsort", _fail)

    unique_rows = dissolve_module._dedupe_two_point_linestring_rows_gpu(owned)

    assert unique_rows is not None
    deduped = np.asarray(
        owned.take(unique_rows.astype(np.int64, copy=False)).to_shapely(), dtype=object
    )
    assert deduped.shape == (3,)

    def _normalized_endpoints(geom) -> tuple[tuple[float, float], tuple[float, float]]:
        start = (float(geom.coords[0][0]), float(geom.coords[0][1]))
        end = (float(geom.coords[-1][0]), float(geom.coords[-1][1]))
        return (start, end) if start <= end else (end, start)

    assert {_normalized_endpoints(geom) for geom in deduped} == {
        ((0.0, 0.0), (10.0, 0.0)),
        ((0.0, 5.0), (10.0, 5.0)),
        ((2.0, 2.0), (2.0, 8.0)),
    }


def test_buffered_two_point_line_exact_union_rewrite_accepts_large_deduped_sets(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        return

    unique_lines = [LineString([(0.0, float(i) * 4.0), (20.0, float(i) * 4.0)]) for i in range(66)]
    lines = unique_lines * 16
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(len(lines), dtype=np.int32),
            "value": np.arange(len(lines), dtype=np.int32),
        },
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(lines, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )
    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.5)

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "large deduped buffered-line dissolve should bypass the generic grouped union path"
        )

    monkeypatch.setattr(dissolve_module, "_BUFFERED_LINE_EXACT_CPU_MAX_ROWS", 1)
    monkeypatch.setattr(dissolve_module, "execute_grouped_union_codes", _fail)

    clear_rewrite_events()
    actual = evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)

    assert any(
        event.rule_name == "R9_dissolve_buffered_two_point_lines_exact_union"
        for event in rewrite_events
    )
    assert getattr(actual.geometry.array, "_owned", None) is not None
    actual_geom = np.asarray(actual.geometry.array, dtype=object)[0]
    expected_geom = shapely.union_all(np.asarray(buffered.geometry.array, dtype=object))
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0


def test_buffered_two_point_line_exact_union_orders_crossed_lines_for_cascaded_union() -> None:
    if not has_gpu_runtime():
        return

    horizontal = [
        LineString([(0.0, float(i)), (20.0, float(i))])
        for i in range(12)
    ]
    vertical = [
        LineString([(float(i), 0.0), (float(i), 20.0)])
        for i in range(12)
    ]
    lines = [line for pair in zip(horizontal, vertical, strict=True) for line in pair] * 8
    frame = geopandas.GeoDataFrame(
        {"group": np.zeros(len(lines), dtype=np.int32)},
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(lines, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )
    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.75)

    actual = evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    actual_geom = np.asarray(actual.geometry.array, dtype=object)[0]
    expected_geom = shapely.union_all(np.asarray(buffered.geometry.array, dtype=object))
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == pytest.approx(
        0.0,
        abs=1.0e-12,
    )


def test_buffered_two_point_line_rewrite_has_bounded_device_grouped_union_shape() -> None:
    source = Path(dissolve_module.__file__).read_text()
    function_start = source.index("def _maybe_execute_buffered_two_point_line_exact_union_rewrite(")
    function_end = source.index("\ndef ", function_start + 1)
    function_source = source[function_start:function_end]

    assert "device_source_rowset_to_grouped_union" in function_source
    assert "source_owned.device_take" in function_source
    assert "_order_two_point_lines_for_cascaded_union_gpu" in function_source
    assert function_source.count("_reduce_buffered_line_polygons_gpu(") == 1
    assert "MAX_UNIQUE_ROWS" not in function_source
    assert "unique_owned.row_count >" not in function_source
    assert "overlay_device_to_host" not in function_source
    assert "_greedy_bbox_disjoint_coloring" not in source
    assert "_reorder_small_partial_union_groups_by_overlap" not in source
    assert "_reduce_partial_rows_gpu" not in source
    assert "cp.lexsort" not in source
    assert "cp.stack(tuple(sort_keys))" not in source
    assert "d_observed_codes.astype(cp.float64" not in source

    dedupe_start = source.index("def _dedupe_two_point_linestring_rows_gpu(")
    dedupe_end = source.index("\ndef ", dedupe_start + 1)
    dedupe_source = source[dedupe_start:dedupe_end]
    assert "certify_two_point_linestring_layout_device" in dedupe_source
    assert "overlay_bool_scalar" not in dedupe_source

    reducer_start = source.index("def _reduce_buffered_line_polygons_gpu(")
    reducer_end = source.index("\ndef ", reducer_start + 1)
    reducer_source = source[reducer_start:reducer_end]
    assert reducer_source.count("single_group_polygon_collective_union_gpu(") == 1
    assert "while True:" not in reducer_source
    assert "if buffered.row_count <= 1:" in reducer_source
    assert "d_direction_partitions" in reducer_source
    assert "cp.flatnonzero(" not in reducer_source
    assert "_regroup_polygonal_parts_with_grouped_union_gpu(" not in source
    assert "except Exception" not in reducer_source
    assert "_tree_reduce_global" not in reducer_source
    assert "_gpu_union_group" not in source
    assert "_BUFFERED_LINE_GROUPED_UNION_FAN_IN" not in source
    assert "_BUFFERED_LINE_AGGREGATE_UNION_FAN_IN" not in source

    from vibespatial.constructive import tiled_union as tiled_union_module

    tiled_source = Path(tiled_union_module.__file__).read_text()
    assert "_DIRECT_COLLECTIVE_SEGMENT_PEER_PRESSURE" in tiled_source
    assert "_build_topology_tile_candidate_relation" in tiled_source
    assert "NativeRelation(" in tiled_source
    assert "_clip_topology_tile_candidate_batch" in tiled_source
    assert "_TOPOLOGY_CONSTRUCTIVE_BATCH_TILES = 32" in tiled_source
    assert "_reduce_topology_tile_coverage" in tiled_source
    assert "_TOPOLOGY_SEAM_FAN_IN" in tiled_source
    assert "_physicalize_topology_coverage_output" in tiled_source
    assert "device_physicalize_owned_row_selections_exact" in tiled_source
    assert "collective union emitted topology exact-allocation packet" in tiled_source
    assert "else:\n            coverage_parts.append(batch_result)" in tiled_source
    assert "_regroup_native_grouped_parts_with_grouped_union_gpu" in tiled_source
    assert "cp.tile(" not in tiled_source
    assert "cp.repeat(" not in tiled_source
    assert "shapely" not in tiled_source
    assert ".to_shapely(" not in tiled_source
    assert ".get(" not in tiled_source

    grouped_start = source.index("def _maybe_execute_buffered_line_grouped_union_rewrite(")
    grouped_end = source.index("\ndef ", grouped_start + 1)
    grouped_source = source[grouped_start:grouped_end]
    assert "except Exception" not in grouped_source


def test_lazy_grouped_union_propagates_admitted_exact_materializer_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_owned = from_shapely_geometries([box(0.0, 0.0, 1.0, 1.0), box(1.0, 0.0, 2.0, 1.0)])
    grouped = NativeGrouped.from_dense_codes(
        np.asarray([0, 0], dtype=np.int32),
        group_count=1,
    )

    def _fail_exact():
        raise RuntimeError("admitted grouped topology failed")

    monkeypatch.setattr(
        dissolve_module,
        "execute_native_grouped_union",
        lambda *_args, **_kwargs: pytest.fail(
            "an admitted exact materializer failure must not switch algorithms"
        ),
    )
    lazy = dissolve_module.LazyGroupedUnionOwned(
        source_owned=source_owned,
        grouped=grouped,
        geometries=np.asarray(source_owned.to_shapely(), dtype=object),
        method=DissolveUnionMethod.UNARY,
        grid_size=None,
        geometry_name="geometry",
        crs=None,
        exact_materializer=_fail_exact,
    )

    with pytest.raises(RuntimeError, match="admitted grouped topology failed"):
        lazy.to_owned()


def test_lazy_grouped_union_make_valid_consumes_semantic_proof_without_materializing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibespatial.constructive.make_valid_pipeline import make_valid_owned

    source_owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), box(1.0, 0.0, 2.0, 1.0)]
    )
    grouped = NativeGrouped.from_dense_codes(
        np.asarray([0, 0], dtype=np.int32),
        group_count=1,
    )

    lazy = dissolve_module.LazyGroupedUnionOwned(
        source_owned=source_owned,
        grouped=grouped,
        geometries=np.asarray(source_owned.to_shapely(), dtype=object),
        method=DissolveUnionMethod.UNARY,
        grid_size=None,
        geometry_name="geometry",
        crs=None,
        exact_materializer=lambda: pytest.fail(
            "make_valid must consume the grouped-union validity proof"
        ),
    )
    monkeypatch.setattr(
        dissolve_module,
        "execute_native_grouped_union",
        lambda *_args, **_kwargs: pytest.fail(
            "make_valid must not execute deferred grouped topology"
        ),
    )

    result = make_valid_owned(owned=lazy)

    assert result.owned is lazy
    assert result.repaired_rows.size == 0
    assert lazy._materialized_owned is None


def test_device_grouped_union_repair_uses_atomic_native_carrier() -> None:
    source = Path(dissolve_module.__file__).read_text()
    helper_start = source.index("def _repair_grouped_union_invalid_subset_device(")
    helper_end = source.index("\ndef ", helper_start + 1)
    helper_source = source[helper_start:helper_end]

    assert "gpu_repair_invalid_polygons(" in helper_source
    assert "make_valid_owned(" not in helper_source
    assert ".device_take(" not in helper_source
    assert "device_concat_owned_scatter(" not in helper_source
    assert ".to_shapely(" not in helper_source

    native_start = source.index("def _execute_device_native_grouped_union(")
    native_end = source.index("\ndef ", native_start + 1)
    native_source = source[native_start:native_end]
    assert "device native grouped union failed" not in native_source


def test_evaluate_geopandas_dissolve_does_not_rewrite_polygon_buffer_unary() -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(64, dtype=np.int32),
            "value": np.arange(64, dtype=np.int32),
            "geometry": [box(float(i), 0.0, float(i) + 0.5, 0.5) for i in range(64)],
        },
        crs="EPSG:3857",
    )
    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.25)

    clear_rewrite_events()
    evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)

    assert not any(
        event.rule_name == "R8_dissolve_buffered_lines_to_disjoint_subset"
        for event in rewrite_events
    )


@pytest.mark.gpu
def test_device_backed_buffered_line_dissolve_preserves_owned_backing() -> None:
    lines = [LineString([(float(i), 0.0), (float(i), 10.0)]) for i in range(64)]
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(64, dtype=np.int32),
            "value": np.arange(64, dtype=np.int32),
        },
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(lines, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )

    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.5)
    assert getattr(buffered.geometry.values, "_provenance", None) is not None

    clear_rewrite_events()
    geopandas.clear_fallback_events()
    result = evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)

    assert len(result) == 1
    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert getattr(result.geometry.values, "_owned", None) is not None
    assert any(
        event.rule_name == "R9_dissolve_buffered_two_point_lines_exact_union"
        for event in rewrite_events
    )
    assert not any(
        event.surface == "geopandas.geodataframe.dissolve"
        and "declined the native grouped carrier" in event.reason
        for event in fallback_events
    )
    actual_geom = np.asarray(result.geometry.array, dtype=object)[0]
    expected_geom = shapely.union_all(np.asarray(buffered.geometry.array, dtype=object))
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0


@pytest.mark.gpu
def test_device_backed_buffered_line_dissolve_strict_native_uses_exact_gpu_rewrite() -> None:
    lines = [LineString([(float(i), 0.0), (float(i), 10.0)]) for i in range(64)]
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(64, dtype=np.int32),
            "value": np.arange(64, dtype=np.int32),
        },
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(lines, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )

    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.5)
    assert getattr(buffered.geometry.values, "_provenance", None) is not None

    clear_rewrite_events()
    geopandas.clear_fallback_events()
    with strict_native_environment():
        result = evaluate_geopandas_dissolve(
            buffered,
            by="group",
            aggfunc="first",
            as_index=True,
            level=None,
            sort=True,
            observed=False,
            dropna=True,
            method="unary",
            grid_size=None,
            agg_kwargs={},
        )
    rewrite_events = get_rewrite_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)

    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert any(
        event.rule_name == "R9_dissolve_buffered_two_point_lines_exact_union"
        for event in rewrite_events
    )
    assert not any(
        event.surface == "geopandas.geodataframe.dissolve"
        and "declined the native grouped carrier" in event.reason
        for event in fallback_events
    )


@pytest.mark.gpu
def test_device_backed_multivertex_buffered_line_dissolve_defers_grouped_union_carrier() -> None:
    lines = [
        LineString(
            [
                (0.0, float(i)),
                (5.0, float(i) + 0.25),
                (10.0, float(i)),
                (15.0, float(i) + 0.25),
            ]
        )
        for i in range(64)
    ]
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(64, dtype=np.int32),
            "value": np.arange(64, dtype=np.int32),
        },
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(lines, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )

    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.5)

    geopandas.clear_dispatch_events()
    clear_rewrite_events()
    geopandas.clear_fallback_events()
    result = evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)

    assert any(
        event.implementation == "lazy_native_grouped_union_carrier" for event in dispatch_events
    )
    assert not any(
        event.rule_name == "R12_dissolve_buffered_lines_grouped_union" for event in rewrite_events
    )
    assert getattr(result.geometry.values._owned, "_is_lazy_grouped_union_owned", False)
    assert not any(
        event.surface == "geopandas.geodataframe.dissolve"
        and "declined the native grouped carrier" in event.reason
        for event in fallback_events
    )
    actual_geom = np.asarray(result.geometry.array, dtype=object)[0]
    expected_geom = shapely.union_all(np.asarray(buffered.geometry.array, dtype=object))
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0


def test_grouped_union_geometry_result_prefers_owned_make_valid_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validity_module = importlib.import_module("vibespatial.constructive.validity")
    make_valid_module = importlib.import_module("vibespatial.constructive.make_valid_pipeline")

    original_owned = from_shapely_geometries([box(0.0, 0.0, 1.0, 1.0)])
    repaired_owned = from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)])

    def _fake_is_valid_owned(arg, *args, **kwargs):
        return np.asarray([arg is repaired_owned], dtype=bool)

    def _fake_make_valid_owned(*, owned, dispatch_mode, **kwargs):
        assert owned is original_owned
        return SimpleNamespace(
            owned=repaired_owned,
            geometries=np.asarray([box(0.0, 0.0, 2.0, 2.0)], dtype=object),
            repaired_rows=np.asarray([0], dtype=np.int32),
            selected=ExecutionMode.GPU,
        )

    monkeypatch.setattr(validity_module, "is_valid_owned", _fake_is_valid_owned)
    monkeypatch.setattr(make_valid_module, "make_valid_owned", _fake_make_valid_owned)

    grouped = dissolve_module.GroupedUnionResult(
        geometries=None,
        group_count=1,
        non_empty_groups=1,
        empty_groups=0,
        method=DissolveUnionMethod.UNARY,
        owned=original_owned,
    )

    result = dissolve_module._grouped_union_geometry_result(
        grouped,
        geometry_name="geometry",
        crs="EPSG:3857",
    )

    assert result.owned is repaired_owned
    assert result.series is None

    frame = geopandas.GeoDataFrame(
        {"value": [10], "geometry": [box(0.0, 0.0, 1.0, 1.0)]},
        geometry="geometry",
        crs="EPSG:3857",
    )
    payload = dissolve_module._grouped_constructive_result(
        grouped,
        frame=frame,
        aggregated_data=pd.DataFrame(
            {"value": [10]},
            index=pd.Index(["a"], name="group"),
        ),
        as_index=True,
    )

    assert payload.provenance is not None
    assert payload.provenance.operation == "grouped_unary_union_repair"
    assert payload.provenance.source_rows.tolist() == [0]


def test_grouped_constructive_result_carries_output_group_provenance() -> None:
    from vibespatial.api._native_result_core import NativeGeometryProvenance

    frame = geopandas.GeoDataFrame(
        {
            "group": ["a", "b"],
            "value": [10, 20],
            "geometry": [box(0.0, 0.0, 1.0, 1.0), box(2.0, 0.0, 3.0, 1.0)],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    grouped = dissolve_module.GroupedUnionResult(
        geometries=None,
        group_count=2,
        non_empty_groups=2,
        empty_groups=0,
        method=DissolveUnionMethod.UNARY,
        owned=from_shapely_geometries(list(frame.geometry)),
    )

    payload = dissolve_module._grouped_constructive_result(
        grouped,
        frame=frame,
        aggregated_data=pd.DataFrame(
            {"value": [10, 20]},
            index=pd.Index(["a", "b"], name="group"),
        ),
        as_index=True,
    )

    assert isinstance(payload.provenance, NativeGeometryProvenance)
    assert payload.provenance.operation == "grouped_unary_union"
    assert payload.provenance.source_rows.tolist() == [0, 1]
    assert payload.geometry_metadata is not None


def test_grouped_constructive_result_converts_host_union_output_to_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": ["a", "b"],
            "value": [10, 20],
            "geometry": [box(0.0, 0.0, 1.0, 1.0), box(2.0, 0.0, 3.0, 1.0)],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    grouped = dissolve_module.GroupedUnionResult(
        geometries=np.asarray(
            [box(0.0, 0.0, 1.0, 1.0), box(2.0, 0.0, 3.0, 1.0)],
            dtype=object,
        ),
        group_count=2,
        non_empty_groups=2,
        empty_groups=0,
        method=DissolveUnionMethod.UNARY,
        owned=None,
    )

    monkeypatch.setattr(
        "vibespatial.api.geoseries.GeoSeries",
        lambda *args, **kwargs: pytest.fail(
            "supported grouped union outputs should enter NativeTabularResult as owned geometry"
        ),
    )

    payload = dissolve_module._grouped_constructive_result(
        grouped,
        frame=frame,
        aggregated_data=pd.DataFrame(
            {"value": [10, 20]},
            index=pd.Index(["a", "b"], name="group"),
        ),
        as_index=True,
    )

    assert payload.geometry.owned is not None
    assert payload.geometry.series is None
    assert payload.geometry_metadata is not None


def test_grouped_union_geometry_result_strict_native_raises_on_host_repair_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validity_module = importlib.import_module("vibespatial.constructive.validity")
    make_valid_module = importlib.import_module("vibespatial.constructive.make_valid_pipeline")

    owned = from_shapely_geometries([box(0.0, 0.0, 1.0, 1.0)])

    def _fake_is_valid_owned(*args, **kwargs):
        return np.asarray([False], dtype=bool)

    def _fake_make_valid_owned(*, owned, dispatch_mode, **kwargs):
        return SimpleNamespace(
            owned=None,
            geometries=np.asarray([box(0.0, 0.0, 1.0, 1.0)], dtype=object),
            repaired_rows=np.asarray([0], dtype=np.int32),
            selected=ExecutionMode.CPU,
        )

    monkeypatch.setattr(validity_module, "is_valid_owned", _fake_is_valid_owned)
    monkeypatch.setattr(make_valid_module, "make_valid_owned", _fake_make_valid_owned)

    grouped = dissolve_module.GroupedUnionResult(
        geometries=None,
        group_count=1,
        non_empty_groups=1,
        empty_groups=0,
        method=DissolveUnionMethod.UNARY,
        owned=owned,
    )

    with strict_native_environment(), pytest.raises(StrictNativeFallbackError):
        dissolve_module._grouped_union_geometry_result(
            grouped,
            geometry_name="geometry",
            crs="EPSG:3857",
        )


def test_grouped_union_geometry_result_make_valid_fallback_extracts_polygonal_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validity_module = importlib.import_module("vibespatial.constructive.validity")
    make_valid_module = importlib.import_module("vibespatial.constructive.make_valid_pipeline")

    owned = from_shapely_geometries([box(0.0, 0.0, 1.0, 1.0)])
    repaired = GeometryCollection(
        [
            box(0.0, 0.0, 2.0, 2.0),
            LineString([(0.0, 0.0), (2.0, 2.0)]),
        ]
    )

    monkeypatch.setattr(
        validity_module,
        "is_valid_owned",
        lambda *_args, **_kwargs: np.asarray([False], dtype=bool),
    )
    monkeypatch.setattr(
        make_valid_module,
        "make_valid_owned",
        lambda **_kwargs: SimpleNamespace(
            owned=None,
            geometries=np.asarray([repaired], dtype=object),
            repaired_rows=np.asarray([0], dtype=np.int32),
            selected=ExecutionMode.CPU,
        ),
    )

    grouped = dissolve_module.GroupedUnionResult(
        geometries=None,
        group_count=1,
        non_empty_groups=1,
        empty_groups=0,
        method=DissolveUnionMethod.UNARY,
        owned=owned,
    )

    result = dissolve_module._grouped_union_geometry_result(
        grouped,
        geometry_name="geometry",
        crs="EPSG:3857",
    )

    assert result.series is not None
    assert result.series.iloc[0].geom_type == "Polygon"
    assert result.series.iloc[0].equals(box(0.0, 0.0, 2.0, 2.0))


def test_execute_grouped_union_codes_recomputes_invalid_gpu_rows_from_original_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segmented_union_module = importlib.import_module(
        "vibespatial.kernels.constructive.segmented_union"
    )

    geoms = [box(0.0, 0.0, 1.0, 1.0), box(1.0, 0.0, 2.0, 1.0)]
    owned = from_shapely_geometries(geoms)
    invalid_union = shapely.Polygon([(0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 1.0), (0.0, 0.0)])

    monkeypatch.setattr(
        segmented_union_module,
        "segmented_union_all",
        lambda *_args, **_kwargs: from_shapely_geometries([invalid_union]),
    )

    grouped = dissolve_module.execute_grouped_union_codes(
        np.asarray(geoms, dtype=object),
        np.asarray([0, 0], dtype=np.int64),
        group_count=1,
        method=DissolveUnionMethod.UNARY,
        owned=owned,
    )

    assert grouped is not None
    assert grouped.owned is not None
    actual = grouped.owned.to_shapely()[0]
    expected = shapely.union_all(np.asarray(geoms, dtype=object))
    assert actual.geom_type == expected.geom_type
    assert shapely.equals(actual, expected)


def test_execute_grouped_box_union_gpu_owned_codes_builds_device_backed_coverage_rectangles() -> (
    None
):
    if not has_gpu_runtime():
        return

    geometry_array = GeometryArray.from_owned(
        from_shapely_geometries(
            [
                box(0, 0, 1, 1),
                box(1, 0, 2, 1),
                box(10, 10, 11, 11),
                box(11, 10, 12, 11),
            ]
        )
    )
    owned = getattr(geometry_array, "_owned", None)
    assert owned is not None

    grouped = dissolve_module.execute_grouped_box_union_gpu_owned_codes(
        pd.Index([0, 0, 1, 1], dtype="int32").to_numpy(),
        group_count=2,
        owned=owned,
    )

    assert grouped is not None
    assert grouped.group_count == 2
    assert grouped.non_empty_groups == 2
    assert grouped.owned is not None
    assert grouped.geometries is None
    actual = np.asarray(grouped.owned.to_shapely(), dtype=object)
    expected = np.asarray([box(0, 0, 2, 1), box(10, 10, 12, 11)], dtype=object)
    assert all(
        actual_geom.equals(expected_geom)
        for actual_geom, expected_geom in zip(actual, expected, strict=True)
    )


def test_execute_grouped_box_union_gpu_owned_codes_accepts_fractional_rectangles() -> None:
    if not has_gpu_runtime():
        return

    geometry_array = GeometryArray.from_owned(
        from_shapely_geometries(
            [
                box(0.0, 0.0, 0.5, 1.0),
                box(0.5, 0.0, 1.0, 1.0),
                box(2.0, 0.0, 2.5, 1.0),
                box(2.5, 0.0, 3.0, 1.0),
            ]
        )
    )
    owned = getattr(geometry_array, "_owned", None)
    assert owned is not None

    grouped = dissolve_module.execute_grouped_box_union_gpu_owned_codes(
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        group_count=2,
        owned=owned,
    )

    assert grouped is not None
    assert grouped.owned is not None
    assert grouped.geometries is None


def test_execute_grouped_box_union_gpu_owned_codes_accepts_unsorted_codes(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        return

    cp = pytest.importorskip("cupy")
    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            box(10, 10, 11, 11),
            box(1, 0, 2, 1),
            box(11, 10, 12, 11),
        ],
        residency=Residency.DEVICE,
    )
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None

    def _fail_host_metadata(*_args, **_kwargs):
        raise AssertionError("unsorted owned coverage reducer should stay device-native")

    monkeypatch.setattr(type(owned), "_ensure_host_metadata", _fail_host_metadata)

    grouped = dissolve_module.execute_grouped_box_union_gpu_owned_codes(
        cp.asarray([0, 1, 0, 1], dtype=cp.int32),
        group_count=2,
        owned=owned,
    )

    assert grouped is not None
    assert grouped.owned is not None
    assert grouped.geometries is None
    assert grouped.owned._validity is None
    assert grouped.owned._tags is None
    assert grouped.owned._family_row_offsets is None
    monkeypatch.undo()
    actual = np.asarray(grouped.owned.to_shapely(), dtype=object)
    expected = np.asarray([box(0, 0, 2, 1), box(10, 10, 12, 11)], dtype=object)
    assert shapely.equals(actual, expected).tolist() == [True, True]


def test_execute_grouped_box_union_gpu_owned_codes_rejects_gapped_groups() -> None:
    if not has_gpu_runtime():
        return

    geometry_array = GeometryArray.from_owned(
        from_shapely_geometries(
            [
                box(0.0, 0.0, 1.0, 1.0),
                box(0.0, 2.0, 1.0, 3.0),
            ]
        )
    )
    owned = getattr(geometry_array, "_owned", None)
    assert owned is not None

    grouped = dissolve_module.execute_grouped_box_union_gpu_owned_codes(
        np.asarray([0, 0], dtype=np.int32),
        group_count=1,
        owned=owned,
    )

    assert grouped is None


def test_execute_grouped_union_codes_linestrings_skip_owned_segmented_union_fast_path(
    monkeypatch,
) -> None:
    geometry_array = GeometryArray.from_owned(
        from_shapely_geometries(
            [
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (2, 2)]),
            ]
        )
    )
    owned = getattr(geometry_array, "_owned", None)
    assert owned is not None

    import vibespatial.kernels.constructive.segmented_union as segmented_union_module

    def _fail(*_args, **_kwargs):
        raise AssertionError("linestring dissolve should not route through segmented_union_all")

    monkeypatch.setattr(segmented_union_module, "segmented_union_all", _fail)

    grouped = dissolve_module.execute_grouped_union_codes(
        geometry_array,
        pd.Index([0, 0], dtype="int32").to_numpy(),
        group_count=1,
        method="unary",
        owned=owned,
    )

    assert grouped is None


def test_join_heavy_synthetic_groups_match_between_coverage_and_unary() -> None:
    frame = _regular_polygons_frame(256)
    frame["group"] = pd.Categorical(np.arange(len(frame), dtype=np.int32) % 128)

    coverage = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=False,
        observed=False,
        dropna=True,
        method="coverage",
        grid_size=None,
        agg_kwargs={},
    )
    unary = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert_geodataframe_equal(unary, coverage)


def test_join_heavy_direct_grouped_dissolve_matches_public_coverage_dissolve(
    monkeypatch,
) -> None:
    frame = _regular_polygons_frame(256)
    unique_right_index = np.arange(len(frame), dtype=np.int64)
    calls: list[DissolveUnionMethod] = []
    real_codes = dissolve_module.execute_grouped_union_codes

    def _count_codes(*args, **kwargs):
        calls.append(DissolveUnionMethod(kwargs["method"]))
        return real_codes(*args, **kwargs)

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "join-heavy benchmark helper should not fall back to public dissolve here"
        )

    monkeypatch.setattr("vibespatial.bench.pipeline.execute_grouped_union_codes", _count_codes)
    monkeypatch.setattr("vibespatial.bench.pipeline.evaluate_geopandas_dissolve", _fail)

    dissolved, used_direct = _dissolve_join_heavy_groups(
        frame.geometry,
        unique_right_index,
        scale=len(frame),
    )
    expected = evaluate_geopandas_dissolve(
        geopandas.GeoDataFrame(
            {
                "group": pd.Categorical(unique_right_index % 128),
                "geometry": frame.geometry,
            },
            geometry="geometry",
            crs=frame.crs,
        ),
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=False,
        observed=False,
        dropna=True,
        method="coverage",
        grid_size=None,
        agg_kwargs={},
    )

    assert used_direct is True
    assert calls == [DissolveUnionMethod.COVERAGE]
    assert isinstance(dissolved, NativeTabularResult)
    assert_geodataframe_equal(dissolved.to_geodataframe(), expected)


@pytest.mark.gpu
def test_join_heavy_device_grouped_dissolve_matches_public_coverage_dissolve() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    frame = _regular_polygons_frame(256)
    owned = from_shapely_geometries(list(frame.geometry), residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    unique_right_index = cp.arange(len(frame), dtype=cp.int64)

    dissolved, used_direct = _dissolve_join_heavy_groups(
        GeometryNativeResult.from_owned(owned, crs=frame.crs),
        unique_right_index,
        scale=len(frame),
    )
    expected = evaluate_geopandas_dissolve(
        geopandas.GeoDataFrame(
            {
                "group": pd.Categorical(np.arange(len(frame), dtype=np.int64) % 128),
                "geometry": frame.geometry,
            },
            geometry="geometry",
            crs=frame.crs,
        ),
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=False,
        observed=False,
        dropna=True,
        method="coverage",
        grid_size=None,
        agg_kwargs={},
    )

    assert used_direct is True
    assert isinstance(dissolved, NativeTabularResult)
    assert_geodataframe_equal(dissolved.to_geodataframe(), expected)
