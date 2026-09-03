from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

import vibespatial
from benchmarks.spatialbench import sf100_evidence
from benchmarks.spatialbench.vibespatial_queries import VibeSpatialQueries
from vibespatial.runtime.hotpath_trace import (
    reset_hotpath_trace,
    summarize_hotpath_trace,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARATOR = (
    REPO_ROOT
    / "benchmark_results/spatialbench/sf100/accepted-geopandas-comparator.json"
)
ACCEPTED_DIR = (
    REPO_ROOT
    / "benchmark_results/spatialbench/sf100/2026-08-14-final-median"
)
CANDIDATE = ACCEPTED_DIR / "vibespatial.json"
RESULTS = ACCEPTED_DIR / "results"


def test_q5_native_spill_crossover_is_shape_based_and_public() -> None:
    assert VibeSpatialQueries._q5_uses_native_spill(384_000_128) is False
    assert VibeSpatialQueries._q5_uses_native_spill(999_999_999) is False
    assert VibeSpatialQueries._q5_uses_native_spill(1_000_000_000) is True

    source = inspect.getsource(VibeSpatialQueries.q5)
    assert "write_geoparquet" in source
    assert "NativePartitionedParquetSink" not in source
    assert "pylibcudf" not in source


def test_spatialbench_terminal_export_records_one_bounded_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "counter")
    reset_hotpath_trace()

    with VibeSpatialQueries._terminal_export(rows=100, columns=5):
        pass

    stage = next(
        item
        for item in summarize_hotpath_trace()
        if item["name"] == "spatialbench.terminal_export"
    )
    packet = stage["metadata"]["work_amplification"]
    assert stage["calls"] == 1
    assert packet["sum"] == {
        "diagnostic_synchronizations": 0,
        "output_columns": 5,
        "output_rows": 100,
        "public_frame_materializations": 1,
    }
    assert packet["semantic_contract"] == {
        "selected_rows_bulk_exported": False,
        "single_terminal_export_phase": True,
    }


@pytest.mark.gpu
def test_q3_public_plan_merges_dense_month_reductions_before_terminal_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    batches = [
        {
            "t_pickuptime": pd.to_datetime(
                ["2024-01-02", "2024-01-15", "2024-02-03"]
            ),
            "duration_seconds": [60, 120, 180],
            "t_distance": [1.0, 2.0, 30.0],
            "t_fare": [10.0, 20.0, 300.0],
            "t_pickuploc": [
                Point(-111.75, 34.80),
                Point(-111.74, 34.81),
                Point(-100.0, 0.0),
            ],
        },
        {
            "t_pickuptime": pd.to_datetime(
                ["2024-02-04", "2024-01-20", "2024-03-05"]
            ),
            "duration_seconds": [240, 420, 300],
            "t_distance": [4.0, 50.0, 5.0],
            "t_fare": [40.0, 500.0, 50.0],
            "t_pickuploc": [
                Point(-111.73, 34.82),
                Point(-100.0, 0.0),
                Point(-111.72, 34.83),
            ],
        },
    ]
    native_batches = []
    for batch_index, values in enumerate(batches):
        pickup = values.pop("t_pickuptime")
        duration = values.pop("duration_seconds")
        source = vibespatial.GeoDataFrame(
            {
                "t_pickuptime": pickup,
                "t_dropofftime": pickup + pd.to_timedelta(duration, unit="s"),
                **values,
            },
            geometry="t_pickuploc",
        )
        path = tmp_path / f"q3-batch-{batch_index}.parquet"
        source.to_parquet(path, index=False, geometry_encoding="geoarrow")
        native_batches.append(vibespatial.read_parquet(path))

    queries = VibeSpatialQueries(vibespatial)
    monkeypatch.setattr(
        queries,
        "_spatial_frames",
        lambda *_args, **_kwargs: iter(native_batches),
    )
    reset_d2h_transfer_count()
    vibespatial.clear_fallback_events()
    vibespatial.clear_materialization_events()

    result = queries.q3({"trip": "unused"})
    transfers = get_d2h_transfer_events(clear=True)

    expected = pd.DataFrame(
        {
            "pickup_month": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01"]
            ),
            "total_trips": np.asarray([2, 1, 1], dtype=np.uint64),
            "avg_distance": [1.5, 4.0, 5.0],
            "avg_duration": [90.0, 240.0, 300.0],
            "avg_fare": [15.0, 40.0, 50.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)
    assert vibespatial.get_fallback_events(clear=True) == []
    assert len(vibespatial.get_materialization_events(clear=True)) == 4
    bulk_transfers = [event for event in transfers if event.bytes_transferred > 8]
    assert len(bulk_transfers) == 4
    assert not any(
        "native_series_arithmetic" in event.reason for event in bulk_transfers
    )
    assert sum("numeric_take_to_public_array" in event.reason for event in bulk_transfers) == 3
    assert all(
        event.bytes_transferred <= 8
        for event in transfers
        if event not in bulk_transfers
    )

    source = inspect.getsource(VibeSpatialQueries.q3)
    assert ".groupby(" not in source
    assert ".dt.to_period(" not in source
    assert "pylibcudf" not in source


def _disable_machine_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sf100_evidence, "_validate_dataset", lambda *_: None)
    monkeypatch.setattr(sf100_evidence, "_validate_host", lambda *_: None)


def test_checked_in_sf100_comparator_derives_exact_accepted_total() -> None:
    comparator = sf100_evidence.validate_comparator(COMPARATOR, REPO_ROOT)

    assert comparator["total_time_seconds"] == 8086.0
    assert comparator["queries"]["q5"]["run_times_seconds"] == [
        746.27,
        742.66,
        743.46,
    ]


def test_checked_in_sf100_candidate_and_outputs_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_machine_checks(monkeypatch)

    report = sf100_evidence.verify_candidate(
        comparator_path=COMPARATOR,
        candidate_path=CANDIDATE,
        result_dir=RESULTS,
        dataset_dir=Path("unused-in-portable-test"),
        repo_root=REPO_ROOT,
    )

    assert report["status"] == "pass"
    assert report["correct_queries"] == 12
    assert report["candidate_total_seconds"] == 643.77
    assert report["suite_speedup"] > 10.0
    assert len(report["comparator_sha256"]) == 64
    assert len(report["candidate_result_manifest_sha256"]) == 64


def test_sf100_comparator_rejects_changed_source_hash(tmp_path: Path) -> None:
    comparator = json.loads(COMPARATOR.read_text())
    comparator["derivation"]["replacement"]["sha256"] = "0" * 64
    changed = tmp_path / "changed-comparator.json"
    changed.write_text(json.dumps(comparator))

    with pytest.raises(sf100_evidence.EvidenceError, match="Q5 replacement hash mismatch"):
        sf100_evidence.validate_comparator(changed, REPO_ROOT)


def test_sf100_candidate_rejects_missing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_machine_checks(monkeypatch)
    result_dir = tmp_path / "results"
    shutil.copytree(RESULTS, result_dir)
    (result_dir / "vibespatial_q12_result.csv").unlink()

    with pytest.raises(
        sf100_evidence.EvidenceError, match=r"vibespatial_q12_result\.csv"
    ):
        sf100_evidence.verify_candidate(
            comparator_path=COMPARATOR,
            candidate_path=CANDIDATE,
            result_dir=result_dir,
            dataset_dir=tmp_path,
            repo_root=REPO_ROOT,
        )


def test_sf100_candidate_rejects_exact_integer_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_machine_checks(monkeypatch)
    result_dir = tmp_path / "results"
    shutil.copytree(RESULTS, result_dir)
    q11 = result_dir / "vibespatial_q11_result.csv"
    q11.write_text("cross_zone_trip_count\n1511054982\n")

    with pytest.raises(
        sf100_evidence.EvidenceError, match=r"cross_zone_trip_count.*mismatch"
    ):
        sf100_evidence.verify_candidate(
            comparator_path=COMPARATOR,
            candidate_path=CANDIDATE,
            result_dir=result_dir,
            dataset_dir=tmp_path,
            repo_root=REPO_ROOT,
        )


def test_sf100_candidate_rejects_non_acceptance_measurement_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_machine_checks(monkeypatch)
    candidate = json.loads(CANDIDATE.read_text())
    candidate["results"][0]["results"][0]["warmup_runs"] = 0
    changed = tmp_path / "changed-candidate.json"
    changed.write_text(json.dumps(candidate))

    with pytest.raises(sf100_evidence.EvidenceError, match="one warmup and median"):
        sf100_evidence.verify_candidate(
            comparator_path=COMPARATOR,
            candidate_path=changed,
            result_dir=RESULTS,
            dataset_dir=tmp_path,
            repo_root=REPO_ROOT,
        )
