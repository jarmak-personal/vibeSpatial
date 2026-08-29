from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmarks.spatialbench import sf100_evidence

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
