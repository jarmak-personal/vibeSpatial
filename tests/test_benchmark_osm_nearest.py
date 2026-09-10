"""Correctness contracts of the OSM nearest comparison, independent of timing."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import shapely

_SPEC = importlib.util.spec_from_file_location(
    "benchmark_osm_nearest", Path(__file__).parents[1] / "scripts" / "benchmark_osm_nearest.py"
)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_segments_preserve_distance_and_parent_rows_without_bridging_roads():
    roads = shapely.from_wkt([
        "LINESTRING (0 0, 1 0, 1 0, 1 2)",
        "LINESTRING (10 10, 12 10)",
    ])
    segments, parents = benchmark.segment_lines(roads)
    np.testing.assert_array_equal(parents, [0, 0, 0, 1])
    np.testing.assert_array_equal(shapely.length(segments), [1, 0, 2, 2])
    queries = shapely.points([0.5, 1, 6, 11], [1, 0, 6, 11])
    whole_pairs, whole_distances = shapely.STRtree(roads).query_nearest(queries, return_distance=True)
    segment_pairs, segment_distances = shapely.STRtree(segments).query_nearest(queries, return_distance=True)
    # Segment endpoint ties are intentionally retained, then deduplicated only
    # for this parent-road equivalence check, never inside the timed operation.
    parent_pairs = segment_pairs.copy()
    parent_pairs[1] = parents[parent_pairs[1]]
    unique_pairs, positions = np.unique(parent_pairs, axis=1, return_index=True)
    assert benchmark.compare((unique_pairs, segment_distances[positions]), (whole_pairs, whole_distances))["passed"]


def test_all_matches_comparison_is_order_independent_but_rejects_missing_ties():
    oracle = (np.array([[0, 0, 1], [4, 5, 2]]), np.array([0., 0., 3.]))
    permutation = [2, 1, 0]
    reordered = (oracle[0][:, permutation], oracle[1][permutation])
    assert benchmark.compare(reordered, oracle)["passed"]
    missing_tie = (oracle[0][:, [0, 2]], oracle[1][[0, 2]])
    assert not benchmark.compare(missing_tie, oracle)["passed"]
    duplicate = (np.array([[0, 0, 1], [4, 4, 2]]), oracle[1])
    assert not benchmark.compare(duplicate, oracle)["passed"]


def test_distance_errors_and_nonfinite_results_fail_parity():
    oracle = (np.array([[0], [2]]), np.array([1.]))
    assert benchmark.compare((oracle[0], np.array([1. + 1e-8])), oracle)["passed"]
    assert not benchmark.compare((oracle[0], np.array([1.01])), oracle)["passed"]
    assert not benchmark.compare((oracle[0], np.array([np.nan])), oracle)["passed"]
    for invalid in (np.inf, -np.inf, -1.):
        bad = (oracle[0], np.array([invalid]))
        result = benchmark.compare(bad, bad)
        assert not result["passed"]
        assert result["max_distance_error_m"] is None


def test_failed_worker_cannot_reuse_previous_candidate_timings(tmp_path, monkeypatch):
    previous = tmp_path / "vibespatial-roads-1000.json"
    previous.write_text(json.dumps({"status": "ok", "median_seconds": {"warm_query": 0.001}}))
    args = SimpleNamespace(cache=tmp_path, output=tmp_path / "report.json",
                           path=tmp_path / "source.osm.pbf", repeat=1, mode="gpu",
                           profile=False, timeout=1)

    def fail_before_checkpoint(command, **kwargs):
        return subprocess.CompletedProcess(command, returncode=7)

    monkeypatch.setattr(benchmark.subprocess, "run", fail_before_checkpoint)
    result = benchmark.run_worker(args, "vibespatial", "roads", "1000")
    assert result["status"] == "error"
    assert "median_seconds" not in result
    assert json.loads(previous.read_text()) == result
