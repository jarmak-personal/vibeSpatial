from __future__ import annotations

import json

from vibespatial.bench.io_benchmark_rails import (
    benchmark_io_arrow_suite,
    benchmark_io_file_suite,
    io_suite_to_json,
)


def test_io_arrow_smoke_suite_reports_enforced_and_informational_cases() -> None:
    results = benchmark_io_arrow_suite(suite="smoke", repeat=1)
    by_id = {result.case_id: result for result in results}

    assert "geoarrow-bridge-encode-point-10000" in by_id
    assert "wkb-decode-point-10000" in by_id
    assert "geoparquet-selective-point-100000" in by_id
    assert all(result.metric in {"speedup", "decoded_row_fraction"} for result in results)

    payload = json.loads(io_suite_to_json(results, suite="smoke", repeat=1, scope="io-arrow"))
    assert payload["metadata"]["suite"] == "smoke"
    assert payload["metadata"]["scope"] == "io-arrow"
    assert "results" in payload


def test_io_file_smoke_suite_keeps_geojson_informational() -> None:
    results = benchmark_io_file_suite(suite="smoke", repeat=1)
    by_id = {result.case_id: result for result in results}

    assert "shapefile-point-10000" in by_id
    assert "shapefile-line-10000" in by_id
    assert "shapefile-polygon-5000" in by_id
    assert "geojson-point-10000" in by_id
    assert by_id["geojson-point-10000"].enforced is False
    assert by_id["geojson-point-10000"].status in {"pass", "informational"}

    payload = json.loads(io_suite_to_json(results, suite="smoke", repeat=1, scope="io-file"))
    assert payload["metadata"]["scope"] == "io-file"
    assert payload["metadata"]["statuses"]["pass"] >= 1
