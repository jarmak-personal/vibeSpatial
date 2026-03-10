from __future__ import annotations

from pathlib import Path

from vibespatial.benchmark_fixtures import BenchmarkFixtureSpec, build_fixture_frame, ensure_fixture, fixture_path
from vibespatial.fixture_profiles import profile_fixture_query


def test_fixture_path_uses_name_and_parquet_suffix(tmp_path: Path) -> None:
    spec = BenchmarkFixtureSpec("points-grid-rows16", "point", "grid", 16)
    path = fixture_path(spec, fixture_dir=tmp_path)

    assert path == tmp_path / "points-grid-rows16.parquet"


def test_ensure_fixture_writes_geoparquet(tmp_path: Path) -> None:
    spec = BenchmarkFixtureSpec("points-grid-rows16", "point", "grid", 16)

    path = ensure_fixture(spec, fixture_dir=tmp_path)

    assert path.exists()
    assert path.suffix == ".parquet"


def test_build_fixture_frame_supports_mixed_geometry() -> None:
    spec = BenchmarkFixtureSpec(
        "mixed-basic-rows32",
        "mixed",
        "mixed",
        32,
        mix_ratios=(("point", 0.5), ("line", 0.25), ("polygon", 0.25)),
    )

    frame = build_fixture_frame(spec)

    assert len(frame) == 32


def test_profile_fixture_query_uses_public_parquet_path(tmp_path: Path) -> None:
    spec = BenchmarkFixtureSpec("polygons-regular-grid-rows16", "polygon", "regular-grid", 16)
    ensure_fixture(spec, fixture_dir=tmp_path)

    trace = profile_fixture_query(spec.name, fixture_dir=tmp_path)

    assert trace.operation == "query"
    assert trace.tree_rows == 16
    assert trace.query_rows == 16
    assert trace.result_pairs >= 16
    assert any(event["operation"] == "read_parquet" for event in trace.dispatch_events)
    assert any(event["operation"] == "query" for event in trace.dispatch_events)