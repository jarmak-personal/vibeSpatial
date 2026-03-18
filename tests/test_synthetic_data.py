from __future__ import annotations

import json

import pytest

import vibespatial.api as geopandas
from vibespatial.testing import (
    SCALE_PRESETS,
    SyntheticSpec,
    generate_invalid_geometries,
    generate_lines,
    generate_mixed_geometries,
    generate_multigeometries,
    generate_points,
    generate_polygons,
)


def test_scale_presets_include_foundation_sizes() -> None:
    assert SCALE_PRESETS["1K"] == 1_000
    assert SCALE_PRESETS["10M"] == 10_000_000


def test_uniform_points_are_reproducible() -> None:
    spec = SyntheticSpec(geometry_type="point", distribution="uniform", count=12, seed=7)
    left = generate_points(spec)
    right = generate_points(spec)
    assert left.geometries == right.geometries


def test_point_distributions_generate_expected_counts() -> None:
    for distribution in ("uniform", "clustered", "grid", "along-line"):
        spec = SyntheticSpec(geometry_type="point", distribution=distribution, count=16, seed=3)
        dataset = generate_points(spec)
        assert len(dataset.geometries) == 16
        assert all(geometry.geom_type == "Point" for geometry in dataset.geometries)


def test_line_and_polygon_generators_cover_core_distributions() -> None:
    lines = generate_lines(SyntheticSpec("line", "random-walk", count=8, seed=2, vertices=6))
    grid_lines = generate_lines(SyntheticSpec("line", "grid", count=8, seed=2))
    polygons = generate_polygons(SyntheticSpec("polygon", "star", count=8, seed=2, hole_probability=0.5))
    grid_polygons = generate_polygons(SyntheticSpec("polygon", "regular-grid", count=8, seed=2))

    assert all(geometry.geom_type == "LineString" for geometry in lines.geometries)
    assert all(geometry.geom_type == "LineString" for geometry in grid_lines.geometries)
    assert all(geometry.geom_type == "Polygon" for geometry in polygons.geometries)
    assert all(geometry.geom_type == "Polygon" for geometry in grid_polygons.geometries)


def test_multigeometry_generation_groups_base_geometries() -> None:
    dataset = generate_multigeometries(
        SyntheticSpec("polygon", "regular-grid", count=4, seed=4, part_count=2)
    )
    assert len(dataset.geometries) == 4
    assert all(geometry.geom_type == "MultiPolygon" for geometry in dataset.geometries)


def test_mixed_generation_uses_requested_ratios() -> None:
    dataset = generate_mixed_geometries(
        SyntheticSpec(
            geometry_type="mixed",
            distribution="ratio-mix",
            count=20,
            seed=5,
            mix_ratios=(("point", 0.5), ("line", 0.25), ("polygon", 0.25)),
        )
    )
    counts = geopandas.GeoSeries(list(dataset.geometries)).geom_type.value_counts().to_dict()
    assert counts["Point"] == 10
    assert counts["LineString"] == 5
    assert counts["Polygon"] == 5


def test_invalid_generator_includes_non_valid_shapes() -> None:
    dataset = generate_invalid_geometries(count=8, seed=1)
    validity = geopandas.GeoSeries(list(dataset.geometries)).is_valid.tolist()
    assert any(not value for value in validity)


def test_dataset_exports_geojson(tmp_path) -> None:
    dataset = generate_polygons(SyntheticSpec("polygon", "convex-hull", count=5, seed=9, crs="EPSG:4326"))
    path = dataset.write_geojson(tmp_path / "synthetic.geojson")
    payload = json.loads(path.read_text())
    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) == 5


def test_dataset_parquet_export_works_or_skips(tmp_path) -> None:
    dataset = generate_points(SyntheticSpec("point", "grid", count=4, seed=2, crs="EPSG:4326"))
    try:
        path = dataset.write_geoparquet(tmp_path / "synthetic.parquet")
    except ImportError:
        pytest.skip("pyarrow is not installed")
    assert path.exists()


def test_synthetic_dataset_fixture_returns_requested_geometry(synthetic_dataset) -> None:
    dataset = synthetic_dataset(SyntheticSpec("point", "grid", count=9, seed=11))
    assert len(dataset.geometries) == 9
    assert all(geometry.geom_type == "Point" for geometry in dataset.geometries)
