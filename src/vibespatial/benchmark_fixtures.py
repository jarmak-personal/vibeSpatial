from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import vibespatial.api as geopandas
from vibespatial.io_arrow import write_geoparquet
from vibespatial.testing.synthetic import (
    SyntheticSpec,
    generate_lines,
    generate_mixed_geometries,
    generate_points,
    generate_polygons,
)

_DEFAULT_FIXTURE_DIR = Path(__file__).resolve().parents[2] / ".benchmark_fixtures"


@dataclass(frozen=True)
class BenchmarkFixtureSpec:
    name: str
    geometry_type: str
    distribution: str
    rows: int
    seed: int = 0
    crs: str = "EPSG:4326"
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1_000.0, 1_000.0)
    vertices: int = 8
    clusters: int = 4
    hole_probability: float = 0.0
    mix_ratios: tuple[tuple[str, float], ...] = ()
    part_count: int = 2

    def to_synthetic_spec(self) -> SyntheticSpec:
        return SyntheticSpec(
            geometry_type=self.geometry_type,
            distribution=self.distribution,
            count=self.rows,
            seed=self.seed,
            bounds=self.bounds,
            crs=self.crs,
            vertices=self.vertices,
            clusters=self.clusters,
            hole_probability=self.hole_probability,
            mix_ratios=self.mix_ratios,
            part_count=self.part_count,
        )


DEFAULT_BENCHMARK_FIXTURES: tuple[BenchmarkFixtureSpec, ...] = (
    BenchmarkFixtureSpec("points-grid-rows1000", "point", "grid", 1_000),
    BenchmarkFixtureSpec("points-grid-rows10000", "point", "grid", 10_000),
    BenchmarkFixtureSpec("points-grid-rows100000", "point", "grid", 100_000),
    BenchmarkFixtureSpec("points-grid-rows1000000", "point", "grid", 1_000_000),
    BenchmarkFixtureSpec("lines-random-walk-rows1000", "line", "random-walk", 1_000, vertices=8),
    BenchmarkFixtureSpec("lines-random-walk-rows10000", "line", "random-walk", 10_000, vertices=8),
    BenchmarkFixtureSpec("lines-random-walk-rows100000", "line", "random-walk", 100_000, vertices=8),
    BenchmarkFixtureSpec("polygons-regular-grid-rows1000", "polygon", "regular-grid", 1_000),
    BenchmarkFixtureSpec("polygons-regular-grid-rows10000", "polygon", "regular-grid", 10_000),
    BenchmarkFixtureSpec("polygons-regular-grid-rows100000", "polygon", "regular-grid", 100_000),
    BenchmarkFixtureSpec("polygons-regular-grid-rows1000000", "polygon", "regular-grid", 1_000_000),
    BenchmarkFixtureSpec(
        "mixed-basic-rows1000",
        "mixed",
        "mixed",
        1_000,
        mix_ratios=(("point", 0.4), ("line", 0.3), ("polygon", 0.3)),
    ),
    BenchmarkFixtureSpec(
        "mixed-basic-rows10000",
        "mixed",
        "mixed",
        10_000,
        mix_ratios=(("point", 0.4), ("line", 0.3), ("polygon", 0.3)),
    ),
    BenchmarkFixtureSpec(
        "mixed-basic-rows100000",
        "mixed",
        "mixed",
        100_000,
        mix_ratios=(("point", 0.4), ("line", 0.3), ("polygon", 0.3)),
    ),
)


def default_fixture_dir() -> Path:
    return _DEFAULT_FIXTURE_DIR


def list_fixture_specs() -> tuple[BenchmarkFixtureSpec, ...]:
    return DEFAULT_BENCHMARK_FIXTURES


def get_fixture_spec(name: str) -> BenchmarkFixtureSpec:
    for spec in DEFAULT_BENCHMARK_FIXTURES:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown benchmark fixture: {name}")


def fixture_path(spec_or_name: BenchmarkFixtureSpec | str, *, fixture_dir: str | Path | None = None) -> Path:
    spec = get_fixture_spec(spec_or_name) if isinstance(spec_or_name, str) else spec_or_name
    root = Path(fixture_dir) if fixture_dir is not None else default_fixture_dir()
    return root / f"{spec.name}.parquet"


def build_fixture_frame(spec: BenchmarkFixtureSpec) -> geopandas.GeoDataFrame:
    synthetic = spec.to_synthetic_spec()
    if spec.geometry_type == "point":
        dataset = generate_points(synthetic)
    elif spec.geometry_type == "line":
        dataset = generate_lines(synthetic)
    elif spec.geometry_type == "polygon":
        dataset = generate_polygons(synthetic)
    elif spec.geometry_type == "mixed":
        dataset = generate_mixed_geometries(synthetic)
    else:
        raise ValueError(f"Unsupported fixture geometry type: {spec.geometry_type}")
    return dataset.to_geodataframe()


def ensure_fixture(
    spec_or_name: BenchmarkFixtureSpec | str,
    *,
    fixture_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    spec = get_fixture_spec(spec_or_name) if isinstance(spec_or_name, str) else spec_or_name
    path = fixture_path(spec, fixture_dir=fixture_dir)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = build_fixture_frame(spec)
    write_geoparquet(frame, path, geometry_encoding="geoarrow")
    return path
