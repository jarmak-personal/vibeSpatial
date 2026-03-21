from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import vibespatial.api as geopandas
from vibespatial.io.arrow import write_geoparquet
from vibespatial.testing.synthetic import (
    SyntheticSpec,
    generate_lines,
    generate_mixed_geometries,
    generate_points,
    generate_polygons,
)

_DEFAULT_FIXTURE_DIR = Path(__file__).resolve().parents[3] / ".benchmark_fixtures"


# ---------------------------------------------------------------------------
# Input format enum
# ---------------------------------------------------------------------------

class InputFormat(StrEnum):
    PARQUET = "parquet"
    GEOJSON = "geojson"
    SHAPEFILE = "shapefile"
    GPKG = "gpkg"


ALL_INPUT_FORMATS = tuple(InputFormat)


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
    # Mixed geometry types can't use geoarrow encoding (Shapely ragged array
    # limitation), fall back to WKB.
    encoding = "WKB" if spec.geometry_type == "mixed" else "geoarrow"
    write_geoparquet(frame, path, geometry_encoding=encoding)
    return path


# ---------------------------------------------------------------------------
# Multi-format fixture support
# ---------------------------------------------------------------------------

def fixture_path_for_format(
    spec_or_name: BenchmarkFixtureSpec | str,
    fmt: InputFormat | str,
    *,
    fixture_dir: str | Path | None = None,
) -> Path:
    """Resolve the file path for a fixture in a given format."""
    spec = get_fixture_spec(spec_or_name) if isinstance(spec_or_name, str) else spec_or_name
    root = Path(fixture_dir) if fixture_dir is not None else default_fixture_dir()
    fmt = InputFormat(fmt) if not isinstance(fmt, InputFormat) else fmt
    match fmt:
        case InputFormat.PARQUET:
            return root / f"{spec.name}.parquet"
        case InputFormat.GEOJSON:
            return root / f"{spec.name}.geojson"
        case InputFormat.SHAPEFILE:
            return root / spec.name / f"{spec.name}.shp"
        case InputFormat.GPKG:
            return root / f"{spec.name}.gpkg"
    raise ValueError(f"Unsupported format: {fmt}")


def ensure_fixture_format(
    spec_or_name: BenchmarkFixtureSpec | str,
    fmt: InputFormat | str,
    *,
    fixture_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Ensure a fixture exists in the given format, generating if needed.

    The canonical GeoParquet is generated first (via ``ensure_fixture``),
    then converted to the target format if it doesn't already exist.
    """
    spec = get_fixture_spec(spec_or_name) if isinstance(spec_or_name, str) else spec_or_name
    fmt = InputFormat(fmt) if not isinstance(fmt, InputFormat) else fmt
    target = fixture_path_for_format(spec, fmt, fixture_dir=fixture_dir)

    if target.exists() and not force:
        return target

    # Shapefile doesn't support mixed geometry types
    if fmt == InputFormat.SHAPEFILE and spec.geometry_type == "mixed":
        raise ValueError(
            f"Shapefile format does not support mixed geometry types "
            f"(fixture {spec.name!r}). Use parquet, geojson, or gpkg."
        )

    # Always ensure the canonical parquet exists first
    parquet_path = ensure_fixture(spec, fixture_dir=fixture_dir, force=False)

    if fmt == InputFormat.PARQUET:
        return parquet_path

    # Load from parquet (CPU path to avoid GPU dependency during generation)
    import pyarrow.parquet as pq

    table = pq.read_table(str(parquet_path))
    frame = geopandas.GeoDataFrame.from_arrow(table)

    target.parent.mkdir(parents=True, exist_ok=True)

    match fmt:
        case InputFormat.GEOJSON:
            frame.to_file(target, driver="GeoJSON")
        case InputFormat.SHAPEFILE:
            frame.to_file(target, driver="ESRI Shapefile")
        case InputFormat.GPKG:
            frame.to_file(target, driver="GPKG", layer=spec.name)

    return target


def ensure_fixture_all_formats(
    spec_or_name: BenchmarkFixtureSpec | str,
    formats: tuple[InputFormat | str, ...] | None = None,
    *,
    fixture_dir: str | Path | None = None,
    force: bool = False,
) -> dict[InputFormat, Path]:
    """Ensure a fixture exists in all requested formats."""
    if formats is None:
        formats = ALL_INPUT_FORMATS
    return {
        InputFormat(fmt): ensure_fixture_format(
            spec_or_name, fmt, fixture_dir=fixture_dir, force=force,
        )
        for fmt in formats
    }


# ---------------------------------------------------------------------------
# Dynamic fixture resolution
# ---------------------------------------------------------------------------

def resolve_fixture_spec(
    geometry_type: str,
    distribution: str,
    scale: int,
    *,
    seed: int = 0,
    vertices: int = 8,
    hole_probability: float = 0.0,
) -> BenchmarkFixtureSpec:
    """Find a predefined fixture spec or create a dynamic one for the given params."""
    for spec in DEFAULT_BENCHMARK_FIXTURES:
        if (
            spec.geometry_type == geometry_type
            and spec.distribution == distribution
            and spec.rows == scale
            and spec.seed == seed
        ):
            return spec
    # Create dynamic spec
    name = f"{geometry_type}s-{distribution}-rows{scale}"
    if seed != 0:
        name += f"-seed{seed}"
    return BenchmarkFixtureSpec(
        name=name,
        geometry_type=geometry_type,
        distribution=distribution,
        rows=scale,
        seed=seed,
        vertices=vertices,
        hole_probability=hole_probability,
    )


# ---------------------------------------------------------------------------
# Shifted fixture variants (for binary operations)
# ---------------------------------------------------------------------------

def resolve_shifted_fixture_spec(
    base: BenchmarkFixtureSpec,
    *,
    xoff: float,
    yoff: float,
) -> BenchmarkFixtureSpec:
    """Create a spec for a spatially shifted variant of a base fixture."""
    shifted_name = f"{base.name}-shifted-{xoff}-{yoff}"
    return BenchmarkFixtureSpec(
        name=shifted_name,
        geometry_type=base.geometry_type,
        distribution=base.distribution,
        rows=base.rows,
        seed=base.seed,
        crs=base.crs,
        bounds=base.bounds,
        vertices=base.vertices,
        clusters=base.clusters,
        hole_probability=base.hole_probability,
        mix_ratios=base.mix_ratios,
        part_count=base.part_count,
    )


def ensure_shifted_fixture(
    base: BenchmarkFixtureSpec,
    *,
    xoff: float,
    yoff: float,
    fmt: InputFormat | str = InputFormat.PARQUET,
    fixture_dir: str | Path | None = None,
    force: bool = False,
) -> tuple[BenchmarkFixtureSpec, Path]:
    """Generate a spatially shifted variant of a fixture, cached on disk.

    Returns ``(shifted_spec, path)`` for the shifted fixture in the given format.
    """
    from shapely.affinity import translate

    shifted_spec = resolve_shifted_fixture_spec(base, xoff=xoff, yoff=yoff)
    fmt = InputFormat(fmt) if not isinstance(fmt, InputFormat) else fmt

    # Check if canonical parquet already exists
    parquet_path = fixture_path_for_format(shifted_spec, InputFormat.PARQUET, fixture_dir=fixture_dir)
    if not parquet_path.exists() or force:
        # Load base fixture, shift, write
        base_parquet = ensure_fixture(base, fixture_dir=fixture_dir, force=False)

        import pyarrow.parquet as pq

        table = pq.read_table(str(base_parquet))
        frame = geopandas.GeoDataFrame.from_arrow(table)
        frame["geometry"] = frame.geometry.apply(lambda g: translate(g, xoff=xoff, yoff=yoff))

        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        write_geoparquet(frame, parquet_path, geometry_encoding="geoarrow")

    # Now ensure the target format exists
    path = ensure_fixture_format(shifted_spec, fmt, fixture_dir=fixture_dir, force=force)
    return shifted_spec, path


# ---------------------------------------------------------------------------
# Specialized fixture variants
# ---------------------------------------------------------------------------

def resolve_invalids_fixture_spec(scale: int, *, invalid_ratio: float = 0.05) -> BenchmarkFixtureSpec:
    """Spec for a polygon fixture with some invalid geometries baked in."""
    return BenchmarkFixtureSpec(
        name=f"polygons-with-invalids-rows{scale}",
        geometry_type="polygon",
        distribution="regular-grid",
        rows=scale,
    )


def ensure_invalids_fixture(
    scale: int,
    *,
    invalid_ratio: float = 0.05,
    fmt: InputFormat | str = InputFormat.PARQUET,
    fixture_dir: str | Path | None = None,
    force: bool = False,
) -> tuple[BenchmarkFixtureSpec, Path]:
    """Generate a polygon fixture with ~invalid_ratio invalid geometries."""
    from shapely.geometry import Polygon

    spec = resolve_invalids_fixture_spec(scale, invalid_ratio=invalid_ratio)
    fmt = InputFormat(fmt) if not isinstance(fmt, InputFormat) else fmt
    parquet_path = fixture_path_for_format(spec, InputFormat.PARQUET, fixture_dir=fixture_dir)

    if not parquet_path.exists() or force:
        invalid_every = max(1, int(1.0 / invalid_ratio))
        geometries = []
        for i in range(scale):
            x = float(i)
            if i % invalid_every == 0:
                # Self-intersecting bowtie
                geometries.append(Polygon([(x, 0), (x + 1, 1), (x + 1, 0), (x, 1), (x, 0)]))
            else:
                geometries.append(Polygon([(x, 0), (x, 1), (x + 1, 1), (x + 1, 0)]))

        frame = geopandas.GeoDataFrame(
            {"geometry": geometries}, geometry="geometry", crs="EPSG:4326",
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        write_geoparquet(frame, parquet_path, geometry_encoding="geoarrow")

    path = ensure_fixture_format(spec, fmt, fixture_dir=fixture_dir, force=force)
    return spec, path


def resolve_grouped_boxes_fixture_spec(scale: int, groups: int = 100) -> BenchmarkFixtureSpec:
    """Spec for a rectangular coverage fixture with a group column."""
    return BenchmarkFixtureSpec(
        name=f"boxes-grouped-rows{scale}-groups{groups}",
        geometry_type="polygon",
        distribution="regular-grid",
        rows=scale,
    )


def ensure_grouped_boxes_fixture(
    scale: int,
    *,
    groups: int = 100,
    fmt: InputFormat | str = InputFormat.PARQUET,
    fixture_dir: str | Path | None = None,
    force: bool = False,
) -> tuple[BenchmarkFixtureSpec, Path]:
    """Generate a rectangular box coverage with group labels."""
    from shapely.geometry import box

    spec = resolve_grouped_boxes_fixture_spec(scale, groups)
    fmt = InputFormat(fmt) if not isinstance(fmt, InputFormat) else fmt
    parquet_path = fixture_path_for_format(spec, InputFormat.PARQUET, fixture_dir=fixture_dir)

    if not parquet_path.exists() or force:
        rows_per_group = scale // groups
        geometries = []
        group_values = []
        for group in range(groups):
            base_y = float(group * 10.0)
            for idx in range(rows_per_group):
                base_x = float(idx)
                geometries.append(box(base_x, base_y, base_x + 1.0, base_y + 1.0))
                group_values.append(group)

        frame = geopandas.GeoDataFrame(
            {"group": group_values, "geometry": geometries},
            geometry="geometry",
            crs="EPSG:3857",
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        write_geoparquet(frame, parquet_path, geometry_encoding="geoarrow")

    path = ensure_fixture_format(spec, fmt, fixture_dir=fixture_dir, force=force)
    return spec, path
