from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibespatial.geometry.owned import OwnedGeometryArray

import numpy as np
from shapely import affinity
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    mapping,
)

import vibespatial.api as geopandas

SCALE_PRESETS = {
    "1K": 1_000,
    "10K": 10_000,
    "100K": 100_000,
    "1M": 1_000_000,
    "10M": 10_000_000,
}


@dataclass(frozen=True)
class SyntheticSpec:
    geometry_type: str
    distribution: str
    count: int | str = "1K"
    seed: int = 0
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1_000.0, 1_000.0)
    crs: str | None = None
    vertices: int = 8
    clusters: int = 4
    hole_probability: float = 0.0
    mix_ratios: tuple[tuple[str, float], ...] = ()
    part_count: int = 2

    @property
    def resolved_count(self) -> int:
        if isinstance(self.count, str):
            return SCALE_PRESETS[self.count]
        return self.count


@dataclass(frozen=True)
class SyntheticDataset:
    spec: SyntheticSpec
    geometries: tuple[Point | LineString | Polygon | MultiPoint | MultiLineString | MultiPolygon, ...]

    def to_geoseries(self) -> geopandas.GeoSeries:
        return geopandas.GeoSeries(list(self.geometries), crs=self.spec.crs)

    def to_geodataframe(self) -> geopandas.GeoDataFrame:
        return geopandas.GeoDataFrame({"geometry": list(self.geometries)}, crs=self.spec.crs)

    def write_geojson(self, path: str | Path) -> Path:
        output = Path(path)
        features = [
            {
                "type": "Feature",
                "properties": {"index": index},
                "geometry": mapping(geometry),
            }
            for index, geometry in enumerate(self.geometries)
        ]
        payload = {"type": "FeatureCollection", "features": features}
        output.write_text(json.dumps(payload))
        return output

    def write_geoparquet(self, path: str | Path) -> Path:
        output = Path(path)
        self.to_geodataframe().to_parquet(output)
        return output


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _resolve_count(count: int | str) -> int:
    if isinstance(count, str):
        return SCALE_PRESETS[count]
    return count


def _grid_side(count: int) -> int:
    return max(1, math.ceil(math.sqrt(count)))


def _linspace(minimum: float, maximum: float, count: int) -> np.ndarray:
    if count == 1:
        return np.array([(minimum + maximum) / 2.0])
    return np.linspace(minimum, maximum, count)


def _coerce_dataset(spec: SyntheticSpec, geometries: Iterable[object]) -> SyntheticDataset:
    return SyntheticDataset(spec=spec, geometries=tuple(geometries))


def generate_points(spec: SyntheticSpec) -> SyntheticDataset:
    count = spec.resolved_count
    xmin, ymin, xmax, ymax = spec.bounds
    rng = _rng(spec.seed)

    if spec.distribution == "uniform":
        xs = rng.uniform(xmin, xmax, count)
        ys = rng.uniform(ymin, ymax, count)
    elif spec.distribution == "clustered":
        centers_x = rng.uniform(xmin, xmax, spec.clusters)
        centers_y = rng.uniform(ymin, ymax, spec.clusters)
        picks = rng.integers(0, spec.clusters, count)
        xs = centers_x[picks] + rng.normal(0.0, (xmax - xmin) / 30.0, count)
        ys = centers_y[picks] + rng.normal(0.0, (ymax - ymin) / 30.0, count)
        xs = np.clip(xs, xmin, xmax)
        ys = np.clip(ys, ymin, ymax)
    elif spec.distribution == "grid":
        side = _grid_side(count)
        xs_grid, ys_grid = np.meshgrid(_linspace(xmin, xmax, side), _linspace(ymin, ymax, side))
        xs = xs_grid.ravel()[:count]
        ys = ys_grid.ravel()[:count]
    elif spec.distribution == "along-line":
        t = _linspace(0.0, 1.0, count)
        xs = xmin + (xmax - xmin) * t
        ys = ymin + (ymax - ymin) * (0.5 + 0.3 * np.sin(2 * math.pi * t))
    else:
        raise ValueError(f"Unsupported point distribution: {spec.distribution}")

    return _coerce_dataset(spec, (Point(float(x), float(y)) for x, y in zip(xs, ys, strict=True)))


def generate_points_xy(spec: SyntheticSpec) -> tuple[np.ndarray, np.ndarray]:
    """Generate x/y coordinate arrays for points without Shapely objects.

    Returns raw ``(xs, ys)`` numpy float64 arrays using the same generation
    logic as ``generate_points``.
    """
    count = spec.resolved_count
    xmin, ymin, xmax, ymax = spec.bounds
    rng = _rng(spec.seed)

    if spec.distribution == "uniform":
        xs = rng.uniform(xmin, xmax, count)
        ys = rng.uniform(ymin, ymax, count)
    elif spec.distribution == "clustered":
        centers_x = rng.uniform(xmin, xmax, spec.clusters)
        centers_y = rng.uniform(ymin, ymax, spec.clusters)
        picks = rng.integers(0, spec.clusters, count)
        xs = centers_x[picks] + rng.normal(0.0, (xmax - xmin) / 30.0, count)
        ys = centers_y[picks] + rng.normal(0.0, (ymax - ymin) / 30.0, count)
        xs = np.clip(xs, xmin, xmax)
        ys = np.clip(ys, ymin, ymax)
    elif spec.distribution == "grid":
        side = _grid_side(count)
        xs_grid, ys_grid = np.meshgrid(
            _linspace(xmin, xmax, side), _linspace(ymin, ymax, side)
        )
        xs = xs_grid.ravel()[:count]
        ys = ys_grid.ravel()[:count]
    elif spec.distribution == "along-line":
        t = _linspace(0.0, 1.0, count)
        xs = xmin + (xmax - xmin) * t
        ys = ymin + (ymax - ymin) * (0.5 + 0.3 * np.sin(2 * math.pi * t))
    else:
        raise ValueError(f"Unsupported point distribution: {spec.distribution}")

    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def generate_points_owned(spec: SyntheticSpec) -> OwnedGeometryArray:
    """Generate points directly as an OwnedGeometryArray, bypassing Shapely.

    Returns a HOST-resident OwnedGeometryArray built from raw coordinate
    arrays via ``point_owned_from_xy``.  Callers that need device residency
    can call ``owned.move_to("device", trigger="benchmark")`` afterward.
    """
    from vibespatial.constructive.point import point_owned_from_xy

    xs, ys = generate_points_xy(spec)
    return point_owned_from_xy(xs, ys)


def generate_lines(spec: SyntheticSpec) -> SyntheticDataset:
    count = spec.resolved_count
    xmin, ymin, xmax, ymax = spec.bounds
    rng = _rng(spec.seed)
    geometries: list[LineString] = []

    if spec.distribution == "random-walk":
        step = min(xmax - xmin, ymax - ymin) / max(spec.vertices, 2)
        for _ in range(count):
            x = float(rng.uniform(xmin, xmax))
            y = float(rng.uniform(ymin, ymax))
            coords = [(x, y)]
            for _ in range(max(spec.vertices - 1, 1)):
                x = float(np.clip(x + rng.normal(0.0, step), xmin, xmax))
                y = float(np.clip(y + rng.normal(0.0, step), ymin, ymax))
                coords.append((x, y))
            geometries.append(LineString(coords))
    elif spec.distribution == "grid":
        side = _grid_side(count)
        xs = _linspace(xmin, xmax, side + 1)
        ys = _linspace(ymin, ymax, side + 1)
        for index in range(count):
            if index % 2 == 0:
                y = ys[(index // 2) % len(ys)]
                geometries.append(LineString([(xmin, float(y)), (xmax, float(y))]))
            else:
                x = xs[(index // 2) % len(xs)]
                geometries.append(LineString([(float(x), ymin), (float(x), ymax)]))
    elif spec.distribution == "river":
        xs = _linspace(xmin, xmax, spec.vertices)
        for offset in rng.uniform(ymin, ymax, count):
            amplitude = (ymax - ymin) / 8.0
            phase = rng.uniform(0.0, 2 * math.pi)
            coords = [
                (
                    float(x),
                    float(np.clip(offset + amplitude * math.sin(phase + i / 2.0), ymin, ymax)),
                )
                for i, x in enumerate(xs)
            ]
            geometries.append(LineString(coords))
    else:
        raise ValueError(f"Unsupported line distribution: {spec.distribution}")

    return _coerce_dataset(spec, geometries)


def _star_polygon(cx: float, cy: float, outer_radius: float, inner_radius: float, vertices: int) -> Polygon:
    coords: list[tuple[float, float]] = []
    for i in range(vertices * 2):
        angle = math.pi * i / vertices
        radius = outer_radius if i % 2 == 0 else inner_radius
        coords.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return Polygon(coords)


def _convexish_polygon(
    rng: np.random.Generator,
    center_x: float,
    center_y: float,
    radius: float,
    vertices: int,
) -> Polygon:
    angles = np.sort(rng.uniform(0.0, 2 * math.pi, vertices))
    scales = rng.uniform(0.5, 1.0, vertices)
    coords = [
        (center_x + radius * scale * math.cos(angle), center_y + radius * scale * math.sin(angle))
        for angle, scale in zip(angles, scales, strict=True)
    ]
    return Polygon(coords)


def generate_polygons(spec: SyntheticSpec) -> SyntheticDataset:
    count = spec.resolved_count
    xmin, ymin, xmax, ymax = spec.bounds
    rng = _rng(spec.seed)
    geometries: list[Polygon] = []

    if spec.distribution == "regular-grid":
        side = _grid_side(count)
        xs = _linspace(xmin, xmax, side + 1)
        ys = _linspace(ymin, ymax, side + 1)
        for row in range(side):
            for col in range(side):
                if len(geometries) >= count:
                    break
                polygon = Polygon(
                    [
                        (float(xs[col]), float(ys[row])),
                        (float(xs[col + 1]), float(ys[row])),
                        (float(xs[col + 1]), float(ys[row + 1])),
                        (float(xs[col]), float(ys[row + 1])),
                    ]
                )
                geometries.append(polygon)
    elif spec.distribution == "convex-hull":
        radius = min(xmax - xmin, ymax - ymin) / max(_grid_side(count) * 3, 2)
        for point in generate_points(
            SyntheticSpec(
                geometry_type="point",
                distribution="clustered",
                count=count,
                seed=spec.seed,
                bounds=spec.bounds,
                clusters=spec.clusters,
            )
        ).geometries:
            geometries.append(
                _convexish_polygon(rng, point.x, point.y, radius, max(spec.vertices, 3))
            )
    elif spec.distribution == "star":
        centers = generate_points(
            SyntheticSpec(
                geometry_type="point",
                distribution="grid",
                count=count,
                seed=spec.seed,
                bounds=spec.bounds,
            )
        ).geometries
        radius = min(xmax - xmin, ymax - ymin) / max(_grid_side(count) * 4, 2)
        for center in centers:
            polygon = _star_polygon(center.x, center.y, radius, radius / 2.5, max(spec.vertices, 3))
            if spec.hole_probability and rng.uniform() < spec.hole_probability:
                hole = affinity.scale(polygon, xfact=0.35, yfact=0.35, origin="center")
                polygon = Polygon(polygon.exterior.coords, [hole.exterior.coords])
            geometries.append(polygon)
    else:
        raise ValueError(f"Unsupported polygon distribution: {spec.distribution}")

    return _coerce_dataset(spec, geometries[:count])


def generate_multigeometries(
    spec: SyntheticSpec,
    base_geometries: Sequence[Point | LineString | Polygon] | None = None,
) -> SyntheticDataset:
    count = spec.resolved_count
    if base_geometries is None:
        base_spec = SyntheticSpec(
            geometry_type="polygon",
            distribution="regular-grid",
            count=count * spec.part_count,
            seed=spec.seed,
            bounds=spec.bounds,
            crs=spec.crs,
        )
        generators = {
            "point": generate_points,
            "line": generate_lines,
            "polygon": generate_polygons,
        }
        base_geometries = generators[spec.geometry_type](base_spec).geometries

    geometries = []
    for index in range(count):
        parts = base_geometries[index * spec.part_count : (index + 1) * spec.part_count]
        if spec.geometry_type == "point":
            geometries.append(MultiPoint(parts))
        elif spec.geometry_type == "line":
            geometries.append(MultiLineString(parts))
        elif spec.geometry_type == "polygon":
            geometries.append(MultiPolygon(parts))
        else:
            raise ValueError(f"Unsupported multi geometry base type: {spec.geometry_type}")

    return _coerce_dataset(spec, geometries)


def generate_invalid_geometries(
    count: int | str = "1K",
    *,
    seed: int = 0,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1_000.0, 1_000.0),
    crs: str | None = None,
) -> SyntheticDataset:
    resolved_count = _resolve_count(count)
    xmin, ymin, xmax, ymax = bounds
    rng = _rng(seed)
    geometries: list[Polygon | LineString] = []
    for index in range(resolved_count):
        cx = float(rng.uniform(xmin, xmax))
        cy = float(rng.uniform(ymin, ymax))
        width = (xmax - xmin) / 50.0
        height = (ymax - ymin) / 50.0
        match index % 4:
            case 0:
                geometries.append(
                    Polygon(
                        [
                            (cx - width, cy - height),
                            (cx + width, cy + height),
                            (cx - width, cy + height),
                            (cx + width, cy - height),
                        ]
                    )
                )
            case 1:
                geometries.append(
                    LineString(
                        [
                            (cx - width, cy),
                            (cx, cy + height),
                            (cx + width, cy),
                            (cx, cy + height),
                        ]
                    )
                )
            case 2:
                geometries.append(
                    Polygon(
                        [
                            (cx - width, cy - height),
                            (cx + width, cy - height),
                            (cx + width, cy + height),
                            (cx + width, cy + height),
                            (cx - width, cy + height),
                        ]
                    )
                )
            case _:
                geometries.append(
                    Polygon(
                        [
                            (cx - width, cy - height),
                            (float("nan"), cy),
                            (cx + width, cy + height),
                            (cx - width, cy + height),
                        ]
                    )
                )

    return _coerce_dataset(
        SyntheticSpec(
            geometry_type="invalid",
            distribution="mixed-invalid",
            count=resolved_count,
            seed=seed,
            bounds=bounds,
            crs=crs,
        ),
        geometries,
    )


def generate_mixed_geometries(spec: SyntheticSpec) -> SyntheticDataset:
    if not spec.mix_ratios:
        raise ValueError("Mixed geometry generation requires non-empty mix_ratios")

    count = spec.resolved_count
    allocations = []
    remaining = count
    for index, (geometry_type, ratio) in enumerate(spec.mix_ratios):
        if index == len(spec.mix_ratios) - 1:
            allocation = remaining
        else:
            allocation = int(round(count * ratio))
            allocation = min(allocation, remaining)
            remaining -= allocation
        allocations.append((geometry_type, allocation))

    generated = []
    seed_offset = 0
    for geometry_type, allocation in allocations:
        if allocation <= 0:
            continue
        seed = spec.seed + seed_offset
        seed_offset += 1
        if geometry_type == "point":
            generated.extend(
                generate_points(
                    SyntheticSpec(
                        geometry_type="point",
                        distribution="uniform",
                        count=allocation,
                        seed=seed,
                        bounds=spec.bounds,
                        crs=spec.crs,
                    )
                ).geometries
            )
        elif geometry_type == "line":
            generated.extend(
                generate_lines(
                    SyntheticSpec(
                        geometry_type="line",
                        distribution="random-walk",
                        count=allocation,
                        seed=seed,
                        bounds=spec.bounds,
                        crs=spec.crs,
                        vertices=max(spec.vertices, 4),
                    )
                ).geometries
            )
        elif geometry_type == "polygon":
            generated.extend(
                generate_polygons(
                    SyntheticSpec(
                        geometry_type="polygon",
                        distribution="star",
                        count=allocation,
                        seed=seed,
                        bounds=spec.bounds,
                        crs=spec.crs,
                        vertices=max(spec.vertices, 5),
                    )
                ).geometries
            )
        else:
            raise ValueError(f"Unsupported mixed geometry type: {geometry_type}")

    return _coerce_dataset(spec, generated)
