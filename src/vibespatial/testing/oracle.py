from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, replace
from typing import Any

import vibespatial.api as geopandas
import numpy as np
from shapely.geometry import Point, Polygon

from vibespatial import ExecutionMode
from vibespatial.runtime import RuntimeSelection, select_runtime
from vibespatial.testing.synthetic import SyntheticSpec, generate_points, generate_polygons


ReferenceCallable = Any
OperationCallable = Any


@dataclass(frozen=True)
class OracleConfig:
    reference: ReferenceCallable | None = None
    rtol: float = 1e-7
    atol: float = 1e-9
    handle_empty: bool = True
    max_mismatches: int = 5
    check_determinism: bool = False
    repeat_count: int = 2


@dataclass(frozen=True)
class OracleMismatch:
    index: int
    actual: str
    expected: str
    context: tuple[str, ...]


@dataclass(frozen=True)
class OracleComparison:
    actual: tuple[Any, ...]
    expected: tuple[Any, ...]
    dispatch_mode: ExecutionMode
    selection: RuntimeSelection
    mismatches: tuple[OracleMismatch, ...]


@dataclass(frozen=True)
class OracleScenario:
    name: str
    operation: OperationCallable
    config: OracleConfig
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class OracleAssertionError(AssertionError):
    pass


def compare_with_shapely(
    *,
    reference: ReferenceCallable,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    handle_empty: bool = True,
    max_mismatches: int = 5,
    check_determinism: bool = False,
    repeat_count: int = 2,
):
    """Attach oracle comparison config to a pytest test function."""

    config = OracleConfig(
        reference=reference,
        rtol=rtol,
        atol=atol,
        handle_empty=handle_empty,
        max_mismatches=max_mismatches,
        check_determinism=check_determinism,
        repeat_count=repeat_count,
    )

    def decorator(func):
        setattr(func, "__oracle_config__", config)
        return func

    return decorator


def get_oracle_config(func: object) -> OracleConfig | None:
    return getattr(func, "__oracle_config__", None)


def _call_with_supported_kwargs(func: Any, *args: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(func)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        accepted_kwargs = kwargs
    else:
        accepted_kwargs = {
            name: value for name, value in kwargs.items() if name in signature.parameters
        }
    return func(*args, **accepted_kwargs)


def _to_sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    if isinstance(value, geopandas.GeoSeries):
        return tuple(value.tolist())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        converted = value.tolist()
        if isinstance(converted, list):
            return tuple(converted)
    return (value,)


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, np.generic) and np.isnan(value):
        return True
    return False


def _is_geometry(value: Any) -> bool:
    return hasattr(value, "geom_type") and hasattr(value, "is_empty")


def _geometry_equal(left: Any, right: Any, *, config: OracleConfig) -> bool:
    if config.handle_empty and left.is_empty and right.is_empty:
        return left.geom_type == right.geom_type
    return bool(left.equals_exact(right, tolerance=config.atol))


def _values_equal(left: Any, right: Any, *, config: OracleConfig) -> bool:
    if _is_null(left) or _is_null(right):
        return _is_null(left) and _is_null(right)
    if _is_geometry(left) and _is_geometry(right):
        return _geometry_equal(left, right, config=config)
    if isinstance(left, (float, int, np.number)) and isinstance(right, (float, int, np.number)):
        return math.isclose(float(left), float(right), rel_tol=config.rtol, abs_tol=config.atol)
    return left == right


def _render_value(value: Any, *, config: OracleConfig) -> str:
    if _is_null(value):
        return "NULL"
    if _is_geometry(value):
        if config.handle_empty and value.is_empty:
            return f"{value.geom_type}(EMPTY)"
        return value.wkt
    return repr(value)


def _summarize_inputs(args: tuple[Any, ...], index: int, result_length: int, *, config: OracleConfig) -> tuple[str, ...]:
    context: list[str] = []
    for position, argument in enumerate(args):
        values = _to_sequence(argument)
        if len(values) != result_length:
            continue
        context.append(f"arg{position}={_render_value(values[index], config=config)}")
    return tuple(context)


def _compare_sequences(
    actual: tuple[Any, ...],
    expected: tuple[Any, ...],
    args: tuple[Any, ...],
    *,
    config: OracleConfig,
) -> tuple[OracleMismatch, ...]:
    if len(actual) != len(expected):
        return (
            OracleMismatch(
                index=-1,
                actual=f"len={len(actual)}",
                expected=f"len={len(expected)}",
                context=(),
            ),
        )

    mismatches: list[OracleMismatch] = []
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        if _values_equal(left, right, config=config):
            continue
        mismatches.append(
            OracleMismatch(
                index=index,
                actual=_render_value(left, config=config),
                expected=_render_value(right, config=config),
                context=_summarize_inputs(args, index, len(actual), config=config),
            )
        )
        if len(mismatches) >= config.max_mismatches:
            break
    return tuple(mismatches)


def _raise_for_mismatches(mismatches: tuple[OracleMismatch, ...], *, heading: str) -> None:
    lines = [heading]
    for mismatch in mismatches:
        context = f" ({', '.join(mismatch.context)})" if mismatch.context else ""
        lines.append(
            f"- index {mismatch.index}: actual={mismatch.actual}, expected={mismatch.expected}{context}"
        )
    raise OracleAssertionError("\n".join(lines))


def assert_matches_shapely(
    operation: OperationCallable,
    *args: Any,
    dispatch_mode: ExecutionMode | str = ExecutionMode.CPU,
    config: OracleConfig | None = None,
    reference: ReferenceCallable | None = None,
    **kwargs: Any,
) -> OracleComparison:
    resolved_mode = dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    selection = select_runtime(resolved_mode)
    effective_config = replace(config or OracleConfig(), reference=reference or (config.reference if config else None))
    if effective_config.reference is None:
        raise ValueError("Reference oracle requires a reference callable")

    actual_raw = _call_with_supported_kwargs(operation, *args, dispatch_mode=resolved_mode, **kwargs)
    expected_raw = _call_with_supported_kwargs(effective_config.reference, *args, **kwargs)
    actual = _to_sequence(actual_raw)
    expected = _to_sequence(expected_raw)
    mismatches = _compare_sequences(actual, expected, args, config=effective_config)
    if mismatches:
        _raise_for_mismatches(
            mismatches,
            heading=f"Oracle comparison failed for dispatch mode {selection.selected.value}",
        )

    if effective_config.check_determinism and effective_config.repeat_count > 1:
        baseline = actual
        for repeat in range(1, effective_config.repeat_count):
            rerun = _to_sequence(
                _call_with_supported_kwargs(operation, *args, dispatch_mode=resolved_mode, **kwargs)
            )
            drift = _compare_sequences(rerun, baseline, args, config=effective_config)
            if drift:
                _raise_for_mismatches(
                    drift,
                    heading=(
                        f"Determinism check failed on repeat {repeat + 1} "
                        f"for dispatch mode {selection.selected.value}"
                    ),
                )

    return OracleComparison(
        actual=actual,
        expected=expected,
        dispatch_mode=resolved_mode,
        selection=selection,
        mismatches=(),
    )


def point_in_polygon_reference(
    points: list[Point | None],
    polygons: list[Polygon | None],
) -> list[bool | None]:
    results: list[bool | None] = []
    for point, polygon in zip(points, polygons, strict=True):
        if point is None or polygon is None:
            results.append(None)
            continue
        if point.is_empty or polygon.is_empty:
            results.append(False)
            continue
        results.append(bool(polygon.covers(point)))
    return results


def mock_point_in_polygon(
    points: list[Point | None],
    polygons: list[Polygon | None],
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.CPU,
) -> list[bool | None]:
    del dispatch_mode
    return point_in_polygon_reference(points, polygons)


def build_point_in_polygon_scenario(
    *,
    scale: int | str = "1K",
    seed: int = 0,
    check_determinism: bool = False,
) -> OracleScenario:
    points = list(
        generate_points(
            SyntheticSpec(
                geometry_type="point",
                distribution="uniform",
                count=scale,
                seed=seed,
                bounds=(0.0, 0.0, 1_000.0, 1_000.0),
            )
        ).geometries
    )
    polygons = list(
        generate_polygons(
            SyntheticSpec(
                geometry_type="polygon",
                distribution="regular-grid",
                count=max(1, len(points) // 8),
                seed=seed,
                bounds=(0.0, 0.0, 1_000.0, 1_000.0),
            )
        ).geometries
    )
    tiled_polygons = [polygons[index % len(polygons)] for index in range(len(points))]

    if points:
        points[0] = None
    if len(points) > 1:
        points[1] = Point()
    if len(tiled_polygons) > 2:
        tiled_polygons[2] = None
    if len(tiled_polygons) > 3:
        tiled_polygons[3] = Polygon()

    return OracleScenario(
        name="point_in_polygon",
        operation=mock_point_in_polygon,
        config=OracleConfig(
            reference=point_in_polygon_reference,
            handle_empty=True,
            check_determinism=check_determinism,
            max_mismatches=5,
        ),
        args=(points, tiled_polygons),
        kwargs={},
    )


ORACLE_SCENARIOS = {
    "point_in_polygon": build_point_in_polygon_scenario,
}

