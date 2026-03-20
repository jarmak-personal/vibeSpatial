from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from shapely.geometry import LineString, Polygon, box

import vibespatial.api as geopandas
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.spatial.segment_primitives import classify_segment_intersections

ClipOutcome = Literal["nonempty", "empty", "error"]


@dataclass(frozen=True)
class OverlayExpectation:
    make_valid_rows: int
    raw_error_substring: str | None = None


@dataclass(frozen=True)
class ClipExpectation:
    outcome: ClipOutcome
    rows: int | None = None
    error_substring: str | None = None


@dataclass(frozen=True)
class SegmentExpectation:
    kind_names: tuple[str, ...]


@dataclass(frozen=True)
class DegeneracyCase:
    name: str
    category: str
    description: str
    left_geometries: tuple[object | None, ...]
    right_geometries: tuple[object | None, ...] = ()
    clip_mask: object | None = None
    overlay_expectation: OverlayExpectation | None = None
    clip_expectation: ClipExpectation | None = None
    segment_expectation: SegmentExpectation | None = None


@dataclass(frozen=True)
class OverlayVerification:
    name: str
    rows_with_make_valid: int
    raw_failed: bool
    raw_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClipVerification:
    name: str
    outcome: ClipOutcome
    rows: int
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentVerification:
    name: str
    kind_names: tuple[str, ...]
    ambiguous_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_geodataframe(geometries: tuple[object | None, ...], *, prefix: str) -> geopandas.GeoDataFrame:
    return geopandas.GeoDataFrame(
        {f"{prefix}_id": list(range(len(geometries)))},
        geometry=list(geometries),
        crs="EPSG:3857",
    )


def _corpus() -> tuple[DegeneracyCase, ...]:
    donut = Polygon(
        shell=[(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
        holes=[[(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]],
    )
    duplicate_vertex = Polygon([(0, 0), (4, 0), (4, 4), (4, 4), (0, 4), (0, 0)])
    bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
    touching_hole = Polygon(
        shell=[(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
        holes=[[(0, 2), (2, 2), (2, 4), (0, 4), (0, 2)]],
    )
    return (
        DegeneracyCase(
            name="shared_vertex_lines",
            category="segment-touch",
            description="Two lines meet at one endpoint and should classify as touch.",
            left_geometries=(LineString([(0, 0), (2, 2)]),),
            right_geometries=(LineString([(2, 2), (4, 0)]),),
            clip_mask=box(1.5, 1.5, 2.5, 2.5),
            overlay_expectation=OverlayExpectation(make_valid_rows=1),
            clip_expectation=ClipExpectation(outcome="nonempty", rows=1),
            segment_expectation=SegmentExpectation(kind_names=("touch",)),
        ),
        DegeneracyCase(
            name="collinear_overlap_lines",
            category="segment-overlap",
            description="Two lines share a collinear span and should classify as overlap.",
            left_geometries=(LineString([(0, 0), (5, 0)]),),
            right_geometries=(LineString([(2, 0), (7, 0)]),),
            clip_mask=box(1, -1, 6, 1),
            overlay_expectation=OverlayExpectation(make_valid_rows=1),
            clip_expectation=ClipExpectation(outcome="nonempty", rows=1),
            segment_expectation=SegmentExpectation(kind_names=("overlap",)),
        ),
        DegeneracyCase(
            name="donut_window_polygon",
            category="hole-clip",
            description="A polygon with a hole is clipped and overlaid against a window crossing the hole.",
            left_geometries=(donut,),
            right_geometries=(box(1, 1, 5, 5),),
            clip_mask=box(1, 1, 5, 5),
            overlay_expectation=OverlayExpectation(make_valid_rows=1),
            clip_expectation=ClipExpectation(outcome="nonempty", rows=1),
        ),
        DegeneracyCase(
            name="duplicate_vertex_polygon",
            category="duplicate-vertex",
            description="A polygon with repeated adjacent vertices should remain stable under overlay and clip.",
            left_geometries=(duplicate_vertex,),
            right_geometries=(box(2, 2, 5, 5),),
            clip_mask=box(1, 1, 3, 3),
            overlay_expectation=OverlayExpectation(make_valid_rows=1),
            clip_expectation=ClipExpectation(outcome="nonempty", rows=1),
        ),
        DegeneracyCase(
            name="bowtie_invalid_polygon",
            category="invalid-self-intersection",
            description="A self-intersecting bowtie should fail raw overlay/clip and only pass overlay with make_valid.",
            left_geometries=(bowtie,),
            right_geometries=(box(0, 0, 2, 2),),
            clip_mask=box(0, 0, 2, 2),
            overlay_expectation=OverlayExpectation(
                make_valid_rows=1,
                raw_error_substring="invalid input geometries",
            ),
            clip_expectation=ClipExpectation(
                outcome="error",
                error_substring="TopologyException",
            ),
        ),
        DegeneracyCase(
            name="touching_hole_invalid_polygon",
            category="touching-rings",
            description="A polygon whose hole touches the shell should require make_valid handling for overlay.",
            left_geometries=(touching_hole,),
            right_geometries=(box(1, 1, 5, 5),),
            clip_mask=box(1, 1, 5, 5),
            overlay_expectation=OverlayExpectation(
                make_valid_rows=1,
                raw_error_substring="invalid input geometries",
            ),
            clip_expectation=ClipExpectation(outcome="nonempty", rows=1),
        ),
        DegeneracyCase(
            name="null_and_empty_polygon_rows",
            category="null-empty",
            description="Null and empty polygon rows should disappear cleanly from overlay and clip outputs.",
            left_geometries=(None, Polygon()),
            right_geometries=(box(0, 0, 2, 2),),
            clip_mask=box(0, 0, 2, 2),
            overlay_expectation=OverlayExpectation(make_valid_rows=0),
            clip_expectation=ClipExpectation(outcome="empty", rows=0),
        ),
    )


DEGENERACY_CORPUS = _corpus()


def get_degeneracy_corpus() -> tuple[DegeneracyCase, ...]:
    return DEGENERACY_CORPUS


def overlay_cases() -> tuple[DegeneracyCase, ...]:
    return tuple(case for case in DEGENERACY_CORPUS if case.overlay_expectation is not None)


def clip_cases() -> tuple[DegeneracyCase, ...]:
    return tuple(case for case in DEGENERACY_CORPUS if case.clip_expectation is not None and case.clip_mask is not None)


def segment_cases() -> tuple[DegeneracyCase, ...]:
    return tuple(case for case in DEGENERACY_CORPUS if case.segment_expectation is not None)


def verify_overlay_case(case: DegeneracyCase) -> OverlayVerification:
    if case.overlay_expectation is None:
        raise ValueError(f"{case.name} has no overlay expectation")
    if not case.right_geometries:
        raise ValueError(f"{case.name} requires right geometries for overlay verification")

    left = _build_geodataframe(case.left_geometries, prefix="left")
    right = _build_geodataframe(case.right_geometries, prefix="right")

    raw_failed = False
    raw_error: str | None = None
    if case.overlay_expectation.raw_error_substring is not None:
        try:
            geopandas.overlay(left, right, how="intersection", make_valid=False, keep_geom_type=False)
        except Exception as exc:
            raw_failed = True
            raw_error = str(exc)
        else:
            raw_error = None

    result = geopandas.overlay(left, right, how="intersection", make_valid=True, keep_geom_type=False)
    return OverlayVerification(
        name=case.name,
        rows_with_make_valid=len(result),
        raw_failed=raw_failed,
        raw_error=raw_error,
    )


def verify_clip_case(case: DegeneracyCase) -> ClipVerification:
    if case.clip_expectation is None or case.clip_mask is None:
        raise ValueError(f"{case.name} has no clip expectation")

    left = _build_geodataframe(case.left_geometries, prefix="left")
    try:
        result = geopandas.clip(left, case.clip_mask)
    except Exception as exc:
        return ClipVerification(name=case.name, outcome="error", rows=0, error=str(exc))

    outcome: ClipOutcome = "empty" if len(result) == 0 else "nonempty"
    return ClipVerification(name=case.name, outcome=outcome, rows=len(result), error=None)


def verify_segment_case(case: DegeneracyCase) -> SegmentVerification:
    if case.segment_expectation is None:
        raise ValueError(f"{case.name} has no segment expectation")
    if not case.right_geometries:
        raise ValueError(f"{case.name} requires right geometries for segment verification")

    left = from_shapely_geometries(list(case.left_geometries))
    right = from_shapely_geometries(list(case.right_geometries))
    result = classify_segment_intersections(left, right)
    return SegmentVerification(
        name=case.name,
        kind_names=tuple(result.kind_names()),
        ambiguous_rows=int(result.ambiguous_rows.size),
    )


def verify_degeneracy_corpus() -> dict[str, list[dict[str, Any]]]:
    return {
        "overlay": [verify_overlay_case(case).to_dict() for case in overlay_cases()],
        "clip": [verify_clip_case(case).to_dict() for case in clip_cases()],
        "segment": [verify_segment_case(case).to_dict() for case in segment_cases()],
    }
