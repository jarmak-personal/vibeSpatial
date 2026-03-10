from __future__ import annotations

import pytest

from vibespatial.testing import (
    clip_cases,
    get_degeneracy_corpus,
    overlay_cases,
    segment_cases,
    verify_clip_case,
    verify_degeneracy_corpus,
    verify_overlay_case,
    verify_segment_case,
)


def test_degeneracy_corpus_covers_phase5_risks() -> None:
    names = {case.name for case in get_degeneracy_corpus()}
    assert {
        "shared_vertex_lines",
        "collinear_overlap_lines",
        "donut_window_polygon",
        "duplicate_vertex_polygon",
        "bowtie_invalid_polygon",
        "touching_hole_invalid_polygon",
        "null_and_empty_polygon_rows",
    } <= names


@pytest.mark.parametrize("case", overlay_cases(), ids=lambda case: case.name)
def test_overlay_verification_matches_corpus(case) -> None:
    result = verify_overlay_case(case)
    expected = case.overlay_expectation

    assert result.rows_with_make_valid == expected.make_valid_rows
    if expected.raw_error_substring is None:
        assert result.raw_failed is False
        assert result.raw_error is None
    else:
        assert result.raw_failed is True
        assert expected.raw_error_substring in (result.raw_error or "")


@pytest.mark.parametrize("case", clip_cases(), ids=lambda case: case.name)
def test_clip_verification_matches_corpus(case) -> None:
    result = verify_clip_case(case)
    expected = case.clip_expectation

    assert result.outcome == expected.outcome
    if expected.rows is not None:
        assert result.rows == expected.rows
    if expected.error_substring is None:
        assert result.error is None
    else:
        assert expected.error_substring in (result.error or "")


@pytest.mark.parametrize("case", segment_cases(), ids=lambda case: case.name)
def test_segment_verification_matches_corpus(case) -> None:
    result = verify_segment_case(case)

    assert result.kind_names == case.segment_expectation.kind_names


def test_bulk_corpus_verification_groups_overlay_clip_and_segment_results() -> None:
    report = verify_degeneracy_corpus()

    assert set(report) == {"overlay", "clip", "segment"}
    assert len(report["overlay"]) >= 4
    assert len(report["clip"]) >= 4
    assert len(report["segment"]) >= 2
