"""Tests for the zero-copy linter inline suppression mechanism."""
from __future__ import annotations

from pathlib import Path

from scripts.check_zero_copy import (
    _VIOLATION_BASELINE,
    REPO_ROOT,
    _find_empty_suppression_lines,
    _parse_suppression_comments,
    check_boundary_type_leak,
    check_loop_transfers,
    check_pingpong_transfers,
    run_checks,
)


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Unit tests for the parser helpers
# ---------------------------------------------------------------------------


def test_parse_suppression_comments_extracts_reason() -> None:
    source = (
        "x = arr.get()  # zcopy:ok(batch D2H for DataFrame construction)\n"
        "y = arr.asnumpy()\n"
        "z = arr.get()  # zcopy:ok(OOM prevention)\n"
    )
    result = _parse_suppression_comments(source)
    assert result == {
        1: "batch D2H for DataFrame construction",
        3: "OOM prevention",
    }


def test_parse_suppression_comments_skips_empty_reason() -> None:
    source = "x = arr.get()  # zcopy:ok()\n"
    result = _parse_suppression_comments(source)
    assert result == {}


def test_parse_suppression_comments_handles_whitespace_in_marker() -> None:
    source = "x = arr.get()  #  zcopy:ok( spaced reason )\n"
    result = _parse_suppression_comments(source)
    assert result == {1: "spaced reason"}


def test_find_empty_suppression_lines() -> None:
    source = (
        "x = arr.get()  # zcopy:ok()\n"
        "y = arr.get()  # zcopy:ok(valid)\n"
        "z = arr.get()  # zcopy:ok(  )\n"
    )
    result = _find_empty_suppression_lines(source)
    assert result == [1, 3]


# ---------------------------------------------------------------------------
# ZCOPY001: Ping-pong suppression
# ---------------------------------------------------------------------------


def test_suppression_skips_pingpong(tmp_path: Path) -> None:
    """Lines with # zcopy:ok(reason) should not appear in ZCOPY001 violations."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def process(arr):
    host = arr.get()  # zcopy:ok(intentional batch D2H)
    result = cp.asarray(host)
    return result
""",
    )
    errors, suppressed = check_pingpong_transfers(tmp_path)
    assert errors == []
    assert suppressed == 1


def test_unsuppressed_pingpong_still_flagged(tmp_path: Path) -> None:
    """Lines without annotation should still be flagged normally."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def process(arr):
    host = arr.get()
    result = cp.asarray(host)
    return result
""",
    )
    errors, suppressed = check_pingpong_transfers(tmp_path)
    assert len(errors) == 1
    assert errors[0].code == "ZCOPY001"
    assert suppressed == 0


# ---------------------------------------------------------------------------
# ZCOPY002: Loop transfer suppression
# ---------------------------------------------------------------------------


def test_suppression_skips_loop_transfer(tmp_path: Path) -> None:
    """Lines with # zcopy:ok(reason) should not appear in ZCOPY002 violations."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def process(chunks):
    for chunk in chunks:
        host = chunk.get()  # zcopy:ok(OOM prevention: stream chunks to host)
        do_stuff(host)
""",
    )
    errors, suppressed = check_loop_transfers(tmp_path)
    assert errors == []
    assert suppressed == 1


def test_unsuppressed_loop_transfer_still_flagged(tmp_path: Path) -> None:
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def process(chunks):
    for chunk in chunks:
        host = chunk.get()
        do_stuff(host)
""",
    )
    errors, suppressed = check_loop_transfers(tmp_path)
    assert len(errors) == 1
    assert errors[0].code == "ZCOPY002"
    assert suppressed == 0


# ---------------------------------------------------------------------------
# ZCOPY003: Boundary type leak suppression
# ---------------------------------------------------------------------------


def test_suppression_skips_boundary_leak(tmp_path: Path) -> None:
    """Lines with # zcopy:ok(reason) should not appear in ZCOPY003 violations."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def compute(device_array):
    result = cp.launch(device_array)
    return result.get()  # zcopy:ok(caller needs host array for pandas ctor)
""",
    )
    errors, suppressed = check_boundary_type_leak(tmp_path)
    assert errors == []
    assert suppressed == 1


def test_unsuppressed_boundary_leak_still_flagged(tmp_path: Path) -> None:
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def compute(device_array):
    result = cp.launch(device_array)
    return result.get()
""",
    )
    errors, suppressed = check_boundary_type_leak(tmp_path)
    assert len(errors) == 1
    assert errors[0].code == "ZCOPY003"
    assert suppressed == 0


# ---------------------------------------------------------------------------
# Empty reason rejection
# ---------------------------------------------------------------------------


def test_empty_reason_not_suppressed(tmp_path: Path) -> None:
    """# zcopy:ok() with empty reason should NOT suppress the violation."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def process(chunks):
    for chunk in chunks:
        host = chunk.get()  # zcopy:ok()
        do_stuff(host)
""",
    )
    errors, suppressed = run_checks(tmp_path)
    codes = [e.code for e in errors]
    # The loop transfer is still flagged (not suppressed) AND
    # an empty-suppression error is emitted.
    assert "ZCOPY002" in codes
    assert "ZCOPY_EMPTY_SUPPRESSION" in codes
    assert suppressed == 0


def test_empty_reason_whitespace_only_not_suppressed(tmp_path: Path) -> None:
    """# zcopy:ok(   ) with whitespace-only reason is treated as empty."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def process(chunks):
    for chunk in chunks:
        host = chunk.get()  # zcopy:ok(   )
        do_stuff(host)
""",
    )
    errors, suppressed = run_checks(tmp_path)
    codes = [e.code for e in errors]
    assert "ZCOPY002" in codes
    assert "ZCOPY_EMPTY_SUPPRESSION" in codes
    assert suppressed == 0


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------


def test_suppression_count_in_summary(tmp_path: Path, capsys) -> None:
    """Summary output should report suppressed count separately."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def process(arr):
    host = arr.get()  # zcopy:ok(intentional batch D2H)
    result = cp.asarray(host)
    return result
""",
    )
    _, suppressed = run_checks(tmp_path)
    assert suppressed == 1


# ---------------------------------------------------------------------------
# Integration: no change to real repo violation count
# ---------------------------------------------------------------------------


def test_real_repo_still_at_baseline() -> None:
    """Running against the real repo should still report the baseline count.

    No annotations have been added yet, so the count must be unchanged.
    """
    errors, suppressed = run_checks(REPO_ROOT)
    assert len(errors) == _VIOLATION_BASELINE, (
        f"Expected {_VIOLATION_BASELINE} violations, got {len(errors)}. "
        f"Suppressed: {suppressed}."
    )
    assert suppressed == 0
