"""Tests for the zero-copy linter inline suppression mechanism."""
from __future__ import annotations

from pathlib import Path

from scripts.check_zero_copy import (
    _VIOLATION_BASELINE,
    REPO_ROOT,
    _collect_numpy_names,
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


# ---------------------------------------------------------------------------
# Receiver-type analysis: numpy false positive elimination
# ---------------------------------------------------------------------------


def test_numpy_tolist_not_flagged_in_loop(tmp_path: Path) -> None:
    """np.flatnonzero().tolist() inside a loop should NOT be flagged."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
import numpy as np

def process(data):
    mask = data > 0
    for idx in np.flatnonzero(mask).tolist():
        do_stuff(idx)
""",
    )
    errors, suppressed = check_loop_transfers(tmp_path)
    assert errors == []
    assert suppressed == 0


def test_numpy_param_tolist_not_flagged(tmp_path: Path) -> None:
    """arr.tolist() where arr is annotated np.ndarray should NOT be flagged."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
import numpy as np

def process(rows: np.ndarray):
    for idx in rows.tolist():
        do_stuff(idx)
""",
    )
    errors, suppressed = check_loop_transfers(tmp_path)
    assert errors == []
    assert suppressed == 0


def test_numpy_union_annotation_not_flagged(tmp_path: Path) -> None:
    """arr.tolist() where arr: np.ndarray | None should NOT be flagged."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
import numpy as np

def process(values: np.ndarray | None):
    if values is None:
        return
    for v in values.tolist():
        do_stuff(v)
""",
    )
    errors, suppressed = check_loop_transfers(tmp_path)
    assert errors == []
    assert suppressed == 0


def test_numpy_assigned_var_not_flagged(tmp_path: Path) -> None:
    """var.tolist() where var = np.diff(...) should NOT be flagged."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
import numpy as np

def process(offsets):
    counts = np.diff(offsets)
    for c in counts.tolist():
        accumulate(c)
""",
    )
    errors, suppressed = check_loop_transfers(tmp_path)
    assert errors == []
    assert suppressed == 0


def test_numpy_subscript_not_flagged(tmp_path: Path) -> None:
    """arr[:-1].tolist() where arr is derived from numpy should NOT be flagged."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
import numpy as np

def process(ring_offsets: np.ndarray):
    local = ring_offsets[1:5]
    for val in local[:-1].tolist():
        use(val)
""",
    )
    errors, suppressed = check_loop_transfers(tmp_path)
    assert errors == []
    assert suppressed == 0


def test_numpy_arithmetic_propagation_not_flagged(tmp_path: Path) -> None:
    """var.tolist() where var = numpy_arr - scalar should NOT be flagged."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
import numpy as np

def process(offsets: np.ndarray, start: int):
    rebased = offsets - start
    for val in rebased.tolist():
        use(val)
""",
    )
    errors, suppressed = check_loop_transfers(tmp_path)
    assert errors == []
    assert suppressed == 0


def test_cupy_tolist_still_flagged(tmp_path: Path) -> None:
    """cupy_array.tolist() should still be flagged as D2H."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
import cupy as cp

def process(data):
    result = cp.arange(10)
    for val in result.tolist():
        use(val)
""",
    )
    errors, suppressed = check_loop_transfers(tmp_path)
    assert len(errors) == 1
    assert errors[0].code == "ZCOPY002"
    assert suppressed == 0


def test_unknown_receiver_still_flagged(tmp_path: Path) -> None:
    """unknown_var.tolist() should still be flagged (conservative)."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def process(data):
    result = compute_something(data)
    for val in result.tolist():
        use(val)
""",
    )
    errors, suppressed = check_loop_transfers(tmp_path)
    assert len(errors) == 1
    assert errors[0].code == "ZCOPY002"
    assert suppressed == 0


def test_shapely_result_tolist_not_flagged(tmp_path: Path) -> None:
    """shapely.func().tolist() should NOT be flagged (host-to-host)."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
import shapely

def process(geoms):
    inter = shapely.intersection(geoms, geoms)
    for g in inter.tolist():
        use(g)
""",
    )
    errors, suppressed = check_loop_transfers(tmp_path)
    assert errors == []
    assert suppressed == 0


def test_numpy_pingpong_false_positive_eliminated(tmp_path: Path) -> None:
    """numpy_param.tolist() followed by cp.asarray() is NOT a ping-pong.

    The .tolist() on a numpy array is host-to-host, so the subsequent
    cp.asarray() is a fresh H2D transfer, not a D2H->H2D ping-pong.
    """
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
import numpy as np

def process(invalid_rows: np.ndarray):
    invalid_set = set(invalid_rows.tolist())
    valid = np.array([i for i in range(100) if i not in invalid_set])
    d_valid = cp.asarray(valid)
    return d_valid
""",
    )
    errors, suppressed = check_pingpong_transfers(tmp_path)
    assert errors == []
    assert suppressed == 0


# ---------------------------------------------------------------------------
# _collect_numpy_names unit tests
# ---------------------------------------------------------------------------


def test_collect_numpy_names_simple_param() -> None:
    import ast
    source = "def f(x: np.ndarray, y: int): pass"
    tree = ast.parse(source)
    func = tree.body[0]
    names = _collect_numpy_names(func)
    assert "x" in names
    assert "y" not in names


def test_collect_numpy_names_union_param() -> None:
    import ast
    source = "def f(x: np.ndarray | None): pass"
    tree = ast.parse(source)
    func = tree.body[0]
    names = _collect_numpy_names(func)
    assert "x" in names


def test_collect_numpy_names_np_call() -> None:
    import ast
    source = """\
def f():
    x = np.flatnonzero(mask)
    y = something_else()
"""
    tree = ast.parse(source)
    func = tree.body[0]
    names = _collect_numpy_names(func)
    assert "x" in names
    assert "y" not in names


def test_collect_numpy_names_subscript_propagation() -> None:
    import ast
    source = """\
def f(offsets: np.ndarray):
    local = offsets[1:5]
"""
    tree = ast.parse(source)
    func = tree.body[0]
    names = _collect_numpy_names(func)
    assert "offsets" in names
    assert "local" in names


def test_collect_numpy_names_arithmetic_propagation() -> None:
    import ast
    source = """\
def f(offsets: np.ndarray):
    local = offsets[1:5]
    rebased = local - 10 + 20
"""
    tree = ast.parse(source)
    func = tree.body[0]
    names = _collect_numpy_names(func)
    assert "rebased" in names


# ---------------------------------------------------------------------------
# INFRA-04: PyArrow to_pylist() false positive elimination
# ---------------------------------------------------------------------------


def test_pyarrow_to_pylist_not_flagged_in_loop(tmp_path: Path) -> None:
    """to_pylist() in a function that uses pyarrow .field() should NOT be flagged.

    Covers the pattern in geojson.py where .field(0) result is stored in
    an intermediate variable and then .to_pylist() is called on it.
    """
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def process(feature_struct):
    for index in range(feature_struct.type.num_fields):
        child = feature_struct.field(index)
        maybe_type = child.field(0)
        values = [v for v in maybe_type.to_pylist() if v is not None]
        if values:
            return index
    return None
""",
    )
    errors, suppressed = check_loop_transfers(tmp_path)
    assert errors == []
    assert suppressed == 0


def test_pyarrow_direct_field_to_pylist_not_flagged(tmp_path: Path) -> None:
    """struct.field(0).to_pylist() should NOT be flagged (existing heuristic)."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def process(table):
    for col_idx in range(table.num_columns):
        vals = table.column(col_idx).to_pylist()
        use(vals)
""",
    )
    errors, suppressed = check_loop_transfers(tmp_path)
    assert errors == []
    assert suppressed == 0


def test_cupy_to_pylist_still_flagged_with_pyarrow(tmp_path: Path) -> None:
    """cp.something.to_pylist() should still be flagged even with pyarrow in scope."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def process(data):
    import pyarrow as pa
    result = cp.arange(10)
    for val in result.to_pylist():
        use(val)
""",
    )
    errors, suppressed = check_loop_transfers(tmp_path)
    assert len(errors) == 1
    assert errors[0].code == "ZCOPY002"


def test_pyarrow_to_pylist_not_flagged_in_pingpong(tmp_path: Path) -> None:
    """to_pylist() on pyarrow data should not count as D2H in ping-pong detection."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def process(struct, data):
    child = struct.field(0)
    values = child.to_pylist()
    result = cp.asarray(data)
    return result
""",
    )
    errors, suppressed = check_pingpong_transfers(tmp_path)
    assert errors == []
    assert suppressed == 0


# ---------------------------------------------------------------------------
# INFRA-05: Branch mutual exclusion for ZCOPY001
# ---------------------------------------------------------------------------


def test_branch_exclusion_returning_if(tmp_path: Path) -> None:
    """D2H in an if-branch that returns should not pair with H2D after the branch."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def process(data, flag):
    if flag is None:
        host = data.to_host()
        return host

    result = cp.asarray(data)
    return result
""",
    )
    errors, suppressed = check_pingpong_transfers(tmp_path)
    assert errors == []
    assert suppressed == 0


def test_branch_exclusion_non_returning_if_still_flagged(tmp_path: Path) -> None:
    """D2H in an if-branch that does NOT return should still pair with later H2D."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def process(data, flag):
    if flag is None:
        host = data.to_host()
        use(host)

    result = cp.asarray(data)
    return result
""",
    )
    errors, suppressed = check_pingpong_transfers(tmp_path)
    assert len(errors) == 1
    assert errors[0].code == "ZCOPY001"
    assert suppressed == 0


def test_branch_exclusion_else_with_return(tmp_path: Path) -> None:
    """D2H in else-branch that returns should not pair with H2D after the branch."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def process(data, flag):
    if flag:
        pass
    else:
        host = data.to_host()
        return host

    result = cp.asarray(data)
    return result
""",
    )
    errors, suppressed = check_pingpong_transfers(tmp_path)
    assert errors == []
    assert suppressed == 0


def test_branch_exclusion_both_in_same_branch_still_flagged(tmp_path: Path) -> None:
    """D2H and H2D in the same branch should still be flagged."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def process(data, flag):
    if flag:
        host = data.to_host()
        result = cp.asarray(host)
        return result
    return None
""",
    )
    errors, suppressed = check_pingpong_transfers(tmp_path)
    assert len(errors) == 1
    assert errors[0].code == "ZCOPY001"
    assert suppressed == 0


def test_branch_exclusion_nested_if_with_return(tmp_path: Path) -> None:
    """D2H in nested returning if-block should be excluded from pairing."""
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def process(data, pred, total):
    if pred is None or total == 0:
        if data is not None:
            left, right = data.to_host()
        if pred is None:
            return left, right
        return left, right

    result = cp.asarray(data)
    return result
""",
    )
    errors, suppressed = check_pingpong_transfers(tmp_path)
    assert errors == []
    assert suppressed == 0
