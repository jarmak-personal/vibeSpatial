from __future__ import annotations

from pathlib import Path

from scripts.check_dispatch_once import run_checks


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_import_guard_is_ignored(tmp_path: Path) -> None:
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
""",
    )
    assert run_checks(tmp_path) == []


def test_gpu_to_cpu_fallback_is_flagged(tmp_path: Path) -> None:
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def f(left, right):
    try:
        return binary_constructive_owned("intersection", left, right)
    except NotImplementedError:
        return shapely.intersection(left.to_shapely(), right.to_shapely())
""",
    )
    errors = run_checks(tmp_path)
    assert len(errors) == 1
    assert errors[0].code == "DO001"


def test_gpu_to_cpu_fallback_via_cpu_helper_is_flagged(tmp_path: Path) -> None:
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def f(arr):
    try:
        return point_query_gpu(arr)
    except Exception:
        return point_query_cpu(arr)
""",
    )
    errors = run_checks(tmp_path)
    assert len(errors) == 1
    assert errors[0].code == "DO001"


def test_bare_exception_pass_is_flagged(tmp_path: Path) -> None:
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def f():
    try:
        point_query_gpu()
    except Exception:
        pass
""",
    )
    errors = run_checks(tmp_path)
    assert len(errors) == 1
    assert errors[0].code == "DO002"


def test_non_cpu_except_body_is_not_flagged(tmp_path: Path) -> None:
    _write_file(
        tmp_path / "src" / "vibespatial" / "mod.py",
        """\
def f(arr):
    try:
        return point_query_gpu(arr)
    except RuntimeError:
        return None
""",
    )
    assert run_checks(tmp_path) == []
