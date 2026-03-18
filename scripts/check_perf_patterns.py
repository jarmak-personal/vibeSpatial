"""Performance anti-pattern lints (VPAT001-004).

Static analysis for code patterns known to cause GPU under-utilization or
host-side bottlenecks.  See ADR-0032 for the canonical example of a
Shapely serialization path dominating a nominally-GPU pipeline stage.

Uses a ratchet baseline: fails only when violations INCREASE beyond the
known debt count.  Decrease the baseline as debt is paid down.

Run:
    uv run python scripts/check_perf_patterns.py --all
"""
from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DOC = "docs/architecture/runtime.md"
ADR_0032 = "docs/decisions/0032-point-in-polygon-gpu-utilization-diagnosis.md"

# Known pre-existing violations as of 2026-03-17.
# Decrease this number as debt is paid.  The check fails only if
# the current count EXCEEDS the baseline (new violations introduced).
_VIOLATION_BASELINE = 11

# Attribute names that, when iterated, indicate Python-level geometry looping.
GEOMETRY_ITER_ATTRS = {"geoms", "geometry", "geometries", "exterior", "interiors"}

# Directories excluded from scanning.
_EXCLUDED_DIRS = {"api", "testing", "_vendor"}

# Filename stems/prefixes excluded from scanning (benchmarks, profiles).
_EXCLUDED_STEMS = {"pipeline_benchmarks", "profile_rails", "fixture_profiles"}
_EXCLUDED_PREFIXES = ("benchmark_", "profile_")


@dataclass(frozen=True)
class LintError:
    code: str
    path: Path
    line: int
    message: str
    doc_path: str

    def render(self, repo_root: Path) -> str:
        relative = self.path.relative_to(repo_root)
        return f"{relative}:{self.line}: {self.code} {self.message} See {self.doc_path}."


def _is_excluded(path: Path) -> bool:
    if any(d in path.parts for d in _EXCLUDED_DIRS):
        return True
    if path.stem in _EXCLUDED_STEMS:
        return True
    if any(path.name.startswith(p) for p in _EXCLUDED_PREFIXES):
        return True
    return False


def iter_python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and not _is_excluded(p)
    )


def parse_module(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_test_file(path: Path) -> bool:
    return "test" in path.stem or "tests" in path.parts or "conftest" in path.stem


# ---- VPAT001: Python for-loops iterating over geometry objects ----

def check_geometry_loops(repo_root: Path) -> list[LintError]:
    """Detect for-loops iterating over geometry attributes (host-side serial)."""
    errors: list[LintError] = []
    root = repo_root / "src" / "vibespatial"
    for path in iter_python_files(root):
        tree = parse_module(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.For):
                continue
            iter_node = node.iter
            if isinstance(iter_node, ast.Attribute) and iter_node.attr in GEOMETRY_ITER_ATTRS:
                errors.append(
                    LintError(
                        code="VPAT001",
                        path=path,
                        line=node.lineno,
                        message=(
                            f"Python for-loop iterates over .{iter_node.attr} -- this is serial "
                            "host-side geometry traversal. Use a bulk columnar kernel instead."
                        ),
                        doc_path=ADR_0032,
                    )
                )
            # enumerate(obj.geoms) etc.
            if (
                isinstance(iter_node, ast.Call)
                and isinstance(iter_node.func, ast.Name)
                and iter_node.func.id == "enumerate"
                and iter_node.args
            ):
                arg = iter_node.args[0]
                if isinstance(arg, ast.Attribute) and arg.attr in GEOMETRY_ITER_ATTRS:
                    errors.append(
                        LintError(
                            code="VPAT001",
                            path=path,
                            line=node.lineno,
                            message=(
                                f"Python for-loop enumerates .{arg.attr} -- this is serial "
                                "host-side geometry traversal. Use a bulk columnar kernel instead."
                            ),
                            doc_path=ADR_0032,
                        )
                    )
    return errors


# ---- VPAT002: Shapely calls in GPU kernel modules ----

def check_shapely_in_kernels(repo_root: Path) -> list[LintError]:
    """Detect shapely imports in kernel modules (indicates host round-trip)."""
    errors: list[LintError] = []
    root = repo_root / "src" / "vibespatial" / "kernels"
    for path in iter_python_files(root):
        if _is_test_file(path):
            continue
        tree = parse_module(path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = None
                if isinstance(node, ast.Import):
                    mod = node.names[0].name
                elif node.module:
                    mod = node.module
                if mod and mod.startswith("shapely"):
                    errors.append(
                        LintError(
                            code="VPAT002",
                            path=path,
                            line=node.lineno,
                            message=(
                                "Shapely import in kernel module -- Shapely operates on host "
                                "Python objects and forces D->H transfers. Use device-native "
                                "geometry primitives."
                            ),
                            doc_path=ADR_0032,
                        )
                    )
                    break
    return errors


# ---- VPAT003: np.fromiter in kernel/pipeline code ----

def check_slow_host_patterns(repo_root: Path) -> list[LintError]:
    """Detect np.fromiter in runtime code."""
    errors: list[LintError] = []
    root = repo_root / "src" / "vibespatial"
    for path in iter_python_files(root):
        if _is_test_file(path):
            continue
        tree = parse_module(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # np.fromiter(...)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "fromiter"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"np", "numpy"}
            ):
                errors.append(
                    LintError(
                        code="VPAT003",
                        path=path,
                        line=node.lineno,
                        message=(
                            "np.fromiter() iterates a Python generator element-by-element. "
                            "Use np.array() with pre-allocated data or vectorized construction."
                        ),
                        doc_path=ADR_0032,
                    )
                )
    return errors


# ---- VPAT004: .astype(object) on arrays ----

def check_object_dtype_cast(repo_root: Path) -> list[LintError]:
    """Detect .astype(object) which forces Python object boxing."""
    errors: list[LintError] = []
    root = repo_root / "src" / "vibespatial"
    for path in iter_python_files(root):
        if _is_test_file(path):
            continue
        tree = parse_module(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "astype"):
                continue
            if not node.args:
                continue
            arg = node.args[0]
            is_object = False
            if isinstance(arg, ast.Name) and arg.id == "object":
                is_object = True
            elif isinstance(arg, ast.Constant) and arg.value in {"object", "O"}:
                is_object = True
            if is_object:
                errors.append(
                    LintError(
                        code="VPAT004",
                        path=path,
                        line=node.lineno,
                        message=(
                            ".astype(object) forces Python object boxing on every element. "
                            "This destroys vectorized performance. Use typed arrays."
                        ),
                        doc_path=ADR_0032,
                    )
                )
    return errors


def run_checks(repo_root: Path) -> list[LintError]:
    errors: list[LintError] = []
    errors.extend(check_geometry_loops(repo_root))
    errors.extend(check_shapely_in_kernels(repo_root))
    errors.extend(check_slow_host_patterns(repo_root))
    errors.extend(check_object_dtype_cast(repo_root))
    return sorted(errors, key=lambda e: (str(e.path), e.line, e.code))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check for performance anti-patterns in GPU-first code."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Scan src/vibespatial/ for performance anti-patterns.",
    )
    args = parser.parse_args(argv)

    if not args.all:
        parser.error("pass --all")

    errors = run_checks(REPO_ROOT)
    count = len(errors)

    if count > _VIOLATION_BASELINE:
        for error in errors:
            print(error.render(REPO_ROOT))
        print(
            f"\nPerformance pattern checks FAILED: {count} violations found, "
            f"baseline is {_VIOLATION_BASELINE}. "
            f"New code introduced {count - _VIOLATION_BASELINE} violation(s).",
            file=sys.stderr,
        )
        return 1

    if count < _VIOLATION_BASELINE:
        print(
            f"Performance pattern checks passed ({count} known violations, "
            f"baseline {_VIOLATION_BASELINE}). "
            f"Debt reduced! Update _VIOLATION_BASELINE to {count}."
        )
    else:
        print(f"Performance pattern checks passed ({count} known violations, baseline holds).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
