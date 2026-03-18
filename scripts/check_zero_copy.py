"""Zero-copy enforcement lints (ZCOPY001-003).

Detects device-to-host transfer anti-patterns that violate the GPU-first
execution model.  Extends ARCH004 (which bans D/H transfers outside
materialization methods) with deeper pattern analysis for ping-pongs,
per-element transfers in loops, and boundary type leaks.

Uses a ratchet baseline: fails only when violations INCREASE beyond the
known debt count.  Decrease the baseline as debt is paid down.

Run:
    uv run python scripts/check_zero_copy.py --all
"""
from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DOC = "docs/architecture/runtime.md"

# Known pre-existing violations as of 2026-03-17.
# Decrease this number as debt is paid.  The check fails only if
# the current count EXCEEDS the baseline (new violations introduced).
_VIOLATION_BASELINE = 55

# Method names that pull data from device to host.
D2H_APIS = {"get", "copy_to_host", "to_host", "asnumpy", "tolist", "to_pylist"}

# Method / function names that push data from host to device.
H2D_APIS = {"asarray", "array", "to_device", "as_cupy", "to_gpu"}

# Modules whose attribute calls count as H2D when wrapping host data.
H2D_MODULES = {"cp", "cupy"}

# Directories excluded from scanning (dispatch boundary, instrumentation).
_EXCLUDED_DIRS = {"api", "testing", "_vendor"}

# Filename stems excluded from scanning (benchmarks, profiles, IO boundary).
_EXCLUDED_STEMS = {
    "pipeline_benchmarks", "profile_rails", "fixture_profiles",
}

# Filename prefixes excluded from scanning.
_EXCLUDED_PREFIXES = ("benchmark_", "io_", "profile_")


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


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_d2h_call(node: ast.Call) -> bool:
    name = _call_name(node)
    return name in D2H_APIS


def _is_h2d_call(node: ast.Call) -> bool:
    name = _call_name(node)
    if name in H2D_APIS:
        return True
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        if node.func.value.id in H2D_MODULES and node.func.attr in H2D_APIS:
            return True
    return False


# ---- ZCOPY001: Ping-pong transfers in the same function ----

def check_pingpong_transfers(repo_root: Path) -> list[LintError]:
    """Find functions where data goes D->H then back H->D (or vice versa)."""
    errors: list[LintError] = []
    root = repo_root / "src" / "vibespatial"
    for path in iter_python_files(root):
        tree = parse_module(path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            d2h_lines: list[int] = []
            h2d_lines: list[int] = []
            for descendant in ast.walk(node):
                if not isinstance(descendant, ast.Call):
                    continue
                if _is_d2h_call(descendant):
                    d2h_lines.append(descendant.lineno)
                if _is_h2d_call(descendant):
                    h2d_lines.append(descendant.lineno)
            if d2h_lines and h2d_lines:
                first_d2h = min(d2h_lines)
                later_h2d = [ln for ln in h2d_lines if ln > first_d2h]
                if later_h2d:
                    errors.append(
                        LintError(
                            code="ZCOPY001",
                            path=path,
                            line=first_d2h,
                            message=(
                                f"Ping-pong transfer: D->H at line {first_d2h} followed by "
                                f"H->D at line {later_h2d[0]} in {node.name}(). "
                                "Keep data on device; see execution_trace.py for runtime detection."
                            ),
                            doc_path=RUNTIME_DOC,
                        )
                    )
    return errors


# ---- ZCOPY002: Per-element device transfers in loops ----

def check_loop_transfers(repo_root: Path) -> list[LintError]:
    """Find D->H transfer calls inside for/while loop bodies."""
    errors: list[LintError] = []
    root = repo_root / "src" / "vibespatial"
    for path in iter_python_files(root):
        tree = parse_module(path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.For, ast.While)):
                continue
            for descendant in ast.walk(node):
                if not isinstance(descendant, ast.Call):
                    continue
                if _is_d2h_call(descendant):
                    errors.append(
                        LintError(
                            code="ZCOPY002",
                            path=path,
                            line=descendant.lineno,
                            message=(
                                f"D->H transfer ({_call_name(descendant)}()) inside a loop body. "
                                "Transfer in bulk outside the loop instead of per-element."
                            ),
                            doc_path=RUNTIME_DOC,
                        )
                    )
    return errors


# ---- ZCOPY003: Functions that accept device data but return host data ----

def _returns_host_conversion(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int | None:
    """Return the line number of a D->H call in a return statement, or None."""
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        for descendant in ast.walk(node.value):
            if isinstance(descendant, ast.Call) and _is_d2h_call(descendant):
                return descendant.lineno
    return None


def _uses_device_apis(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Heuristic: does this function reference cupy / device APIs?"""
    for node in ast.walk(func_node):
        if isinstance(node, ast.Name) and node.id in {"cp", "cupy"}:
            return True
        if isinstance(node, ast.Attribute) and node.attr in {
            "launch", "compile_kernels", "device_array", "DeviceGeometryArray",
        }:
            return True
    return False


_ALLOWED_HOST_RETURN = {
    "to_pandas", "to_numpy", "values", "__repr__", "__str__",
    "to_pylist", "tolist", "to_host", "to_wkt", "to_wkb",
    "_to_host_array", "_ensure_host_state",
}


def check_boundary_type_leak(repo_root: Path) -> list[LintError]:
    """Find functions using device APIs that return D->H converted results."""
    errors: list[LintError] = []
    root = repo_root / "src" / "vibespatial"
    for path in iter_python_files(root):
        tree = parse_module(path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name in _ALLOWED_HOST_RETURN or node.name.startswith("_"):
                continue
            if not _uses_device_apis(node):
                continue
            d2h_line = _returns_host_conversion(node)
            if d2h_line is not None:
                errors.append(
                    LintError(
                        code="ZCOPY003",
                        path=path,
                        line=d2h_line,
                        message=(
                            f"{node.name}() uses device APIs but returns host-converted data. "
                            "Return device arrays and let the caller materialize."
                        ),
                        doc_path=RUNTIME_DOC,
                    )
                )
    return errors


def run_checks(repo_root: Path) -> list[LintError]:
    errors: list[LintError] = []
    errors.extend(check_pingpong_transfers(repo_root))
    errors.extend(check_loop_transfers(repo_root))
    errors.extend(check_boundary_type_leak(repo_root))
    return sorted(errors, key=lambda e: (str(e.path), e.line, e.code))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check zero-copy device transfer constraints."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Scan src/vibespatial/ for zero-copy violations.",
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
            f"\nZero-copy checks FAILED: {count} violations found, "
            f"baseline is {_VIOLATION_BASELINE}. "
            f"New code introduced {count - _VIOLATION_BASELINE} violation(s).",
            file=sys.stderr,
        )
        return 1

    if count < _VIOLATION_BASELINE:
        print(
            f"Zero-copy checks passed ({count} known violations, baseline {_VIOLATION_BASELINE}). "
            f"Debt reduced! Update _VIOLATION_BASELINE to {count}."
        )
    else:
        print(f"Zero-copy checks passed ({count} known violations, baseline holds).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
