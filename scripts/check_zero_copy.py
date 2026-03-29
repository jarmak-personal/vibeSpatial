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
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DOC = "docs/architecture/runtime.md"

# Known pre-existing violations as of 2026-03-27.
# Decrease this number as debt is paid.  The check fails only if
# the current count EXCEEDS the baseline (new violations introduced).
_VIOLATION_BASELINE = 45  # false positives eliminated 2026-03-29: dict.get(), np.asarray H2D
# +3 intentional batch D->H in segment_primitives (OOM prevention)
# +4 materialization D->H in dbf_gpu (DataFrame construction)
# +1 intentional bulk D->H in csv_gpu _extract_wkb_and_parse (hex decode on CPU)

# Method names that pull data from device to host.
D2H_APIS = {"get", "copy_to_host", "to_host", "asnumpy", "tolist", "to_pylist"}

# Method / function names that push data from host to device.
H2D_APIS = {"asarray", "array", "to_device", "as_cupy", "to_gpu"}

# Modules whose attribute calls count as H2D when wrapping host data.
H2D_MODULES = {"cp", "cupy"}

# Directories excluded from scanning (dispatch boundary, instrumentation).
_EXCLUDED_DIRS = {"api", "testing", "_vendor", "operations", "kernels"}

# Filename stems excluded from scanning (benchmarks, profiles, IO boundary,
# bench CLI utilities that operate on JSON dicts not device arrays).
_EXCLUDED_STEMS = {
    "pipeline_benchmarks", "profile_rails", "fixture_profiles",
    "catalog", "cli", "compare", "nvbench_runner", "output", "runner", "schema", "suites",
}

# Filename prefixes excluded from scanning.
_EXCLUDED_PREFIXES = ("benchmark_", "io_", "profile_", "bench_")


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



def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_d2h_call(node: ast.Call) -> bool:
    name = _call_name(node)
    if name not in D2H_APIS:
        return False
    # --- Disambiguate dict.get() from cupy_array.get() ---
    # cupy.ndarray.get() signature: get(stream=None, order='C', out=None)
    #   → 0 positional args in normal usage, or keyword args only.
    # dict.get() signature: get(key[, default])
    #   → always has >=1 positional arg.
    #
    # Rule: if .get() has any positional args, it's dict.get() (not D2H).
    # cupy .get() is only called with 0 positional args (keyword-only).
    if name == "get" and isinstance(node.func, ast.Attribute) and node.args:
        return False  # dict.get(key), dict.get(key, default), etc.
    # --- Disambiguate pyarrow.to_pylist() from cupy.to_pylist() ---
    # to_pylist() on pyarrow arrays is host-to-host, not D2H.
    # Heuristic: if the method is to_pylist and the call chain involves
    # .field(), .column(), or the receiver looks like an Arrow type, skip it.
    if name == "to_pylist" and isinstance(node.func, ast.Attribute):
        # Check if the receiver is a result of .field() or .column() call
        if isinstance(node.func.value, ast.Call):
            inner = node.func.value
            if isinstance(inner.func, ast.Attribute) and inner.func.attr in (
                "field", "column", "to_pandas", "to_pydict",
            ):
                return False
    return True


def _is_h2d_call(node: ast.Call) -> bool:
    name = _call_name(node)
    if name not in H2D_APIS:
        return False
    # Only count as H2D if the receiver is a device module (cp/cupy).
    # np.asarray() and np.array() are host-to-host — NOT H2D transfers.
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        receiver = node.func.value.id
        if receiver in H2D_MODULES:
            return True  # cp.asarray(), cupy.array() — real H2D
        if receiver in ("np", "numpy"):
            return False  # np.asarray() — host-to-host, not H2D
        # Other receivers (e.g. torch.asarray) — flag conservatively
        return True
    # Bare function call (e.g. asarray() without module prefix) — flag it
    if isinstance(node.func, ast.Name) and name in H2D_APIS:
        return True
    return False


# ---- Inline suppression: # zcopy:ok(reason) ----

_SUPPRESSION_RE = re.compile(r"#\s*zcopy:ok\(([^)]*)\)")


def _parse_suppression_comments(source: str) -> dict[int, str]:
    """Scan *source* for ``# zcopy:ok(reason)`` comments.

    Returns ``{lineno: reason}`` with 1-indexed line numbers.
    Lines where the reason is empty are **not** included — they will be
    reported as ``ZCOPY_EMPTY_SUPPRESSION`` errors by the caller.
    """
    suppressions: dict[int, str] = {}
    for lineno, line in enumerate(source.splitlines(), start=1):
        m = _SUPPRESSION_RE.search(line)
        if m is not None:
            reason = m.group(1).strip()
            if reason:  # empty-reason lines are NOT valid suppressions
                suppressions[lineno] = reason
    return suppressions


def _find_empty_suppression_lines(source: str) -> list[int]:
    """Return 1-indexed line numbers with ``# zcopy:ok()`` (empty reason)."""
    empty_lines: list[int] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        m = _SUPPRESSION_RE.search(line)
        if m is not None and not m.group(1).strip():
            empty_lines.append(lineno)
    return empty_lines


# ---- ZCOPY001: Ping-pong transfers in the same function ----

def check_pingpong_transfers(
    repo_root: Path,
) -> tuple[list[LintError], int]:
    """Find functions where data goes D->H then back H->D (or vice versa).

    Returns ``(errors, suppressed_count)``.
    """
    errors: list[LintError] = []
    suppressed = 0
    root = repo_root / "src" / "vibespatial"
    for path in iter_python_files(root):
        source = path.read_text(encoding="utf-8")
        suppressions = _parse_suppression_comments(source)
        tree = ast.parse(source, filename=str(path))
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
                    if first_d2h in suppressions:
                        suppressed += 1
                        continue
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
    return errors, suppressed


# ---- ZCOPY002: Per-element device transfers in loops ----

def check_loop_transfers(
    repo_root: Path,
) -> tuple[list[LintError], int]:
    """Find D->H transfer calls inside for/while loop bodies.

    Returns ``(errors, suppressed_count)``.
    """
    errors: list[LintError] = []
    suppressed = 0
    root = repo_root / "src" / "vibespatial"
    for path in iter_python_files(root):
        source = path.read_text(encoding="utf-8")
        suppressions = _parse_suppression_comments(source)
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.For, ast.While)):
                continue
            for descendant in ast.walk(node):
                if not isinstance(descendant, ast.Call):
                    continue
                if _is_d2h_call(descendant):
                    if descendant.lineno in suppressions:
                        suppressed += 1
                        continue
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
    return errors, suppressed


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


def check_boundary_type_leak(
    repo_root: Path,
) -> tuple[list[LintError], int]:
    """Find functions using device APIs that return D->H converted results.

    Returns ``(errors, suppressed_count)``.
    """
    errors: list[LintError] = []
    suppressed = 0
    root = repo_root / "src" / "vibespatial"
    for path in iter_python_files(root):
        source = path.read_text(encoding="utf-8")
        suppressions = _parse_suppression_comments(source)
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name in _ALLOWED_HOST_RETURN or node.name.startswith("_"):
                continue
            if not _uses_device_apis(node):
                continue
            d2h_line = _returns_host_conversion(node)
            if d2h_line is not None:
                if d2h_line in suppressions:
                    suppressed += 1
                    continue
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
    return errors, suppressed


def _check_empty_suppressions(repo_root: Path) -> list[LintError]:
    """Emit errors for ``# zcopy:ok()`` with empty reason strings."""
    errors: list[LintError] = []
    root = repo_root / "src" / "vibespatial"
    for path in iter_python_files(root):
        source = path.read_text(encoding="utf-8")
        for lineno in _find_empty_suppression_lines(source):
            errors.append(
                LintError(
                    code="ZCOPY_EMPTY_SUPPRESSION",
                    path=path,
                    line=lineno,
                    message=(
                        "Suppression comment # zcopy:ok() has empty reason. "
                        "Provide a justification: # zcopy:ok(reason here)."
                    ),
                    doc_path=RUNTIME_DOC,
                )
            )
    return errors


def run_checks(repo_root: Path) -> tuple[list[LintError], int]:
    """Run all ZCOPY checks.

    Returns ``(errors, suppressed_count)``.
    """
    errors: list[LintError] = []
    total_suppressed = 0

    pingpong_errors, pingpong_suppressed = check_pingpong_transfers(repo_root)
    errors.extend(pingpong_errors)
    total_suppressed += pingpong_suppressed

    loop_errors, loop_suppressed = check_loop_transfers(repo_root)
    errors.extend(loop_errors)
    total_suppressed += loop_suppressed

    leak_errors, leak_suppressed = check_boundary_type_leak(repo_root)
    errors.extend(leak_errors)
    total_suppressed += leak_suppressed

    errors.extend(_check_empty_suppressions(repo_root))

    return (
        sorted(errors, key=lambda e: (str(e.path), e.line, e.code)),
        total_suppressed,
    )


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

    errors, suppressed_count = run_checks(REPO_ROOT)
    count = len(errors)

    suppressed_msg = ""
    if suppressed_count:
        suppressed_msg = f" ({suppressed_count} suppressed via # zcopy:ok)"

    if count > _VIOLATION_BASELINE:
        for error in errors:
            print(error.render(REPO_ROOT))
        print(
            f"\nZero-copy checks FAILED: {count} violations found, "
            f"baseline is {_VIOLATION_BASELINE}. "
            f"New code introduced {count - _VIOLATION_BASELINE} violation(s)."
            f"{suppressed_msg}",
            file=sys.stderr,
        )
        return 1

    if count < _VIOLATION_BASELINE:
        print(
            f"Zero-copy checks passed ({count} known violations, baseline {_VIOLATION_BASELINE}). "
            f"Debt reduced! Update _VIOLATION_BASELINE to {count}.{suppressed_msg}"
        )
    else:
        print(
            f"Zero-copy checks passed ({count} known violations, baseline holds)."
            f"{suppressed_msg}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
