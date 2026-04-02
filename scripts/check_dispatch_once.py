"""Detect mid-cycle GPU->CPU fallback anti-patterns.

This check enforces the dispatch-once rule:
- select GPU vs CPU up front
- do not try GPU work and then silently recover into CPU/Shapely work
  inside the same function body

The checker is intentionally conservative. It only flags:
1. try/except blocks whose try body contains GPU-ish calls and whose except
   body contains CPU/Shapely materialization calls
2. bare ``except Exception: pass`` in ``src/vibespatial/`` (unless baselined)

Allowed exceptions are an explicit closed list below and must only shrink.
Import guards are ignored structurally.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "vibespatial"

# Explicit closed-list baseline. Format:
# (relative path from repo root, line number, code) -> reason
_BASELINE_EXCEPTIONS: dict[tuple[str, int, str], str] = {}


@dataclass(frozen=True)
class Violation:
    code: str
    path: Path
    line: int
    message: str

    def render(self, repo_root: Path) -> str:
        rel_path = self.path.relative_to(repo_root)
        return f"{rel_path}:{self.line}: {self.code} {self.message}"


def iter_python_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def parse_module(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _handler_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _handler_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = [func.attr]
        value = func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
            return ".".join(reversed(parts))
        return func.attr
    return None


def _is_import_guard(node: ast.Try) -> bool:
    if not node.handlers:
        return False
    if not all(isinstance(stmt, (ast.Import, ast.ImportFrom)) for stmt in node.body):
        return False
    allowed = {"ImportError", "ModuleNotFoundError"}
    for handler in node.handlers:
        if handler.type is None:
            return False
        name = _handler_name(handler.type)
        if name is None or name.split(".")[-1] not in allowed:
            return False
    return True


def _expr_contains_to_shapely(node: ast.AST) -> bool:
    return any(
        isinstance(descendant, ast.Call)
        and (_call_name(descendant) == "to_shapely" or (_call_name(descendant) or "").endswith(".to_shapely"))
        for descendant in ast.walk(node)
    )


def _node_contains_gpu_calls(node: ast.AST) -> bool:
    for descendant in ast.walk(node):
        if not isinstance(descendant, ast.Call):
            continue
        name = _call_name(descendant) or ""
        if (
            name.startswith("cp.")
            or name.startswith("cupy.")
            or name.endswith("_gpu")
            or name.endswith("_owned")
        ):
            return True
    return False


def _node_contains_cpu_fallback_calls(node: ast.AST) -> bool:
    for descendant in ast.walk(node):
        if not isinstance(descendant, ast.Call):
            continue
        name = _call_name(descendant) or ""
        if name.startswith("shapely.") or name == "shapely":
            return True
        if name.endswith("_cpu"):
            return True
        if name == "to_shapely" or name.endswith(".to_shapely"):
            return True
        if name == "np.asarray" and any(_expr_contains_to_shapely(arg) for arg in descendant.args):
            return True
    return False


def _is_bare_exception_pass(handler: ast.ExceptHandler) -> bool:
    if len(handler.body) != 1 or not isinstance(handler.body[0], ast.Pass):
        return False
    if handler.type is None:
        return True
    name = _handler_name(handler.type)
    if name is None:
        return False
    return name.split(".")[-1] == "Exception"


def run_checks(repo_root: Path = REPO_ROOT) -> list[Violation]:
    errors: list[Violation] = []
    for path in iter_python_files(repo_root / "src" / "vibespatial"):
        tree = parse_module(path)
        rel_path = path.relative_to(repo_root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try) or _is_import_guard(node):
                continue

            try_has_gpu_calls = any(_node_contains_gpu_calls(stmt) for stmt in node.body)

            for handler in node.handlers:
                baseline_key = (rel_path, handler.lineno, "DO002")
                if try_has_gpu_calls and _is_bare_exception_pass(handler):
                    if baseline_key not in _BASELINE_EXCEPTIONS:
                        errors.append(
                            Violation(
                                code="DO002",
                                path=path,
                                line=handler.lineno,
                                message="bare except Exception: pass hides runtime failures; use an explicit dispatch gate or named baseline",
                            )
                        )

            if not try_has_gpu_calls:
                continue

            for handler in node.handlers:
                baseline_key = (rel_path, handler.lineno, "DO001")
                if baseline_key in _BASELINE_EXCEPTIONS:
                    continue
                if any(_node_contains_cpu_fallback_calls(stmt) for stmt in handler.body):
                    errors.append(
                        Violation(
                            code="DO001",
                            path=path,
                            line=handler.lineno,
                            message="try/except GPU->CPU fallback detected; decide dispatch once before entering the implementation path",
                        )
                    )
    return errors


def main() -> int:
    errors = run_checks()
    if errors:
        for error in errors:
            print(error.render(REPO_ROOT))
        return 1
    print("dispatch-once check clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
