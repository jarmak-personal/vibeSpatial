from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DOC = "docs/architecture/runtime.md"
RESIDENCY_DOC = "docs/architecture/residency.md"
SYNTHETIC_DOC = "docs/testing/synthetic-data.md"
ALLOWED_MATERIALIZATION_METHODS = {"to_pandas", "to_numpy", "values", "__repr__"}
DATA_FILE_ALLOWED_SUFFIXES = {".py", ".md"}
GPU_TEST_REQUIRED_TOKENS = {
    "null": "null-case parametrization",
    "empty": "empty-geometry parametrization",
    "mixed": "mixed-type parametrization",
}
KERNEL_TRANSFER_APIS = {
    "get": "device_array.get()",
    "copy_to_host": "device_array.copy_to_host()",
    "to_host": "device_array.to_host()",
    "to_pylist": "arrow_array.to_pylist()",
    "asnumpy": "cupy.asnumpy()",
    "tolist": "array.tolist()",
}
GEOMETRY_CONTEXT_PARAM_NAMES = {
    "geometry_array",
    "geometries",
    "left",
    "owned",
    "points",
    "polygons",
    "query_owned",
    "right",
    "tree_owned",
}
RESIDENCY_LINT_ALLOWLIST: set[tuple[str, str]] = {
    ("src/vibespatial/constructive/normalize.py", "normalize_owned"),
    ("src/vibespatial/constructive/normalize.py", "_normalize_gpu"),
}


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


def iter_python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def parse_module(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def is_public_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if node.name.startswith("_"):
        return False
    return not any(decorator_name(decorator) == "property" for decorator in node.decorator_list)


def decorator_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return decorator_name(node.func)
    return None


def call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def annotation_text(node: ast.arg) -> str:
    if node.annotation is None:
        return ""
    try:
        return ast.unparse(node.annotation)
    except Exception:  # pragma: no cover - defensive only
        return ""


def function_has_geometry_context(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for arg in (*node.args.args, *node.args.kwonlyargs):
        if arg.arg in GEOMETRY_CONTEXT_PARAM_NAMES:
            return True
        if "GeometryArray" in annotation_text(arg):
            return True
    return False


def call_requests_explicit_non_auto_mode(node: ast.Call) -> bool:
    requested_mode = None
    for keyword in node.keywords:
        if keyword.arg == "requested_mode":
            requested_mode = keyword.value
            break
    if requested_mode is None:
        return False
    if isinstance(requested_mode, ast.Attribute):
        return requested_mode.attr in {"CPU", "GPU"}
    if isinstance(requested_mode, ast.Constant) and isinstance(requested_mode.value, str):
        return requested_mode.value.lower() in {"cpu", "gpu"}
    return False


def collect_text_tokens(tree: ast.AST) -> set[str]:
    tokens: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            tokens.update(part for part in node.id.lower().replace("-", "_").split("_") if part)
        elif isinstance(node, ast.arg):
            tokens.update(part for part in node.arg.lower().replace("-", "_").split("_") if part)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            normalized = node.value.lower().replace("-", "_")
            tokens.update(part for part in normalized.split("_") if part)
    return tokens


def check_vendor_imports(repo_root: Path) -> list[LintError]:
    errors: list[LintError] = []
    root = repo_root / "src" / "vibespatial"
    for path in iter_python_files(root):
        if "_vendor" in path.parts:
            continue
        tree = parse_module(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                if name and (name == "vibespatial._vendor" or name.startswith("vibespatial._vendor.")):
                    errors.append(
                        LintError(
                            code="ARCH001",
                            path=path,
                            line=node.lineno,
                            message=(
                                "Direct imports from vibespatial._vendor are forbidden in src/vibespatial; "
                                "runtime code must not recouple to vendored upstream internals."
                            ),
                            doc_path=RUNTIME_DOC,
                        )
                    )
    return errors


def check_dispatch_registrations(repo_root: Path) -> list[LintError]:
    errors: list[LintError] = []
    root = repo_root / "src" / "geopandas"
    for path in iter_python_files(root):
        if path.name == "__init__.py":
            continue
        tree = parse_module(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name not in {"GeoSeries", "GeoDataFrame"}:
                continue
            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) or not is_public_method(child):
                    continue
                if any(decorator_name(decorator) == "dispatches" for decorator in child.decorator_list):
                    continue
                errors.append(
                    LintError(
                        code="ARCH002",
                        path=path,
                        line=child.lineno,
                        message=(
                            f"{node.name}.{child.name} is a public API method without a @dispatches(...) "
                            "registration."
                        ),
                        doc_path=RUNTIME_DOC,
                    )
                )
    return errors


def check_kernel_registrations(repo_root: Path) -> list[LintError]:
    errors: list[LintError] = []
    root = repo_root / "src" / "vibespatial" / "kernels"
    for path in iter_python_files(root):
        if path.name == "__init__.py":
            continue
        tree = parse_module(path)
        has_kernel_defs = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and not node.name.startswith("_")
            for node in ast.iter_child_nodes(tree)
        )
        has_registration = any(
            isinstance(node, ast.Call) and call_name(node) == "register_kernel_variant"
            for node in ast.walk(tree)
        )
        if has_kernel_defs and not has_registration:
            errors.append(
                LintError(
                    code="ARCH003",
                    path=path,
                    line=1,
                    message=(
                        "Kernel modules must register at least one variant via "
                        "register_kernel_variant(...)."
                    ),
                    doc_path=RUNTIME_DOC,
                )
            )
    return errors


def check_kernel_host_transfers(repo_root: Path) -> list[LintError]:
    errors: list[LintError] = []
    root = repo_root / "src" / "vibespatial" / "kernels"
    for path in iter_python_files(root):
        tree = parse_module(path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name in ALLOWED_MATERIALIZATION_METHODS:
                continue
            for descendant in ast.walk(node):
                if not isinstance(descendant, ast.Call):
                    continue
                name = call_name(descendant)
                if name not in KERNEL_TRANSFER_APIS:
                    continue
                errors.append(
                    LintError(
                        code="ARCH004",
                        path=path,
                        line=descendant.lineno,
                        message=(
                            f"{KERNEL_TRANSFER_APIS[name]} is only allowed in explicit materialization methods "
                            f"({', '.join(sorted(ALLOWED_MATERIALIZATION_METHODS))})."
                        ),
                        doc_path=RUNTIME_DOC,
                    )
                )
    return errors


def is_gpu_kernel_test(path: Path, tree: ast.AST) -> bool:
    if "upstream" in path.parts:
        return False
    if "kernel" not in path.stem and "kernels" not in path.parts:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if decorator_name(decorator) == "gpu":
                    return True
                if isinstance(decorator, ast.Attribute) and decorator.attr == "gpu":
                    return True
                if (
                    isinstance(decorator, ast.Attribute)
                    and decorator.attr == "mark"
                    and isinstance(decorator.value, ast.Name)
                    and decorator.value.id == "pytest"
                ):
                    continue
    source = ast.unparse(tree) if hasattr(ast, "unparse") else ""
    return "pytest.mark.gpu" in source


def check_gpu_kernel_tests(repo_root: Path) -> list[LintError]:
    errors: list[LintError] = []
    root = repo_root / "tests"
    for path in iter_python_files(root):
        if "upstream" in path.parts:
            continue
        tree = parse_module(path)
        if not is_gpu_kernel_test(path, tree):
            continue
        tokens = collect_text_tokens(tree)
        for token, description in GPU_TEST_REQUIRED_TOKENS.items():
            if token in tokens:
                continue
            errors.append(
                LintError(
                    code="ARCH005",
                    path=path,
                    line=1,
                    message=(
                        f"GPU kernel tests must include {description} to cover the runtime contract."
                    ),
                    doc_path=SYNTHETIC_DOC,
                )
            )
    return errors


REORG_SHIM_MARKER = "# REORG_SHIM"


def _collect_shim_modules(repo_root: Path) -> set[str]:
    """Return dotted module names for files that carry the REORG_SHIM marker."""
    shim_modules: set[str] = set()
    src = repo_root / "src" / "vibespatial"
    for path in iter_python_files(src):
        try:
            first_line = path.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
        except OSError:
            continue
        if REORG_SHIM_MARKER not in first_line:
            continue
        relative = path.relative_to(repo_root / "src")
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        shim_modules.add(".".join(parts))
    return shim_modules


def check_shim_imports(repo_root: Path) -> list[LintError]:
    """ARCH007: Flag imports from deprecated reorg-shim modules."""
    errors: list[LintError] = []
    shim_modules = _collect_shim_modules(repo_root)
    if not shim_modules:
        return errors
    scan_dirs = [
        repo_root / "src" / "vibespatial",
        repo_root / "tests",
        repo_root / "scripts",
    ]
    for scan_dir in scan_dirs:
        for path in iter_python_files(scan_dir):
            # Shim files themselves are allowed to import from the new location
            try:
                first_line = path.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
            except OSError:
                continue
            if REORG_SHIM_MARKER in first_line:
                continue
            tree = parse_module(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in shim_modules:
                            errors.append(
                                LintError(
                                    code="ARCH007",
                                    path=path,
                                    line=node.lineno,
                                    message=(
                                        f"Import from deprecated shim module '{alias.name}'; "
                                        "use the canonical path instead."
                                    ),
                                    doc_path=RUNTIME_DOC,
                                )
                            )
                    continue
                else:
                    continue
                if module in shim_modules:
                    errors.append(
                        LintError(
                            code="ARCH007",
                            path=path,
                            line=node.lineno,
                            message=(
                                f"Import from deprecated shim module '{module}'; "
                                "use the canonical path instead."
                            ),
                            doc_path=RUNTIME_DOC,
                        )
                    )
    return errors


def check_test_data_files(repo_root: Path) -> list[LintError]:
    errors: list[LintError] = []
    root = repo_root / "tests"
    if not root.exists():
        return errors
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or "upstream" in path.parts:
            continue
        if path.suffix in DATA_FILE_ALLOWED_SUFFIXES:
            continue
        errors.append(
            LintError(
                code="ARCH006",
                path=path,
                line=1,
                message=(
                    "Repo-local tests may not check in external data files; generate fixtures with "
                    "the synthetic dataset builder instead."
                ),
                doc_path=SYNTHETIC_DOC,
            )
        )
    return errors


def check_dispatch_residency_context(repo_root: Path) -> list[LintError]:
    errors: list[LintError] = []
    root = repo_root / "src" / "vibespatial"
    for path in iter_python_files(root):
        relative = str(path.relative_to(repo_root))
        tree = parse_module(path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not function_has_geometry_context(node):
                continue
            if (relative, node.name) in RESIDENCY_LINT_ALLOWLIST:
                continue
            for descendant in ast.walk(node):
                if not isinstance(descendant, ast.Call):
                    continue
                if call_name(descendant) != "plan_dispatch_selection":
                    continue
                keyword_names = {keyword.arg for keyword in descendant.keywords if keyword.arg is not None}
                if "current_residency" in keyword_names:
                    continue
                if call_requests_explicit_non_auto_mode(descendant):
                    continue
                errors.append(
                    LintError(
                        code="ARCH008",
                        path=path,
                        line=descendant.lineno,
                        message=(
                            "Geometry-carrying AUTO dispatch planning must pass current_residency "
                            "explicitly so device-backed pipelines stay GPU-sticky."
                        ),
                        doc_path=RESIDENCY_DOC,
                    )
                )
    return errors


def run_checks(repo_root: Path) -> list[LintError]:
    errors: list[LintError] = []
    errors.extend(check_vendor_imports(repo_root))
    errors.extend(check_dispatch_registrations(repo_root))
    errors.extend(check_kernel_registrations(repo_root))
    errors.extend(check_kernel_host_transfers(repo_root))
    errors.extend(check_gpu_kernel_tests(repo_root))
    errors.extend(check_test_data_files(repo_root))
    errors.extend(check_dispatch_residency_context(repo_root))
    return sorted(errors, key=lambda error: (str(error.path), error.line, error.code))


def run_advisory_checks(repo_root: Path) -> list[LintError]:
    """Advisory checks that report migration progress but do not block landing."""
    errors: list[LintError] = []
    errors.extend(check_shim_imports(repo_root))
    return sorted(errors, key=lambda error: (str(error.path), error.line, error.code))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check repo-specific architecture constraints that must hold before commit."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan the repository root for all architecture lint rules.",
    )
    args = parser.parse_args(argv)

    if not args.all:
        parser.error("pass --all")

    errors = run_checks(REPO_ROOT)
    advisory = run_advisory_checks(REPO_ROOT)

    for error in advisory:
        print(f"[advisory] {error.render(REPO_ROOT)}")
    if advisory:
        print(f"\n{len(advisory)} advisory issue(s) (shim imports to migrate).")

    if not errors:
        print("Architecture lint checks passed.")
        return 0

    for error in errors:
        print(error.render(REPO_ROOT))
    print(f"Architecture lint checks failed: {len(errors)} issue(s).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
