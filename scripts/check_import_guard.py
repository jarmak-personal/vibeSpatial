"""Import guard lints (IGRD001-002).

Hard-blocks numpy and shapely imports in GPU-path source modules to prevent
agents and contributors from introducing CPU shortcuts into GPU pipelines.

Files in the allowlist may import numpy/shapely for legitimate boundary
reasons (dtype constants, WKB serialization, type checking).  All other
files in src/vibespatial/ are BLOCKED.

Directories entirely exempt from scanning:
  - api/        (GeoPandas-compat API layer — shapely is the interface)
  - testing/    (test infrastructure needs both for oracle comparison)
  - bench/      (benchmarks compare CPU vs GPU — both needed)
  - _vendor/    (third-party code, not ours to modify)

Uses a ratchet baseline for pre-existing violations in GPU-path code.
New violations are a hard failure.

Run:
    uv run python scripts/check_import_guard.py --all
"""
from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Directories entirely exempt — these are boundary/API layers where
# numpy and shapely are the interface contract, not a shortcut.
# ---------------------------------------------------------------------------
_EXEMPT_DIRS = {"api", "testing", "bench", "_vendor"}

# ---------------------------------------------------------------------------
# Per-file allowlist: files in GPU-path directories that LEGITIMATELY
# need shapely or numpy beyond dtype constants.
#
# Format:  relative-to-src/vibespatial  →  set of allowed modules
#
# Every entry must have a one-line justification.
# ---------------------------------------------------------------------------
_SHAPELY_ALLOWLIST: dict[str, str] = {
    # geometry/ — host↔device serialization boundary
    "geometry/device_array.py": "Pandas ExtensionArray: shapely for materialization, type checks",
    "geometry/owned.py": "WKB serialization, from_shapely_geometries construction",
    "geometry/equality.py": "Geometry comparison via shapely normalization",
    # overlay/ — tracked fallback paths (shrink over time)
    "overlay/dissolve.py": "GeometryCollection sentinel + fallback aggregation",
    "overlay/gpu.py": "Fallback host path for unsupported overlay cases",
    "overlay/bypass.py": "Empty-geometry sentinel for containment bypass empty result",
    "overlay/host_fallback.py": "Host fallback path for overlay face-to-polygon assembly",
    # spatial/ — vectorized shapely C ops + fallback paths
    "spatial/query_utils.py": "Vectorized shapely.get_type_id/get_coordinates (C-accelerated)",
    "spatial/query_box.py": "BaseGeometry type check for input validation",
    "spatial/query_cpu.py": "CPU-only Shapely bounds extraction helpers for spatial query fast paths",
    "spatial/distance_owned.py": "Fallback distance computation",
    "spatial/nearest.py": "Fallback nearest-neighbor computation",
    # predicates/ — predicate definitions reference shapely
    "predicates/binary.py": "Shapely predicate names and fallback evaluation",
    "predicates/relate.py": "DE-9IM mask computation via shapely fallback",
    "predicates/point_in_polygon_cpu.py": "CPU-only point-in-polygon variant registration module",
    # constructive/ — CPU-only variant registration modules and host-side constructive boundaries
    "constructive/polygon_difference_cpu.py": "CPU-only polygon difference variant registration module",
    "constructive/polygon_intersection_cpu.py": "CPU-only polygon intersection variant registration module",
    "constructive/segmented_union_cpu.py": "CPU-only segmented union variant registration module",
    "constructive/binary_constructive_cpu.py": "CPU-only binary constructive variant registration module",
    "constructive/boundary_cpu.py": "CPU-only boundary variant registration module",
    "constructive/exterior_cpu.py": "CPU-only exterior-ring variant registration module",
    "constructive/interiors_cpu.py": "CPU-only interior-rings variant registration module",
    "constructive/extract_unique_points_cpu.py": "CPU-only extract-unique-points variant registration module",
    "constructive/normalize_cpu.py": "CPU-only normalize variant registration module",
    "constructive/linestring_buffer_cpu.py": "CPU-only linestring buffer helper",
    "constructive/line_merge_cpu.py": "CPU-only line-merge variant registration module",
    "constructive/remove_repeated_points_cpu.py": "CPU-only remove-repeated-points variant registration module",
    "constructive/minimum_rotated_rectangle_cpu.py": "CPU-only minimum-rotated-rectangle helper and variant",
    "constructive/minimum_clearance_cpu.py": "CPU-only minimum-clearance helper",
    "constructive/polygon_buffer_cpu.py": "CPU-only polygon buffer helper",
    "constructive/set_precision_cpu.py": "CPU-only set-precision variant registration module",
    "constructive/simplify_cpu.py": "CPU-only simplify variant registration module",
    "constructive/snap_cpu.py": "CPU-only snap variant registration module",
    "constructive/shared_paths_cpu.py": "CPU-only shared-paths assembly and variant registration module",
    "constructive/shortest_line_cpu.py": "CPU-only shortest-line variant registration module",
    "constructive/stroke_cpu.py": "CPU-only stroke buffer and offset-curve helper module",
    "constructive/union_all_cpu.py": "CPU-only union-all reduction helpers and empty-result sentinels",
    "constructive/make_valid_pipeline_cpu.py": "CPU-only make-valid helpers and warmup geometry builder",
    "constructive/properties_cpu.py": "CPU-only get-geometry variant registration module",
    "constructive/clip_rect_cpu.py": "CPU-only clip-by-rect host assembly and baseline module",
    "constructive/multipoint_polygon_constructive.py": "Host-side multipoint-polygon constructive boundary",
    # io/ boundary — format-specific needs
    "io/wkb_cpu.py": "CPU-only WKB multipart extraction helper",
    "io/postgis_gpu.py": "WKB decode for PostGIS binary format",
    "io/geoarrow.py": "GeoArrow metadata requires shapely geometry type IDs",
    "spatial/indexing_cpu.py": "CPU-only Shapely multipart extraction helper for segment MBR fallback",
    # runtime/ — provenance tracking
    "runtime/provenance.py": "Geometry type pattern matching for rewrite rules",
}

_NUMPY_ALLOWLIST: dict[str, str] = {
    # cuda/ — dtype constants and memory protocol
    "cuda/_runtime.py": "RMM allocator setup, dtype for memory protocol",
    "cuda/cccl_primitives.py": "dtype definitions for CCCL primitive wrappers",
    # geometry/ — host↔device boundary
    "geometry/device_array.py": "Pandas ExtensionArray: dtype, array protocol",
    "geometry/owned.py": "Coordinate arrays, struct packing, cumsum for offsets",
    # runtime/
    "runtime/nulls.py": "np.isnan for null detection",
    "runtime/provenance.py": "Operation tagging arrays",
}

# Pre-existing violations as of 2026-03-29.
# Decrease as debt is paid.  Fails only if current count EXCEEDS baseline.
# These are shapely/numpy imports in GPU-path code that predate the guard.
# Every file here is tracked debt — the baseline MUST only go down.
_SHAPELY_VIOLATION_BASELINE = 0
_NUMPY_VIOLATION_BASELINE = 0


@dataclass(frozen=True)
class LintError:
    code: str
    path: Path
    line: int
    message: str

    def render(self, repo_root: Path) -> str:
        relative = self.path.relative_to(repo_root)
        return f"{relative}:{self.line}: {self.code} {self.message}"


def _is_exempt(path: Path, src_root: Path) -> bool:
    """Check if a file is in an entirely exempt directory."""
    rel = path.relative_to(src_root)
    parts = rel.parts
    if parts and parts[0] in _EXEMPT_DIRS:
        return True
    return False


def _relative_key(path: Path, src_root: Path) -> str:
    """Get the allowlist lookup key for a file."""
    return str(path.relative_to(src_root))


def _is_in_allowlist(path: Path, src_root: Path, allowlist: dict[str, str]) -> bool:
    """Check if a file is in the per-file allowlist."""
    key = _relative_key(path, src_root)
    return key in allowlist


def iter_python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*.py")
        if "__pycache__" not in p.parts
    )


def _has_shapely_import(tree: ast.AST) -> list[int]:
    """Return line numbers of shapely imports (excluding docstring examples)."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "shapely" or alias.name.startswith("shapely."):
                    lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "shapely" or node.module.startswith("shapely."):
                lines.append(node.lineno)
    return lines


def _has_numpy_import(tree: ast.AST) -> list[int]:
    """Return line numbers of numpy imports."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy" or alias.name.startswith("numpy."):
                    lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "numpy" or node.module.startswith("numpy."):
                lines.append(node.lineno)
    return lines


def check_shapely_imports(repo_root: Path) -> list[LintError]:
    """IGRD001: Block shapely imports outside the allowlist."""
    errors: list[LintError] = []
    src_root = repo_root / "src" / "vibespatial"

    for path in iter_python_files(src_root):
        if _is_exempt(path, src_root):
            continue
        if _is_in_allowlist(path, src_root, _SHAPELY_ALLOWLIST):
            continue

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue

        for line in _has_shapely_import(tree):
            key = _relative_key(path, src_root)
            errors.append(LintError(
                code="IGRD001",
                path=path,
                line=line,
                message=(
                    f"shapely import in GPU-path module '{key}'. "
                    f"shapely is blocked here — use GPU primitives instead. "
                    f"If this is a genuine boundary need, add to _SHAPELY_ALLOWLIST "
                    f"in scripts/check_import_guard.py with justification."
                ),
            ))
    return errors


def check_numpy_imports(repo_root: Path) -> list[LintError]:
    """IGRD002: Block numpy imports in GPU-pure modules."""
    errors: list[LintError] = []
    src_root = repo_root / "src" / "vibespatial"

    # GPU-pure directories where numpy should not appear
    # (unless in the allowlist).
    gpu_pure_dirs = {"cuda", "kernels", "runtime"}

    for path in iter_python_files(src_root):
        rel = path.relative_to(src_root)
        parts = rel.parts
        if not parts or parts[0] not in gpu_pure_dirs:
            continue
        if _is_in_allowlist(path, src_root, _NUMPY_ALLOWLIST):
            continue

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue

        for line in _has_numpy_import(tree):
            key = _relative_key(path, src_root)
            errors.append(LintError(
                code="IGRD002",
                path=path,
                line=line,
                message=(
                    f"numpy import in GPU-pure module '{key}'. "
                    f"numpy is blocked in {parts[0]}/ — use cupy for device data, "
                    f"or add to _NUMPY_ALLOWLIST with justification."
                ),
            ))
    return errors


def run_checks(repo_root: Path) -> tuple[list[LintError], list[LintError]]:
    shapely_errors = check_shapely_imports(repo_root)
    numpy_errors = check_numpy_imports(repo_root)
    shapely_errors.sort(key=lambda e: (str(e.path), e.line))
    numpy_errors.sort(key=lambda e: (str(e.path), e.line))
    return shapely_errors, numpy_errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check import guards for numpy/shapely in GPU-path modules."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Scan src/vibespatial/ for import guard violations.",
    )
    args = parser.parse_args(argv)

    if not args.all:
        parser.error("pass --all")

    shapely_errors, numpy_errors = run_checks(REPO_ROOT)
    shapely_count = len(shapely_errors)
    numpy_count = len(numpy_errors)

    failed = False

    if shapely_count > _SHAPELY_VIOLATION_BASELINE:
        for error in shapely_errors:
            print(error.render(REPO_ROOT))
        new = shapely_count - _SHAPELY_VIOLATION_BASELINE
        print(
            f"\nIGRD001 FAILED: {shapely_count} shapely import violations found, "
            f"baseline is {_SHAPELY_VIOLATION_BASELINE}. "
            f"New code introduced {new} violation(s).",
            file=sys.stderr,
        )
        failed = True
    elif shapely_count < _SHAPELY_VIOLATION_BASELINE:
        print(
            f"IGRD001 passed ({shapely_count} known violations, baseline {_SHAPELY_VIOLATION_BASELINE}). "
            f"Debt reduced! Update _SHAPELY_VIOLATION_BASELINE to {shapely_count}."
        )
    else:
        print(f"IGRD001 passed ({shapely_count} violations, baseline holds).")

    if numpy_count > _NUMPY_VIOLATION_BASELINE:
        for error in numpy_errors:
            print(error.render(REPO_ROOT))
        new = numpy_count - _NUMPY_VIOLATION_BASELINE
        print(
            f"\nIGRD002 FAILED: {numpy_count} numpy import violations found, "
            f"baseline is {_NUMPY_VIOLATION_BASELINE}. "
            f"New code introduced {new} violation(s).",
            file=sys.stderr,
        )
        failed = True
    elif numpy_count < _NUMPY_VIOLATION_BASELINE:
        print(
            f"IGRD002 passed ({numpy_count} known violations, baseline {_NUMPY_VIOLATION_BASELINE}). "
            f"Debt reduced! Update _NUMPY_VIOLATION_BASELINE to {numpy_count}."
        )
    else:
        print(f"IGRD002 passed ({numpy_count} violations, baseline holds).")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
