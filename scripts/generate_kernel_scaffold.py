from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODULE = "vibespatial.kernels.predicates"
DEFAULT_TIER = 4
DEFAULT_GEOM_TYPES = ("point", "polygon")
KERNEL_INVENTORY_PATH = Path("docs/testing/kernel-inventory.md")
VARIANT_MANIFEST_PATH = Path("src/vibespatial/kernels/variant_manifest.json")


@dataclass(frozen=True)
class KernelSpec:
    name: str
    tier: int = DEFAULT_TIER
    geom_types: tuple[str, ...] = DEFAULT_GEOM_TYPES
    module: str = DEFAULT_MODULE

    @property
    def module_parts(self) -> tuple[str, ...]:
        return tuple(self.module.split("."))

    @property
    def relative_module_dir(self) -> Path:
        return Path(*self.module_parts)

    @property
    def source_file(self) -> Path:
        return Path("src") / self.relative_module_dir / f"{self.name}.py"

    @property
    def test_file(self) -> Path:
        return Path("tests") / f"test_{self.name}.py"

    @property
    def benchmark_file(self) -> Path:
        return Path("benchmarks") / f"bench_{self.name}.py"

    @property
    def kernel_function(self) -> str:
        return self.name

    @property
    def title(self) -> str:
        return self.name.replace("_", " ")

    @property
    def kernel_class(self) -> str:
        if "predicates" in self.module_parts:
            return "PREDICATE"
        if "constructive" in self.module_parts or "overlay" in self.module_parts:
            return "CONSTRUCTIVE"
        if "metrics" in self.module_parts:
            return "METRIC"
        return "COARSE"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a repo-standard kernel scaffold.")
    parser.add_argument("name", help="Kernel name, for example point_bounds")
    parser.add_argument("--tier", type=int, default=DEFAULT_TIER)
    parser.add_argument("--geom-types", default=",".join(DEFAULT_GEOM_TYPES))
    parser.add_argument("--module", default=DEFAULT_MODULE)
    parser.add_argument("--check", action="store_true", help="Verify the scaffold is up to date without mutating files.")
    return parser.parse_args(argv)


def build_spec(args: argparse.Namespace) -> KernelSpec:
    geom_types = tuple(part.strip() for part in args.geom_types.split(",") if part.strip())
    if not geom_types:
        raise ValueError("at least one geometry type is required")
    return KernelSpec(name=args.name, tier=args.tier, geom_types=geom_types, module=args.module)


def render_package_init(module_dir: Path, planned_modules: set[str] | None = None) -> str:
    exported = set(planned_modules or set())
    for path in sorted(module_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        exported.add(path.stem)

    if not exported:
        return "__all__: list[str] = []\n"

    names = sorted(exported)
    imports = [f"from .{name} import {name}" for name in names]
    exports = ",\n    ".join(f'"{name}"' for name in names)
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            *imports,
            "",
            "__all__ = [",
            f"    {exports}",
            "]",
            "",
        ]
    )


def render_kernel_source(spec: KernelSpec) -> str:
    primary = spec.geom_types[0]
    secondary = spec.geom_types[1] if len(spec.geom_types) > 1 else spec.geom_types[0]
    return f"""from __future__ import annotations

from vibespatial import ExecutionMode, KernelClass, PrecisionMode, normalize_precision_mode
from vibespatial.kernel_registry import register_kernel_variant


@register_kernel_variant(
    "{spec.name}",
    "cpu",
    kernel_class=KernelClass.{spec.kernel_class},
    geometry_families={spec.geom_types!r},
    execution_modes=(ExecutionMode.CPU,),
    supports_mixed=False,
)
def {spec.kernel_function}(
    left,
    right,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.CPU,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
):
    \"\"\"Stub for the {spec.title} kernel on {primary}/{secondary} inputs.\"\"\"
    del dispatch_mode
    normalize_precision_mode(precision)
    raise NotImplementedError("{spec.kernel_function} kernel scaffold is not implemented yet")
"""


def render_test_file(spec: KernelSpec) -> str:
    function = spec.kernel_function
    geom_tokens = ", ".join(spec.geom_types)
    return f"""from __future__ import annotations

import pytest
from shapely.geometry import Point, Polygon, box

from vibespatial import ExecutionMode
from vibespatial.testing import compare_with_shapely
from {spec.module}.{function} import {function}


def {function}_reference(points, polygons):
    results = []
    for point, polygon in zip(points, polygons, strict=True):
        if point is None or polygon is None:
            results.append(None)
            continue
        if point.is_empty or polygon.is_empty:
            results.append((float("nan"),) * 4)
            continue
        minx, miny, maxx, maxy = polygon.bounds
        results.append((minx, miny, maxx, maxy))
    return results


@compare_with_shapely(reference={function}_reference, handle_empty=True)
@pytest.mark.parametrize(
    "null_case,empty_case,mixed_case",
    [
        (True, True, "{geom_tokens}"),
    ],
)
def test_{function}(dispatch_mode, oracle_runner, null_case, empty_case, mixed_case) -> None:
    del null_case, empty_case, mixed_case
    points = [None, Point(), Point(1, 1), Point(8, 2)]
    polygons = [box(0, 0, 2, 2), Polygon(), None, box(5, 1, 9, 3)]

    try:
        oracle_runner(
            {function},
            points,
            polygons,
            dispatch_mode=dispatch_mode,
        )
    except NotImplementedError:
        pytest.xfail("{function} kernel scaffold is still a placeholder")

def test_{function}_reports_scaffold_placeholder() -> None:
    with pytest.raises(NotImplementedError):
        {function}([], [], dispatch_mode=ExecutionMode.CPU)
"""


def render_benchmark_file(spec: KernelSpec) -> str:
    return f"""from __future__ import annotations

import time

from vibespatial import ExecutionMode
from vibespatial.testing import SyntheticSpec, generate_points, generate_polygons
from {spec.module}.{spec.kernel_function} import {spec.kernel_function}


TIER = {spec.tier}
REFERENCE_SCALE = "100K"


def run_benchmark() -> dict[str, object]:
    points = list(
        generate_points(
            SyntheticSpec(geometry_type="point", distribution="uniform", count=REFERENCE_SCALE, seed=0)
        ).geometries
    )
    polygons = list(
        generate_polygons(
            SyntheticSpec(geometry_type="polygon", distribution="regular-grid", count="10K", seed=0)
        ).geometries
    )
    tiled_polygons = [polygons[index % len(polygons)] for index in range(len(points))]

    started = time.perf_counter()
    try:
        {spec.kernel_function}(points, tiled_polygons, dispatch_mode=ExecutionMode.CPU)
    except NotImplementedError:
        elapsed = None
    else:
        elapsed = time.perf_counter() - started

    return {{
        "kernel": "{spec.name}",
        "tier": TIER,
        "scale": REFERENCE_SCALE,
        "requested_runtime": ExecutionMode.CPU.value,
        "selected_runtime": ExecutionMode.CPU.value,
        "elapsed_seconds": elapsed,
    }}


if __name__ == "__main__":
    print(run_benchmark())
"""


def load_manifest(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def render_manifest(spec: KernelSpec, existing: list[dict[str, object]]) -> str:
    entries = [entry for entry in existing if entry["kernel"] != spec.name]
    entries.append(
        {
            "kernel": spec.name,
            "module": spec.module,
            "tier": spec.tier,
            "geom_types": list(spec.geom_types),
            "variants": ["cpu"],
        }
    )
    entries.sort(key=lambda entry: str(entry["kernel"]))
    return json.dumps(entries, indent=2, sort_keys=True) + "\n"


def render_inventory(existing: str, spec: KernelSpec) -> str:
    marker = "<!-- KERNEL_INVENTORY:ROWS -->"
    row = (
        f"| `{spec.name}` | `{spec.module}` | Tier {spec.tier} | "
        f"`{', '.join(spec.geom_types)}` | "
        f"`{spec.source_file}` | "
        f"`{spec.test_file}` | "
        f"`{spec.benchmark_file}` |"
    )

    if not existing:
        header = "\n".join(
            [
                "# Kernel Inventory",
                "",
                "Generated kernel scaffolds land here first so agents can audit what exists and which tier gate applies.",
                "",
                "## Intent",
                "",
                "Track scaffolded kernel modules, tests, and benchmark stubs.",
                "",
                "## Request Signals",
                "",
                "- kernel scaffold",
                "- benchmark stub",
                "- kernel inventory",
                "",
                "## Open First",
                "",
                "- docs/testing/kernel-inventory.md",
                "- scripts/generate_kernel_scaffold.py",
                "- docs/testing/performance-tiers.md",
                "",
                "## Verify",
                "",
                "- `uv run python scripts/generate_kernel_scaffold.py --check point_bounds`",
                "- `uv run python scripts/check_docs.py --check`",
                "",
                "## Risks",
                "",
                "- Scaffold drift can leave tests, benchmarks, and manifests out of sync.",
                "- Tier metadata becomes meaningless if generated benchmarks do not match policy.",
                "",
                "## Scaffolds",
                "",
                "| Kernel | Module | Tier | Geometry Types | Source | Test | Benchmark |",
                "|---|---|---|---|---|---|---|",
                marker,
                "",
            ]
        )
        existing = header

    lines = existing.splitlines()
    try:
        index = lines.index(marker)
    except ValueError as exc:
        raise ValueError(f"missing inventory marker in {KERNEL_INVENTORY_PATH}") from exc

    rows = [line for line in lines[index + 1 :] if line.startswith("| `")]
    rows = [line for line in rows if not line.startswith(f"| `{spec.name}` |")]
    rows.append(row)
    rows.sort()
    new_lines = lines[: index + 1] + rows
    return "\n".join(new_lines) + "\n"


def expected_files(repo_root: Path, spec: KernelSpec) -> dict[Path, str]:
    source_dir = repo_root / "src" / spec.relative_module_dir
    manifest_path = repo_root / VARIANT_MANIFEST_PATH
    inventory_path = repo_root / KERNEL_INVENTORY_PATH
    existing_inventory = inventory_path.read_text(encoding="utf-8") if inventory_path.exists() else ""
    existing_manifest = load_manifest(manifest_path)
    files = {
        repo_root / spec.source_file: render_kernel_source(spec),
        repo_root / spec.test_file: render_test_file(spec),
        repo_root / spec.benchmark_file: render_benchmark_file(spec),
        source_dir / "__init__.py": "",
        source_dir.parent / "__init__.py": "",
        repo_root / VARIANT_MANIFEST_PATH: render_manifest(spec, existing_manifest),
        repo_root / KERNEL_INVENTORY_PATH: render_inventory(existing_inventory, spec),
    }

    # package __init__ files depend on sibling modules, so render after file map exists
    planned_by_dir: dict[Path, set[str]] = {
        source_dir: {spec.name},
        source_dir.parent: set(),
    }
    for init_path in (source_dir.parent / "__init__.py", source_dir / "__init__.py"):
        module_dir = init_path.parent
        files[init_path] = render_package_init(module_dir, planned_by_dir.get(module_dir))
    return files


def write_files(expected: dict[Path, str]) -> list[Path]:
    updated: list[Path] = []
    for path, content in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == content:
            continue
        path.write_text(content, encoding="utf-8")
        updated.append(path)
    return updated


def check_files(expected: dict[Path, str], repo_root: Path) -> list[str]:
    mismatches: list[str] = []
    for path, content in expected.items():
        relative = path.relative_to(repo_root)
        if not path.exists():
            mismatches.append(f"{relative} is missing")
            continue
        current = path.read_text(encoding="utf-8")
        if current != content:
            mismatches.append(f"{relative} is out of date")
    return mismatches


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    spec = build_spec(args)
    expected = expected_files(repo_root, spec)

    if args.check:
        mismatches = check_files(expected, repo_root)
        if mismatches:
            for mismatch in mismatches:
                print(mismatch)
            return 1
        print(f"Kernel scaffold for {spec.name} is up to date.")
        return 0

    updated = write_files(expected)
    if updated:
        print("Updated kernel scaffold files:")
        for path in updated:
            print(f"- {path.relative_to(repo_root)}")
    else:
        print(f"Kernel scaffold for {spec.name} already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
