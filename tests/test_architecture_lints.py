from __future__ import annotations

from pathlib import Path

from scripts.check_architecture_lints import REPO_ROOT, run_checks


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_architecture_lints_pass_for_current_repo() -> None:
    assert run_checks(REPO_ROOT) == []


def test_vendor_import_rule_catches_vendored_runtime_dependency(tmp_path: Path) -> None:
    write_file(
        tmp_path / "src" / "vibespatial" / "ops.py",
        "from vibespatial._vendor.geopandas import GeoSeries\n",
    )

    errors = run_checks(tmp_path)
    assert [error.code for error in errors] == ["ARCH001"]


def test_dispatch_rule_requires_public_method_registration(tmp_path: Path) -> None:
    write_file(
        tmp_path / "src" / "geopandas" / "geoseries.py",
        """
class GeoSeries:
    def area(self):
        return 1
""".strip()
        + "\n",
    )

    errors = run_checks(tmp_path)
    assert [error.code for error in errors] == ["ARCH002"]


def test_kernel_modules_must_register_variants(tmp_path: Path) -> None:
    write_file(
        tmp_path / "src" / "vibespatial" / "kernels" / "distance.py",
        """
def point_distance():
    return 1
""".strip()
        + "\n",
    )

    errors = run_checks(tmp_path)
    assert [error.code for error in errors] == ["ARCH003"]


def test_kernel_host_transfer_rule_blocks_non_materialization_methods(tmp_path: Path) -> None:
    write_file(
        tmp_path / "src" / "vibespatial" / "kernels" / "distance.py",
        """
from vibespatial.runtime.kernel_registry import register_kernel_variant


@register_kernel_variant("point_distance", "cpu")
def point_distance(device_array):
    return device_array.get()
""".strip()
        + "\n",
    )

    errors = run_checks(tmp_path)
    assert [error.code for error in errors] == ["ARCH004"]


def test_materialization_methods_may_transfer_to_host(tmp_path: Path) -> None:
    write_file(
        tmp_path / "src" / "vibespatial" / "kernels" / "distance.py",
        """
from vibespatial.runtime.kernel_registry import register_kernel_variant


@register_kernel_variant("point_distance", "cpu")
def to_numpy(device_array):
    return device_array.get()
""".strip()
        + "\n",
    )

    assert run_checks(tmp_path) == []


def test_gpu_kernel_tests_require_null_empty_and_mixed_coverage(tmp_path: Path) -> None:
    write_file(
        tmp_path / "tests" / "kernels" / "test_point_kernel.py",
        """
import pytest


@pytest.mark.gpu
def test_point_kernel():
    assert True
""".strip()
        + "\n",
    )

    errors = run_checks(tmp_path)
    assert [error.code for error in errors] == ["ARCH005", "ARCH005", "ARCH005"]


def test_gpu_kernel_test_rule_accepts_null_empty_and_mixed_tokens(tmp_path: Path) -> None:
    write_file(
        tmp_path / "tests" / "kernels" / "test_point_kernel.py",
        """
import pytest


@pytest.mark.gpu
@pytest.mark.parametrize("null_case,empty_case,mixed_case", [(None, [], "mixed")])
def test_point_kernel(null_case, empty_case, mixed_case):
    assert (null_case, empty_case, mixed_case) is not None
""".strip()
        + "\n",
    )

    assert run_checks(tmp_path) == []


def test_local_test_data_files_are_rejected(tmp_path: Path) -> None:
    write_file(tmp_path / "tests" / "fixtures" / "sample.geojson", "{}\n")

    errors = run_checks(tmp_path)
    assert [error.code for error in errors] == ["ARCH006"]


def test_upstream_test_data_files_are_allowed(tmp_path: Path) -> None:
    write_file(tmp_path / "tests" / "upstream" / "geopandas" / "tests" / "data" / "sample.geojson", "{}\n")

    assert run_checks(tmp_path) == []


def test_auto_geometry_dispatch_requires_explicit_residency(tmp_path: Path) -> None:
    write_file(
        tmp_path / "src" / "vibespatial" / "ops.py",
        """
from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.precision import KernelClass


def area_owned(owned: OwnedGeometryArray, dispatch_mode):
    return plan_dispatch_selection(
        kernel_name="geometry_area",
        kernel_class=KernelClass.METRIC,
        row_count=owned.row_count,
        requested_mode=dispatch_mode,
    )
""".strip()
        + "\n",
    )

    errors = run_checks(tmp_path)
    assert [error.code for error in errors] == ["ARCH008"]


def test_auto_geometry_dispatch_with_explicit_residency_passes(tmp_path: Path) -> None:
    write_file(
        tmp_path / "src" / "vibespatial" / "ops.py",
        """
from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.precision import KernelClass


def area_owned(owned: OwnedGeometryArray, dispatch_mode):
    return plan_dispatch_selection(
        kernel_name="geometry_area",
        kernel_class=KernelClass.METRIC,
        row_count=owned.row_count,
        requested_mode=dispatch_mode,
        current_residency=owned.residency,
    )
""".strip()
        + "\n",
    )

    assert run_checks(tmp_path) == []
