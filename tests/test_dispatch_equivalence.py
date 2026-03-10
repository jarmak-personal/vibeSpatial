from __future__ import annotations

import pytest

from vibespatial import (
    ExecutionMode,
    KernelClass,
    invalidate_snapshot_cache,
    plan_dispatch_selection,
)


@pytest.fixture(autouse=True)
def _clean_snapshot_cache():
    invalidate_snapshot_cache()
    yield
    invalidate_snapshot_cache()


@pytest.mark.parametrize(
    "kernel_name,kernel_class,threshold",
    [
        ("point_clip", KernelClass.CONSTRUCTIVE, 10_000),
        ("point_buffer", KernelClass.CONSTRUCTIVE, 10_000),
        ("linestring_buffer", KernelClass.CONSTRUCTIVE, 5_000),
        ("polygon_buffer", KernelClass.CONSTRUCTIVE, 50_000),
        ("segment_classify", KernelClass.CONSTRUCTIVE, 4_096),
        ("flat_index_build", KernelClass.COARSE, 0),
    ],
)
def test_threshold_equivalence(
    kernel_name: str,
    kernel_class: KernelClass,
    threshold: int,
) -> None:
    for row_count in [0, max(0, threshold - 1), threshold, threshold + 1, threshold * 2]:
        selection = plan_dispatch_selection(
            kernel_name=kernel_name,
            kernel_class=kernel_class,
            row_count=row_count,
            gpu_available=True,
        )
        expected_gpu = row_count >= threshold
        assert (selection.selected is ExecutionMode.GPU) == expected_gpu, (
            f"{kernel_name} at {row_count} rows: expected "
            f"{'GPU' if expected_gpu else 'CPU'}, got {selection.selected}"
        )


def test_explicit_gpu_pin_respected() -> None:
    selection = plan_dispatch_selection(
        kernel_name="point_clip",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=1,
        requested_mode=ExecutionMode.GPU,
        gpu_available=True,
    )
    assert selection.selected is ExecutionMode.GPU


def test_explicit_cpu_pin_respected() -> None:
    selection = plan_dispatch_selection(
        kernel_name="flat_index_build",
        kernel_class=KernelClass.COARSE,
        row_count=1_000_000,
        requested_mode=ExecutionMode.CPU,
        gpu_available=True,
    )
    assert selection.selected is ExecutionMode.CPU


def test_gpu_unavailable_falls_back_to_cpu() -> None:
    selection = plan_dispatch_selection(
        kernel_name="linestring_buffer",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=100_000,
        gpu_available=False,
    )
    assert selection.selected is ExecutionMode.CPU
