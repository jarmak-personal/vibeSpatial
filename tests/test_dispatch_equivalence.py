from __future__ import annotations

import pytest

from vibespatial import (
    DEFAULT_DATACENTER_PROFILE,
    DeviceSnapshot,
    ExecutionMode,
    KernelClass,
    MonitoringBackend,
    PrecisionMode,
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
        ("normalize", KernelClass.COARSE, 500),
        ("point_clip", KernelClass.CONSTRUCTIVE, 10_000),
        ("point_buffer", KernelClass.CONSTRUCTIVE, 0),
        ("linestring_buffer", KernelClass.CONSTRUCTIVE, 5_000),
        ("polygon_centroid", KernelClass.METRIC, 500),
        ("polygon_buffer", KernelClass.CONSTRUCTIVE, 50_000),
        ("segment_classify", KernelClass.CONSTRUCTIVE, 4_096),
        ("bbox_overlap_candidates", KernelClass.COARSE, 0),
        ("flat_index_build", KernelClass.COARSE, 0),
        ("intersects", KernelClass.PREDICATE, 0),
        ("contains", KernelClass.PREDICATE, 0),
        ("within", KernelClass.PREDICATE, 0),
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


def test_precision_kernel_override_preserves_cached_device_profile(monkeypatch) -> None:
    import vibespatial.runtime.adaptive as adaptive

    monkeypatch.setattr(
        adaptive,
        "get_cached_snapshot",
        lambda: DeviceSnapshot(
            backend=MonitoringBackend.UNAVAILABLE,
            gpu_available=True,
            device_profile=DEFAULT_DATACENTER_PROFILE,
            reason="test snapshot",
        ),
    )

    selection = plan_dispatch_selection(
        kernel_name="bbox_overlap_candidates",
        kernel_class=KernelClass.COARSE,
        precision_kernel_class=KernelClass.PREDICATE,
        row_count=10_000,
        requested_precision=PrecisionMode.AUTO,
    )

    assert selection.selected is ExecutionMode.GPU
    assert selection.device_profile == DEFAULT_DATACENTER_PROFILE
    assert selection.precision_plan.kernel_class is KernelClass.PREDICATE
    assert selection.precision_plan.compute_precision is PrecisionMode.FP64
