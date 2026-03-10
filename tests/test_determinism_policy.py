from __future__ import annotations

from scripts.check_determinism import probe_determinism
from vibespatial import (
    DETERMINISM_ENV_VAR,
    DeterminismMode,
    KernelClass,
    ReproducibilityGuarantee,
    determinism_mode_from_env,
    deterministic_mode_enabled,
    normalize_determinism_mode,
    select_determinism_plan,
)


def test_env_mode_defaults_to_default(monkeypatch) -> None:
    monkeypatch.delenv(DETERMINISM_ENV_VAR, raising=False)

    assert determinism_mode_from_env() is DeterminismMode.DEFAULT
    assert deterministic_mode_enabled() is False
    assert normalize_determinism_mode("deterministic") is DeterminismMode.DETERMINISTIC


def test_env_mode_can_force_deterministic(monkeypatch) -> None:
    monkeypatch.setenv(DETERMINISM_ENV_VAR, "deterministic")

    assert determinism_mode_from_env() is DeterminismMode.DETERMINISTIC
    assert deterministic_mode_enabled() is True


def test_default_metric_plan_allows_fast_reductions() -> None:
    plan = select_determinism_plan(kernel_class=KernelClass.METRIC, requested=DeterminismMode.DEFAULT)

    assert plan.guarantee is ReproducibilityGuarantee.NONE
    assert plan.fixed_reduction_order is False
    assert plan.floating_atomics_allowed is True


def test_deterministic_constructive_plan_requires_fixed_order() -> None:
    plan = select_determinism_plan(kernel_class=KernelClass.CONSTRUCTIVE, requested=DeterminismMode.DETERMINISTIC)

    assert plan.guarantee is ReproducibilityGuarantee.SAME_DEVICE_BITWISE
    assert plan.stable_output_order is True
    assert plan.fixed_reduction_order is True
    assert plan.fixed_scan_order is True
    assert plan.floating_atomics_allowed is False
    assert plan.same_device_only is True
    assert plan.expected_max_overhead_factor == 2.0


def test_dissolve_area_probe_is_bitwise_identical_on_current_cpu_path() -> None:
    result = probe_determinism(rows=64, groups=8, repeats=5)

    assert result.bitwise_identical is True
    assert result.unique_fingerprints == 1
    assert result.overhead_factor > 0.0
