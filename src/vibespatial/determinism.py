from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import os

from vibespatial.precision import KernelClass


DETERMINISM_ENV_VAR = "VIBESPATIAL_DETERMINISM"


class DeterminismMode(StrEnum):
    DEFAULT = "default"
    DETERMINISTIC = "deterministic"


class ReproducibilityGuarantee(StrEnum):
    NONE = "none"
    SAME_DEVICE_BITWISE = "same-device-bitwise"


@dataclass(frozen=True)
class DeterminismPlan:
    kernel_class: KernelClass
    mode: DeterminismMode
    guarantee: ReproducibilityGuarantee
    stable_output_order: bool
    fixed_reduction_order: bool
    fixed_scan_order: bool
    floating_atomics_allowed: bool
    same_device_only: bool
    expected_max_overhead_factor: float
    reason: str


def normalize_determinism_mode(value: DeterminismMode | str | None) -> DeterminismMode:
    if value is None:
        return DeterminismMode.DEFAULT
    return value if isinstance(value, DeterminismMode) else DeterminismMode(value)


def determinism_mode_from_env() -> DeterminismMode:
    return normalize_determinism_mode(os.environ.get(DETERMINISM_ENV_VAR, DeterminismMode.DEFAULT.value))


def deterministic_mode_enabled(requested: DeterminismMode | str | None = None) -> bool:
    if requested is not None:
        return normalize_determinism_mode(requested) is DeterminismMode.DETERMINISTIC
    return determinism_mode_from_env() is DeterminismMode.DETERMINISTIC


def select_determinism_plan(
    *,
    kernel_class: KernelClass,
    requested: DeterminismMode | str | None = None,
) -> DeterminismPlan:
    mode = determinism_mode_from_env() if requested is None else normalize_determinism_mode(requested)

    if mode is DeterminismMode.DEFAULT:
        return DeterminismPlan(
            kernel_class=kernel_class,
            mode=mode,
            guarantee=ReproducibilityGuarantee.NONE,
            stable_output_order=kernel_class in {KernelClass.PREDICATE, KernelClass.CONSTRUCTIVE},
            fixed_reduction_order=False,
            fixed_scan_order=False,
            floating_atomics_allowed=kernel_class in {KernelClass.COARSE, KernelClass.METRIC, KernelClass.CONSTRUCTIVE},
            same_device_only=False,
            expected_max_overhead_factor=1.0,
            reason=(
                "default mode preserves only the stable output ordering already required by the current kernel shape "
                "and otherwise allows faster reduction or scan implementations"
            ),
        )

    return DeterminismPlan(
        kernel_class=kernel_class,
        mode=mode,
        guarantee=ReproducibilityGuarantee.SAME_DEVICE_BITWISE,
        stable_output_order=True,
        fixed_reduction_order=kernel_class in {KernelClass.METRIC, KernelClass.PREDICATE, KernelClass.CONSTRUCTIVE},
        fixed_scan_order=kernel_class in {KernelClass.METRIC, KernelClass.PREDICATE, KernelClass.CONSTRUCTIVE},
        floating_atomics_allowed=False,
        same_device_only=True,
        expected_max_overhead_factor=2.0 if kernel_class in {KernelClass.METRIC, KernelClass.CONSTRUCTIVE} else 1.5,
        reason=(
            "deterministic mode forces stable output ordering plus fixed scan and reduction order so the same input on "
            "the same device and driver can be reproduced bitwise"
        ),
    )
