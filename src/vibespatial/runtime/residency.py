from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Residency(StrEnum):
    HOST = "host"
    DEVICE = "device"


class TransferTrigger(StrEnum):
    USER_MATERIALIZATION = "user-materialization"
    EXPLICIT_RUNTIME_REQUEST = "explicit-runtime-request"
    UNSUPPORTED_GPU_PATH = "unsupported-gpu-path"
    INTEROP_VIEW = "interop-view"


def normalize_residency(value: Residency | str) -> Residency:
    return value if isinstance(value, Residency) else Residency(value)


def combined_residency(*values: object) -> Residency:
    """Return DEVICE when any provided value is device-resident, else HOST.

    Accepts explicit ``Residency`` values, strings, or arbitrary objects with a
    ``residency`` attribute. This lets dispatch wrappers preserve device
    stickiness without special-casing every owned-array type.
    """
    for value in values:
        if value is None:
            continue
        residency = getattr(value, "residency", value)
        try:
            if normalize_residency(residency) is Residency.DEVICE:
                return Residency.DEVICE
        except Exception:
            continue
    return Residency.HOST


@dataclass(frozen=True)
class ResidencyPlan:
    current: Residency
    target: Residency
    trigger: TransferTrigger
    transfer_required: bool
    visible_to_user: bool
    zero_copy_eligible: bool
    reason: str


def select_residency_plan(
    *,
    current: Residency | str,
    target: Residency | str,
    trigger: TransferTrigger | str,
) -> ResidencyPlan:
    current_residency = normalize_residency(current)
    target_residency = normalize_residency(target)
    normalized_trigger = (
        trigger if isinstance(trigger, TransferTrigger) else TransferTrigger(trigger)
    )

    if current_residency is target_residency:
        zero_copy = normalized_trigger is TransferTrigger.INTEROP_VIEW
        return ResidencyPlan(
            current=current_residency,
            target=target_residency,
            trigger=normalized_trigger,
            transfer_required=False,
            visible_to_user=False,
            zero_copy_eligible=zero_copy,
            reason="buffers already reside on the requested side",
        )

    if normalized_trigger is TransferTrigger.INTEROP_VIEW:
        return ResidencyPlan(
            current=current_residency,
            target=target_residency,
            trigger=normalized_trigger,
            transfer_required=False,
            visible_to_user=False,
            zero_copy_eligible=True,
            reason="interop views should prefer shared ownership over a copy when layouts align",
        )

    if normalized_trigger is TransferTrigger.USER_MATERIALIZATION:
        return ResidencyPlan(
            current=current_residency,
            target=target_residency,
            trigger=normalized_trigger,
            transfer_required=True,
            visible_to_user=True,
            zero_copy_eligible=False,
            reason="host materialization is an explicit user-visible transfer boundary",
        )

    if normalized_trigger is TransferTrigger.UNSUPPORTED_GPU_PATH:
        return ResidencyPlan(
            current=current_residency,
            target=target_residency,
            trigger=normalized_trigger,
            transfer_required=True,
            visible_to_user=True,
            zero_copy_eligible=False,
            reason="runtime-mandated fallback transfers must stay visible to avoid silent host execution",
        )

    return ResidencyPlan(
        current=current_residency,
        target=target_residency,
        trigger=normalized_trigger,
        transfer_required=True,
        visible_to_user=True,
        zero_copy_eligible=False,
        reason="explicit runtime requests may move buffers, but the transfer must remain observable",
    )
