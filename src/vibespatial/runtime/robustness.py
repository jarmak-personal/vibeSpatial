from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .nulls import GeometryPresence
from .precision import KernelClass, PrecisionMode, PrecisionPlan


class RobustnessGuarantee(StrEnum):
    EXACT = "exact"
    BOUNDED_ERROR = "bounded-error"
    BEST_EFFORT = "best-effort"


class PredicateFallback(StrEnum):
    NONE = "none"
    SELECTIVE_FP64 = "selective-fp64"
    EXPANSION_ARITHMETIC = "expansion-arithmetic"
    RATIONAL_RECONSTRUCTION = "rational-reconstruction"


class TopologyPolicy(StrEnum):
    PRESERVE = "preserve"
    SNAP_GRID = "snap-grid"
    BEST_EFFORT = "best-effort"


@dataclass(frozen=True)
class RobustnessPlan:
    kernel_class: KernelClass
    guarantee: RobustnessGuarantee
    predicate_fallback: PredicateFallback
    topology_policy: TopologyPolicy
    handles_nulls: bool
    handles_empties: bool
    reason: str


@dataclass(frozen=True)
class NumericalErrorEnvelope:
    """Conservative ambiguity bound carried between GPU planning stages.

    ``bound`` may be a host scalar or a device scalar.  Consumers compare a
    decision margin with this value and refine only decisions inside the
    envelope.  The carrier deliberately describes numerical uncertainty, not
    an accuracy preference: exact-mode consumers must still resolve every
    ambiguous decision with their declared fallback.
    """

    bound: Any
    quantity: str
    arithmetic_precision: PrecisionMode
    derivation: str
    conservative: bool = True

    @classmethod
    def exact(cls, *, quantity: str) -> NumericalErrorEnvelope:
        return cls(
            bound=0.0,
            quantity=quantity,
            arithmetic_precision=PrecisionMode.FP64,
            derivation="exact or already-refined decisions have zero remaining ambiguity",
        )


def select_robustness_plan(
    *,
    kernel_class: KernelClass,
    precision_plan: PrecisionPlan,
    null_state: GeometryPresence | None = None,
    empty_state: GeometryPresence | None = None,
) -> RobustnessPlan:
    del null_state, empty_state

    if kernel_class is KernelClass.COARSE:
        return RobustnessPlan(
            kernel_class=kernel_class,
            guarantee=RobustnessGuarantee.BOUNDED_ERROR,
            predicate_fallback=PredicateFallback.NONE,
            topology_policy=TopologyPolicy.BEST_EFFORT,
            handles_nulls=True,
            handles_empties=True,
            reason="coarse kernels may use bounded-error arithmetic because they do not decide topology",
        )

    if kernel_class is KernelClass.METRIC:
        return RobustnessPlan(
            kernel_class=kernel_class,
            guarantee=RobustnessGuarantee.BOUNDED_ERROR,
            predicate_fallback=PredicateFallback.NONE,
            topology_policy=TopologyPolicy.BEST_EFFORT,
            handles_nulls=True,
            handles_empties=True,
            reason="metric kernels accept bounded error but must preserve null and empty semantics",
        )

    if kernel_class is KernelClass.PREDICATE:
        fallback = (
            PredicateFallback.EXPANSION_ARITHMETIC
            if precision_plan.compute_precision is PrecisionMode.FP32
            else PredicateFallback.SELECTIVE_FP64
        )
        return RobustnessPlan(
            kernel_class=kernel_class,
            guarantee=RobustnessGuarantee.EXACT,
            predicate_fallback=fallback,
            topology_policy=TopologyPolicy.PRESERVE,
            handles_nulls=True,
            handles_empties=True,
            reason="predicate kernels must return the correct sign even for nearly-degenerate inputs",
        )

    return RobustnessPlan(
        kernel_class=kernel_class,
        guarantee=RobustnessGuarantee.EXACT,
        predicate_fallback=PredicateFallback.RATIONAL_RECONSTRUCTION,
        topology_policy=TopologyPolicy.PRESERVE,
        handles_nulls=True,
        handles_empties=True,
        reason="constructive kernels must preserve topology and use exact-style fallback for intersection decisions",
    )
