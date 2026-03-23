from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vibespatial.predicates.binary import BinaryPredicateResult, evaluate_binary_predicate
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import Residency

PredicateInput = Sequence[object | None] | np.ndarray | object


def _variant(
    name: str,
):
    return register_kernel_variant(
        name,
        "cpu",
        kernel_class=KernelClass.PREDICATE,
        geometry_families=("point", "linestring", "polygon", "multipoint", "multilinestring", "multipolygon"),
        execution_modes=(ExecutionMode.CPU,),
        supports_mixed=True,
        tags=("exact-refine", "binary-predicate"),
    )


def _gpu_variant(
    name: str,
):
    return register_kernel_variant(
        name,
        "gpu-cuda-python",
        kernel_class=KernelClass.PREDICATE,
        geometry_families=("point", "linestring", "polygon", "multipoint", "multilinestring", "multipolygon"),
        execution_modes=(ExecutionMode.GPU,),
        preferred_residency=Residency.DEVICE,
        supports_mixed=True,
        tags=("exact-refine", "binary-predicate", "cuda-python", "point-centric"),
    )


@_variant("intersects")
@_gpu_variant("intersects")
def intersects_exact(
    left: PredicateInput,
    right: PredicateInput,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> BinaryPredicateResult:
    return evaluate_binary_predicate("intersects", left, right, dispatch_mode=dispatch_mode, precision=precision)


@_variant("within")
@_gpu_variant("within")
def within_exact(
    left: PredicateInput,
    right: PredicateInput,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> BinaryPredicateResult:
    return evaluate_binary_predicate("within", left, right, dispatch_mode=dispatch_mode, precision=precision)


@_variant("contains")
@_gpu_variant("contains")
def contains_exact(
    left: PredicateInput,
    right: PredicateInput,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> BinaryPredicateResult:
    return evaluate_binary_predicate("contains", left, right, dispatch_mode=dispatch_mode, precision=precision)


@_variant("covers")
@_gpu_variant("covers")
def covers_exact(
    left: PredicateInput,
    right: PredicateInput,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> BinaryPredicateResult:
    return evaluate_binary_predicate("covers", left, right, dispatch_mode=dispatch_mode, precision=precision)


@_variant("covered_by")
@_gpu_variant("covered_by")
def covered_by_exact(
    left: PredicateInput,
    right: PredicateInput,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> BinaryPredicateResult:
    return evaluate_binary_predicate("covered_by", left, right, dispatch_mode=dispatch_mode, precision=precision)


@_variant("touches")
@_gpu_variant("touches")
def touches_exact(
    left: PredicateInput,
    right: PredicateInput,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> BinaryPredicateResult:
    return evaluate_binary_predicate("touches", left, right, dispatch_mode=dispatch_mode, precision=precision)


@_variant("crosses")
@_gpu_variant("crosses")
def crosses_exact(
    left: PredicateInput,
    right: PredicateInput,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> BinaryPredicateResult:
    return evaluate_binary_predicate("crosses", left, right, dispatch_mode=dispatch_mode, precision=precision)


@_variant("contains_properly")
@_gpu_variant("contains_properly")
def contains_properly_exact(
    left: PredicateInput,
    right: PredicateInput,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> BinaryPredicateResult:
    return evaluate_binary_predicate(
        "contains_properly",
        left,
        right,
        dispatch_mode=dispatch_mode,
        precision=precision,
    )


@_variant("overlaps")
@_gpu_variant("overlaps")
def overlaps_exact(
    left: PredicateInput,
    right: PredicateInput,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> BinaryPredicateResult:
    return evaluate_binary_predicate("overlaps", left, right, dispatch_mode=dispatch_mode, precision=precision)


@_variant("disjoint")
@_gpu_variant("disjoint")
def disjoint_exact(
    left: PredicateInput,
    right: PredicateInput,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> BinaryPredicateResult:
    return evaluate_binary_predicate("disjoint", left, right, dispatch_mode=dispatch_mode, precision=precision)


@_variant("equals")
@_gpu_variant("equals")
def equals_exact(
    left: PredicateInput,
    right: PredicateInput,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> BinaryPredicateResult:
    return evaluate_binary_predicate("equals", left, right, dispatch_mode=dispatch_mode, precision=precision)
