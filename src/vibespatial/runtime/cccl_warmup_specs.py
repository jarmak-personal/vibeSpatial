from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None


class AlgorithmFamily(StrEnum):
    EXCLUSIVE_SCAN = "exclusive_scan"
    SELECT = "select"
    REDUCE_INTO = "reduce_into"
    SEGMENTED_REDUCE = "segmented_reduce"
    LOWER_BOUND = "lower_bound"
    UPPER_BOUND = "upper_bound"
    RADIX_SORT = "radix_sort"
    MERGE_SORT = "merge_sort"
    UNIQUE_BY_KEY = "unique_by_key"
    SEGMENTED_SORT = "segmented_sort"


@dataclass(frozen=True, slots=True)
class CCCLWarmupSpec:
    """Specification for a CCCL make_* pre-compilation target."""
    name: str
    family: AlgorithmFamily
    key_dtype: Any
    value_dtype: Any | None
    op_name: str


@dataclass(frozen=True, slots=True)
class WarmupDiagnostic:
    name: str
    elapsed_ms: float
    success: bool
    error: str = ""


def build_spec_registry() -> dict[str, CCCLWarmupSpec]:
    S = CCCLWarmupSpec
    F = AlgorithmFamily
    if cp is not None:
        i32 = cp.dtype(cp.int32)
        i64 = cp.dtype(cp.int64)
        u64 = cp.dtype(cp.uint64)
        f64 = cp.dtype(cp.float64)
    else:
        i32 = "int32"
        i64 = "int64"
        u64 = "uint64"
        f64 = "float64"
    return {
        "exclusive_scan_i32": S("exclusive_scan_i32", F.EXCLUSIVE_SCAN, i32, None, "sum"),
        "exclusive_scan_i64": S("exclusive_scan_i64", F.EXCLUSIVE_SCAN, i64, None, "sum"),
        "select_i32": S("select_i32", F.SELECT, i32, None, "select_predicate"),
        "select_i64": S("select_i64", F.SELECT, i64, None, "select_predicate"),
        "reduce_sum_f64": S("reduce_sum_f64", F.REDUCE_INTO, f64, None, "sum"),
        "reduce_sum_i32": S("reduce_sum_i32", F.REDUCE_INTO, i32, None, "sum"),
        "segmented_reduce_sum_i32": S("segmented_reduce_sum_i32", F.SEGMENTED_REDUCE, i32, None, "sum"),
        "segmented_reduce_sum_f64": S("segmented_reduce_sum_f64", F.SEGMENTED_REDUCE, f64, None, "sum"),
        "segmented_reduce_min_f64": S("segmented_reduce_min_f64", F.SEGMENTED_REDUCE, f64, None, "min"),
        "segmented_reduce_max_f64": S("segmented_reduce_max_f64", F.SEGMENTED_REDUCE, f64, None, "max"),
        "lower_bound_i32": S("lower_bound_i32", F.LOWER_BOUND, i32, None, "none"),
        "lower_bound_i64": S("lower_bound_i64", F.LOWER_BOUND, i64, None, "none"),
        "lower_bound_f64": S("lower_bound_f64", F.LOWER_BOUND, f64, None, "none"),
        "lower_bound_u64": S("lower_bound_u64", F.LOWER_BOUND, u64, None, "none"),
        "upper_bound_i32": S("upper_bound_i32", F.UPPER_BOUND, i32, None, "none"),
        "upper_bound_i64": S("upper_bound_i64", F.UPPER_BOUND, i64, None, "none"),
        "upper_bound_f64": S("upper_bound_f64", F.UPPER_BOUND, f64, None, "none"),
        "upper_bound_u64": S("upper_bound_u64", F.UPPER_BOUND, u64, None, "none"),
        "radix_sort_i32_i32": S("radix_sort_i32_i32", F.RADIX_SORT, i32, i32, "ascending"),
        "radix_sort_i64_i32": S("radix_sort_i64_i32", F.RADIX_SORT, i64, i32, "ascending"),
        "radix_sort_u64_i32": S("radix_sort_u64_i32", F.RADIX_SORT, u64, i32, "ascending"),
        "merge_sort_u64_i32": S("merge_sort_u64_i32", F.MERGE_SORT, u64, i32, "less_than"),
        "unique_by_key_i32_i32": S("unique_by_key_i32_i32", F.UNIQUE_BY_KEY, i32, i32, "equal_to"),
        "unique_by_key_u64_i32": S("unique_by_key_u64_i32", F.UNIQUE_BY_KEY, u64, i32, "equal_to"),
        "segmented_sort_asc_f64": S("segmented_sort_asc_f64", F.SEGMENTED_SORT, f64, i32, "less_than"),
        "segmented_sort_asc_i32": S("segmented_sort_asc_i32", F.SEGMENTED_SORT, i32, i32, "less_than"),
    }
