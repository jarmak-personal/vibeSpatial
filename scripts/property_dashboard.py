#!/usr/bin/env python3
"""Unified property dashboard — aggregate all check scripts into one view.

Reports every codebase property with: current violations, baseline,
distance (0.0 = fully satisfied), and status (OK / debt / regressed).

Usage:
    uv run python scripts/property_dashboard.py          # human-readable
    uv run python scripts/property_dashboard.py --json    # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# Property model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PropertyState:
    name: str
    code: str           # e.g. "ZCOPY", "VPAT", "ARCH"
    category: str       # structural, residency, performance, isolation, discoverability
    violations: int
    baseline: int | None  # None = hard-fail (no debt allowed)
    distance: float     # 0.0 = satisfied, 1.0 = maximally violated
    status: str         # "ok", "debt", "regressed"
    detail: str


# ---------------------------------------------------------------------------
# Collectors — import each check script's internals directly
# ---------------------------------------------------------------------------

def _collect_arch() -> list[PropertyState]:
    """ARCH001-007: hard-fail architecture lints."""
    from check_architecture_lints import run_advisory_checks, run_checks
    errors = run_checks(REPO_ROOT)
    advisory = run_advisory_checks(REPO_ROOT)
    count = len(errors)
    adv_count = len(advisory)
    return [
        PropertyState(
            name="architecture_lints",
            code="ARCH",
            category="structural",
            violations=count,
            baseline=None,
            distance=min(count / 10.0, 1.0) if count else 0.0,
            status="ok" if count == 0 else "regressed",
            detail=f"{count} hard-fail violations"
                   + (f", {adv_count} advisory" if adv_count else ""),
        ),
    ]


def _collect_zcopy() -> list[PropertyState]:
    """ZCOPY001-003: zero-copy compliance (ratchet)."""
    from check_zero_copy import _VIOLATION_BASELINE, run_checks
    errors, _suppressed = run_checks(REPO_ROOT)
    count = len(errors)
    baseline = _VIOLATION_BASELINE
    distance = count / max(baseline, 1)
    if count > baseline:
        status = "regressed"
    elif count == 0:
        status = "ok"
    else:
        status = "debt"
    return [
        PropertyState(
            name="zero_copy_compliance",
            code="ZCOPY",
            category="residency",
            violations=count,
            baseline=baseline,
            distance=round(distance, 3),
            status=status,
            detail=f"{count} violations (baseline {baseline})",
        ),
    ]


def _collect_vpat() -> list[PropertyState]:
    """VPAT001-004: performance anti-patterns (ratchet)."""
    from check_perf_patterns import _VIOLATION_BASELINE, run_checks
    errors = run_checks(REPO_ROOT)
    count = len(errors)
    baseline = _VIOLATION_BASELINE
    distance = count / max(baseline, 1)
    if count > baseline:
        status = "regressed"
    elif count == 0:
        status = "ok"
    else:
        status = "debt"
    return [
        PropertyState(
            name="gpu_first_patterns",
            code="VPAT",
            category="performance",
            violations=count,
            baseline=baseline,
            distance=round(distance, 3),
            status=status,
            detail=f"{count} violations (baseline {baseline})",
        ),
    ]


def _collect_igrd() -> list[PropertyState]:
    """IGRD001-002: import isolation (ratchet, two sub-properties)."""
    from check_import_guard import (
        _NUMPY_VIOLATION_BASELINE,
        _SHAPELY_VIOLATION_BASELINE,
        run_checks,
    )
    shapely_errors, numpy_errors = run_checks(REPO_ROOT)
    sc, nc = len(shapely_errors), len(numpy_errors)
    sb, nb = _SHAPELY_VIOLATION_BASELINE, _NUMPY_VIOLATION_BASELINE

    props = []
    for name, code, count, baseline in [
        ("shapely_isolation", "IGRD001", sc, sb),
        ("numpy_isolation", "IGRD002", nc, nb),
    ]:
        distance = count / max(baseline, 1)
        if count > baseline:
            status = "regressed"
        elif count == 0:
            status = "ok"
        else:
            status = "debt"
        props.append(PropertyState(
            name=name,
            code=code,
            category="isolation",
            violations=count,
            baseline=baseline,
            distance=round(distance, 3),
            status=status,
            detail=f"{count} violations (baseline {baseline})",
        ))
    return props


def _collect_maint() -> list[PropertyState]:
    """MAINT001-003: maintainability/discoverability (ratchet)."""
    from check_maintainability import _VIOLATION_BASELINE, run_checks
    errors = run_checks(REPO_ROOT)
    count = len(errors)
    baseline = _VIOLATION_BASELINE
    distance = count / max(baseline, 1)
    if count > baseline:
        status = "regressed"
    elif count == 0:
        status = "ok"
    else:
        status = "debt"
    return [
        PropertyState(
            name="intake_discoverability",
            code="MAINT",
            category="discoverability",
            violations=count,
            baseline=baseline,
            distance=round(distance, 3),
            status=status,
            detail=f"{count} violations (baseline {baseline})",
        ),
    ]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

ALL_COLLECTORS = [
    _collect_arch,
    _collect_zcopy,
    _collect_vpat,
    _collect_igrd,
    _collect_maint,
]


def snapshot() -> list[PropertyState]:
    """Run all collectors and return the full property state."""
    props: list[PropertyState] = []
    for collector in ALL_COLLECTORS:
        props.extend(collector())
    return props


def total_distance(props: list[PropertyState]) -> float:
    return round(sum(p.distance for p in props), 3)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_STATUS_ICON = {"ok": "OK", "debt": "~~", "regressed": "!!"}


def format_dashboard(props: list[PropertyState]) -> str:
    """Human-readable dashboard for agent context injection."""
    lines = ["# Codebase Property Dashboard", ""]

    satisfied = sum(1 for p in props if p.status == "ok")
    debt = sum(1 for p in props if p.status == "debt")
    regressed = sum(1 for p in props if p.status == "regressed")
    td = total_distance(props)
    lines.append(f"{satisfied}/{len(props)} clean | "
                 f"{debt} debt (holding) | "
                 f"{regressed} regressed | "
                 f"total distance = {td}\n")
    lines.append("distance: 0.00 = fully satisfied, "
                 "≤1.00 = at or below baseline, "
                 ">1.00 = regression\n")

    by_cat: dict[str, list[PropertyState]] = {}
    for p in props:
        by_cat.setdefault(p.category, []).append(p)

    for cat in sorted(by_cat):
        lines.append(f"## {cat}")
        for p in by_cat[cat]:
            icon = _STATUS_ICON[p.status]
            base = f" (baseline {p.baseline})" if p.baseline is not None else ""
            lines.append(f"  [{icon:>2}] {p.code:<8} {p.name}: "
                         f"{p.violations} violations{base}  d={p.distance:.2f}")
        lines.append("")

    return "\n".join(lines)


def format_json(props: list[PropertyState]) -> str:
    """Machine-readable JSON output."""
    return json.dumps({
        "properties": [asdict(p) for p in props],
        "total_distance": total_distance(props),
        "satisfied": sum(1 for p in props if p.status == "ok"),
        "total": len(props),
    }, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified codebase property dashboard."
    )
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON.")
    parser.parse_args(argv)

    props = snapshot()
    if "--json" in (argv or sys.argv[1:]):
        print(format_json(props))
    else:
        print(format_dashboard(props))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
