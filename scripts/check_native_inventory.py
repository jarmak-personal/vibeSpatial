#!/usr/bin/env python3
"""Ratchet Native* inventory debt during the full-coverage PRD hold."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = REPO_ROOT / "docs" / "dev" / "native-format-inventory.md"

# Baseline from docs/dev/native-format-inventory.md as of the Native* PRD hold.
# Decrease these counts as surfaces graduate. Do not increase them to hide
# newly-added partial/debt/implicit-host inventory rows.
PARTIAL_ROW_BASELINE = 0
DEBT_ROW_BASELINE = 0
IMPLICIT_HOST_ROW_BASELINE = 0


@dataclass(frozen=True)
class NativeInventoryCounts:
    rows: int
    partial_rows: int
    debt_rows: int
    implicit_host_rows: int


@dataclass(frozen=True)
class NativeInventoryBaselines:
    partial_rows: int = PARTIAL_ROW_BASELINE
    debt_rows: int = DEBT_ROW_BASELINE
    implicit_host_rows: int = IMPLICIT_HOST_ROW_BASELINE


@dataclass(frozen=True)
class NativeInventoryReport:
    counts: NativeInventoryCounts
    baselines: NativeInventoryBaselines
    strict: bool
    ok: bool
    violations: tuple[str, ...]


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_inventory_row(line: str) -> bool:
    if not line.startswith("| "):
        return False
    if line.startswith("|---") or line.startswith("| Surface "):
        return False
    return len(_split_markdown_row(line)) == 5


def inventory_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    in_inventory = False
    for line in text.splitlines():
        if line == "## Inventory":
            in_inventory = True
            continue
        if in_inventory and line.startswith("## "):
            break
        if in_inventory and _is_inventory_row(line):
            rows.append(_split_markdown_row(line))
    return rows


def count_inventory(text: str) -> NativeInventoryCounts:
    rows = inventory_rows(text)
    partial_rows = 0
    debt_rows = 0
    implicit_host_rows = 0
    token_re = {
        "partial": re.compile(r"\bpartial\b"),
        "debt": re.compile(r"\bdebt\b"),
        "implicit_host": re.compile(r"\bimplicit-host\b"),
    }
    for row in rows:
        transfer_class = row[3].lower()
        coverage = row[4].lower()
        searchable = f"{transfer_class} {coverage}"
        partial_rows += int(bool(token_re["partial"].search(coverage)))
        debt_rows += int(bool(token_re["debt"].search(searchable)))
        implicit_host_rows += int(bool(token_re["implicit_host"].search(searchable)))
    return NativeInventoryCounts(
        rows=len(rows),
        partial_rows=partial_rows,
        debt_rows=debt_rows,
        implicit_host_rows=implicit_host_rows,
    )


def evaluate_counts(
    counts: NativeInventoryCounts,
    baselines: NativeInventoryBaselines | None = None,
    *,
    strict: bool = False,
) -> NativeInventoryReport:
    if baselines is None:
        baselines = NativeInventoryBaselines()
    violations: list[str] = []
    if counts.rows == 0:
        violations.append("Native inventory table has no surface rows")

    checks = (
        ("partial", counts.partial_rows, baselines.partial_rows),
        ("debt", counts.debt_rows, baselines.debt_rows),
        ("implicit-host", counts.implicit_host_rows, baselines.implicit_host_rows),
    )
    for label, current, baseline in checks:
        if strict:
            if current:
                violations.append(f"{label} rows remain: {current}")
        elif current > baseline:
            violations.append(
                f"{label} rows regressed: {current} exceeds baseline {baseline}"
            )

    return NativeInventoryReport(
        counts=counts,
        baselines=baselines,
        strict=strict,
        ok=not violations,
        violations=tuple(violations),
    )


def _format_report(report: NativeInventoryReport) -> str:
    mode = "strict" if report.strict else "ratchet"
    lines = [
        f"Native inventory {mode}: "
        f"partial={report.counts.partial_rows}/{report.baselines.partial_rows}, "
        f"debt={report.counts.debt_rows}/{report.baselines.debt_rows}, "
        f"implicit-host={report.counts.implicit_host_rows}/"
        f"{report.baselines.implicit_host_rows}, rows={report.counts.rows}",
    ]
    for violation in report.violations:
        lines.append(f"- {violation}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the Native* inventory partial/debt ratchet.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY,
        help="Path to native-format-inventory.md.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require zero partial/debt/implicit-host rows.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    args = parser.parse_args(argv)

    text = args.inventory.read_text(encoding="utf-8")
    report = evaluate_counts(count_inventory(text), strict=args.strict)
    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print(_format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
