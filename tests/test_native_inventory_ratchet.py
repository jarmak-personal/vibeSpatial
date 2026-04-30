from __future__ import annotations

from pathlib import Path

from scripts.check_native_inventory import (
    NativeInventoryBaselines,
    count_inventory,
    evaluate_counts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_INVENTORY = """
## Inventory

| Surface | Current GPU Shape | Target Carrier | Transfer Class | Coverage |
|---|---|---|---|---|
| Covered | shape | carrier | export only | covered |
| Partial | shape | carrier | public bool Series is debt before rowset | partial |
| Implicit | shape | carrier | implicit-host | covered |

## Immediate Gaps
"""


def test_native_inventory_counts_table_rows_only() -> None:
    counts = count_inventory(SAMPLE_INVENTORY)

    assert counts.rows == 3
    assert counts.partial_rows == 1
    assert counts.debt_rows == 1
    assert counts.implicit_host_rows == 1


def test_native_inventory_ratchet_allows_current_baseline() -> None:
    counts = count_inventory(SAMPLE_INVENTORY)
    report = evaluate_counts(
        counts,
        NativeInventoryBaselines(
            partial_rows=1,
            debt_rows=1,
            implicit_host_rows=1,
        ),
    )

    assert report.ok is True
    assert report.violations == ()


def test_native_inventory_ratchet_rejects_regression() -> None:
    counts = count_inventory(SAMPLE_INVENTORY)
    report = evaluate_counts(
        counts,
        NativeInventoryBaselines(
            partial_rows=0,
            debt_rows=1,
            implicit_host_rows=1,
        ),
    )

    assert report.ok is False
    assert report.violations == ("partial rows regressed: 1 exceeds baseline 0",)


def test_native_inventory_strict_requires_zero_debt() -> None:
    counts = count_inventory(SAMPLE_INVENTORY)
    report = evaluate_counts(
        counts,
        NativeInventoryBaselines(
            partial_rows=3,
            debt_rows=3,
            implicit_host_rows=3,
        ),
        strict=True,
    )

    assert report.ok is False
    assert report.violations == (
        "partial rows remain: 1",
        "debt rows remain: 1",
        "implicit-host rows remain: 1",
    )


def test_native_inventory_default_baseline_tracks_current_document() -> None:
    inventory = REPO_ROOT / "docs" / "dev" / "native-format-inventory.md"
    counts = count_inventory(inventory.read_text(encoding="utf-8"))
    report = evaluate_counts(counts)

    assert report.ok is True
    assert report.baselines.partial_rows == counts.partial_rows
    assert report.baselines.debt_rows == counts.debt_rows
    assert report.baselines.implicit_host_rows == counts.implicit_host_rows
