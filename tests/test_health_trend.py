from __future__ import annotations

import json
from pathlib import Path

from scripts import health_trend


def test_build_trend_rows_groups_by_tier_and_keeps_last_n(tmp_path: Path) -> None:
    bootstrap_old = tmp_path / "bootstrap-old.json"
    bootstrap_new = tmp_path / "bootstrap-new.json"
    contract = tmp_path / "contract.json"

    bootstrap_old.write_text(
        json.dumps(
            {
                "tier": "bootstrap",
                "status": "pass",
                "tests": {"counts": {"passed": 1, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}},
                "properties": {"total_distance": 0.0},
            }
        ),
        encoding="utf-8",
    )
    bootstrap_new.write_text(
        json.dumps(
            {
                "tier": "bootstrap",
                "status": "fail",
                "tests": {"counts": {"passed": 2, "failed": 1, "skipped": 0, "xfailed": 0, "xpassed": 0}},
                "properties": {"total_distance": 1.0},
            }
        ),
        encoding="utf-8",
    )
    contract.write_text(
        json.dumps(
            {
                "tier": "contract",
                "status": "fail",
                "required_passing": 2,
                "required_total": 3,
                "surfaces": [
                    {"name": "versioning", "ok": True},
                    {"name": "overlay", "ok": False},
                ],
            }
        ),
        encoding="utf-8",
    )

    bootstrap_old.touch()
    bootstrap_new.touch()
    contract.touch()

    rows = health_trend.build_trend_rows(
        [bootstrap_old, bootstrap_new, contract],
        last=1,
    )

    assert [row.tier for row in rows] == ["bootstrap", "contract"]
    assert rows[0].status == "FAIL"
    assert "smoke 2/3 passed" in rows[0].summary
    assert "failing=overlay" in rows[1].summary


def test_main_json_outputs_serialized_rows(tmp_path: Path, capsys) -> None:
    report_path = tmp_path / "release.json"
    report_path.write_text(
        json.dumps(
            {
                "tier": "release",
                "status": "unmeasured",
                "checks": [{"name": "versioning", "status": "pass"}],
            }
        ),
        encoding="utf-8",
    )

    exit_code = health_trend.main([str(report_path), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"tier": "release"' in captured.out
