"""Verify all reorg shim files re-export their names correctly."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_validate_reorg_shims():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "validate_reorg.py")],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"Shim validation failed:\n{result.stdout}\n{result.stderr}"
