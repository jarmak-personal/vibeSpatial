from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_import_vibespatial_api_without_cupy(tmp_path: Path) -> None:
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "import builtins\n"
        "_real_import = builtins.__import__\n"
        "def _blocked(name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    if name == 'cupy' or name.startswith('cupy.'):\n"
        "        raise ModuleNotFoundError(\"No module named 'cupy'\")\n"
        "    return _real_import(name, globals, locals, fromlist, level)\n"
        "builtins.__import__ = _blocked\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{tmp_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(tmp_path)
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import vibespatial.api; "
                "from vibespatial.constructive.binary_constructive import binary_constructive_owned; "
                "from vibespatial.kernels.constructive import segmented_union_all; "
                "print('import-ok')"
            ),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert proc.returncode == 0, proc.stderr
    assert "import-ok" in proc.stdout
