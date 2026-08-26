#!/usr/bin/env python3
"""Run one current-source vibeSpatial shootout arm without a CPU comparator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

CAPSULE = Path(__file__).resolve().parent
REPO = CAPSULE.parents[2]
sys.path.insert(0, str(REPO))

from vibespatial.bench.provenance import source_identity  # noqa: E402
from vibespatial.bench.shootout import (  # noqa: E402
    _run_harness,
    _set_repo_shim_precedence,
    shootout_workload_identity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path)
    parser.add_argument("--scale", default="1M")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--profile-mode", choices=("off", "counters", "full"), default="counters")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    script = args.script.resolve()
    environment = os.environ.copy()
    environment["VSBENCH_SCALE"] = args.scale
    environment["VIBESPATIAL_HOTPATH_TRACE"] = "off"
    environment["VIBESPATIAL_SHOOTOUT_PROFILE_MODE"] = args.profile_mode
    environment.setdefault("VIBESPATIAL_STRICT_NATIVE", "1")
    _set_repo_shim_precedence(environment, enabled=True)
    run = _run_harness(
        label="vibespatial",
        python_cmd=[sys.executable],
        script=script,
        repeat=args.repeat,
        warmup=not args.no_warmup,
        pipeline_warm=True,
        env=environment,
        timeout=args.timeout,
        quiet=False,
        profile=args.profile_mode != "off",
    )
    payload = {
        "schema_version": 1,
        "type": "vibespatial_only_shootout",
        "captured_at": datetime.now(UTC).isoformat(),
        "script": str(script),
        "scale": args.scale,
        "repeat": args.repeat,
        "warmup": not args.no_warmup,
        "profile_mode": args.profile_mode,
        "source": source_identity(),
        "workload": shootout_workload_identity(script),
        "run": run.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if run.error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
