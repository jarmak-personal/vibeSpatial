#!/usr/bin/env python3
"""Reproduce an R2 Q11 component-first arm with contemporaneous identity."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from q12_dense_experiment import (
    _dataset_identity,
    _environment_identity,
    _source_tree_identity,
)

CAPSULE = Path(__file__).resolve().parent
REPO = CAPSULE.parents[2]
DATA = Path("/home/picard/datasets/spatialbench/v0.1.0/sf100-geoparquet")
IMPLEMENTATION = CAPSULE / "q11_component_first.py"


def _identity(variant: str, zone_frames: int, profile: bool) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "complete_contemporaneous",
        "captured_at": datetime.now(UTC).isoformat(),
        "source": _source_tree_identity(IMPLEMENTATION, Path(__file__).resolve()),
        "dataset": _dataset_identity(DATA),
        "environment": _environment_identity(),
        "measurement": {
            "query": "SF100 Q11 first 4M-row trip batch",
            "variant": variant,
            "zone_frames": zone_frames,
            "profile_enabled": profile,
            "process_isolation": "one command/process per arm",
            "warmup_runs": 0,
            "repeat_runs": 1,
            "statistic": "single cold observation",
            "clock": "time.perf_counter wall time",
            "timed_scope": "all selected public query_pair_aggregate calls",
            "excluded_scope": (
                "trip/zone reads and host component fixture decomposition"
            ),
            "profile_protocol": (
                "profile arms are separate instrumented processes and are not "
                "used as the wall-time decision arms"
            ),
            "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("parent", "component"), required=True)
    parser.add_argument("--zone-frames", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    identity = _identity(args.variant, args.zone_frames, args.profile)
    with tempfile.TemporaryDirectory(prefix="vs-r2-q11-identified-") as directory:
        raw_output = Path(directory) / "measurement.json"
        command = [
            sys.executable,
            str(IMPLEMENTATION),
            "--variant",
            args.variant,
            "--zone-frames",
            str(args.zone_frames),
            "--output",
            str(raw_output),
        ]
        if args.profile:
            command.append("--profile")
        environment = dict(os.environ)
        environment.setdefault("VIBESPATIAL_STRICT_NATIVE", "1")
        subprocess.run(command, cwd=REPO, env=environment, check=True)
        payload = json.loads(raw_output.read_text())

    payload["evidence_identity"] = identity
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
