from __future__ import annotations

import argparse
import sys

from vibespatial import ExecutionMode
from vibespatial.testing import ORACLE_SCENARIOS, assert_matches_shapely


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic Shapely oracle check for a kernel scenario.")
    parser.add_argument("--kernel", required=True, choices=sorted(ORACLE_SCENARIOS))
    parser.add_argument("--scale", default="1K", help="Synthetic scale preset or integer row count.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dispatch-mode",
        default=ExecutionMode.CPU.value,
        choices=[mode.value for mode in ExecutionMode],
    )
    parser.add_argument("--determinism", action="store_true", help="Repeat the operation and compare reruns.")
    args = parser.parse_args(argv)

    scale: int | str
    scale = int(args.scale) if args.scale.isdigit() else args.scale
    scenario = ORACLE_SCENARIOS[args.kernel](scale=scale, seed=args.seed, check_determinism=args.determinism)

    try:
        comparison = assert_matches_shapely(
            scenario.operation,
            *scenario.args,
            dispatch_mode=args.dispatch_mode,
            config=scenario.config,
            **scenario.kwargs,
        )
    except AssertionError as error:
        print(str(error), file=sys.stderr)
        return 1

    print(
        "oracle_check: "
        f"kernel={scenario.name} rows={len(comparison.actual)} "
        f"requested={comparison.dispatch_mode.value} selected={comparison.selection.selected.value} "
        "status=match"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
