from __future__ import annotations

import argparse
import json

from vibespatial.profile_rails import profile_spatial_query_stack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile repo-owned spatial query stack stages.")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--overlap-ratio", type=float, default=0.2)
    parser.add_argument("--predicate", default="intersects")
    parser.add_argument("--output-format", default="indices")
    parser.add_argument("--sort", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--nvtx", action="store_true", help="Emit NVTX ranges when the nvtx package is available.")
    args = parser.parse_args(argv)

    traces: list[dict[str, object]] = []
    for _ in range(args.repeat):
        trace = profile_spatial_query_stack(
            rows=args.rows,
            overlap_ratio=args.overlap_ratio,
            predicate=args.predicate,
            sort=args.sort,
            output_format=args.output_format,
            enable_nvtx=args.nvtx,
        )
        traces.append(trace.to_dict())

    payload = {
        "rows": args.rows,
        "overlap_ratio": args.overlap_ratio,
        "predicate": args.predicate,
        "output_format": args.output_format,
        "sort": args.sort,
        "repeat": args.repeat,
        "nvtx_requested": args.nvtx,
        "traces": traces,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
