from __future__ import annotations

import argparse
import json

from vibespatial.profile_rails import profile_join_kernel, profile_overlay_kernel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile join and overlay kernel stages.")
    parser.add_argument("--kernel", choices=("join", "overlay", "all"), default="all")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--join-rows", type=int)
    parser.add_argument("--overlay-rows", type=int)
    parser.add_argument("--overlap-ratio", type=float, default=0.2)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--nvtx", action="store_true", help="Emit NVTX ranges when the nvtx package is available.")
    args = parser.parse_args(argv)

    join_rows = args.join_rows or args.rows
    overlay_rows = args.overlay_rows or args.rows
    traces: list[dict[str, object]] = []

    for _ in range(args.repeat):
        if args.kernel in {"join", "all"}:
            trace = profile_join_kernel(
                rows=join_rows,
                overlap_ratio=args.overlap_ratio,
                tile_size=args.tile_size,
                enable_nvtx=args.nvtx,
            )
            traces.append(trace.to_dict())
        if args.kernel in {"overlay", "all"}:
            trace = profile_overlay_kernel(
                rows=overlay_rows,
                tile_size=max(args.tile_size, 512),
                enable_nvtx=args.nvtx,
            )
            traces.append(trace.to_dict())

    payload = {
        "kernel": args.kernel,
        "rows": args.rows,
        "join_rows": join_rows,
        "overlay_rows": overlay_rows,
        "repeat": args.repeat,
        "nvtx_requested": args.nvtx,
        "traces": traces,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
