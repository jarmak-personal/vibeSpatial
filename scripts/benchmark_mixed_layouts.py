from __future__ import annotations

import argparse
from pathlib import Path

from vibespatial.testing.mixed_layouts import benchmark_matrix, write_benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scales",
        nargs="+",
        type=int,
        default=[100_000, 1_000_000],
        help="Benchmark scales to simulate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = benchmark_matrix(tuple(args.scales), seed=args.seed)
    if args.output is not None:
        write_benchmark_report(args.output, results)

    for result in results:
        print(
            f"{result.dataset_name:17} scale={result.scale:>8} "
            f"purity={result.tagged_warp_purity:.3f} "
            f"tagged_ms={result.tagged_prep_ms:>7.2f} "
            f"separate_ms={result.separate_prep_ms:>7.2f} "
            f"sort_ms={result.sort_partition_prep_ms:>7.2f} "
            f"payload_mb={result.payload_bytes / 1_000_000:>7.2f} "
            f"sort_mb={result.sort_partition_bytes / 1_000_000:>7.2f} "
            f"promoted_mb={result.promoted_bytes / 1_000_000:>7.2f} "
            f"recommend={result.recommended_default}"
        )


if __name__ == "__main__":
    main()
