from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
import shapely
from shapely.affinity import translate

from vibespatial import benchmark_binary_predicate, evaluate_binary_predicate
from vibespatial.testing import SyntheticSpec, generate_polygons


def build_inputs(rows: int, *, overlap_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(
        list(
            generate_polygons(
                SyntheticSpec(geometry_type="polygon", distribution="regular-grid", count=rows, seed=0)
            ).geometries
        ),
        dtype=object,
    )
    right = left.copy()
    cutoff = int(rows * overlap_ratio)
    if cutoff < rows:
        shifted = [translate(geometry, xoff=10_000.0, yoff=10_000.0) for geometry in right[cutoff:]]
        right[cutoff:] = np.asarray(shifted, dtype=object)
    return left, right


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark exact binary-predicate refine kernels.")
    parser.add_argument("--predicate", default="intersects")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--overlap-ratio", type=float, default=0.2)
    args = parser.parse_args()

    left, right = build_inputs(args.rows, overlap_ratio=args.overlap_ratio)

    started = perf_counter()
    naive = getattr(shapely, args.predicate)(left, right)
    naive_elapsed = perf_counter() - started

    started = perf_counter()
    refined = evaluate_binary_predicate(args.predicate, left, right, null_behavior="false")
    refined_elapsed = perf_counter() - started

    stats = benchmark_binary_predicate(args.predicate, left, right)
    print(
        json.dumps(
            {
                "predicate": args.predicate,
                "rows": args.rows,
                "overlap_ratio": args.overlap_ratio,
                "naive_true_rows": int(np.count_nonzero(naive)),
                "refined_true_rows": int(np.count_nonzero(np.asarray(refined.values, dtype=bool))),
                "candidate_rows": stats["candidate_rows"],
                "coarse_true_rows": stats["coarse_true_rows"],
                "coarse_false_rows": stats["coarse_false_rows"],
                "naive_elapsed_seconds": naive_elapsed,
                "refined_elapsed_seconds": refined_elapsed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
