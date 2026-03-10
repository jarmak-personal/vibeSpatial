from __future__ import annotations

import argparse
import json
from time import perf_counter

from shapely.affinity import translate

from vibespatial import (
    ExecutionMode,
    classify_segment_intersections,
    from_shapely_geometries,
    generate_segment_candidates,
    has_gpu_runtime,
)
from vibespatial.testing.synthetic import SyntheticSpec, generate_lines


def build_datasets(rows: int):
    base = list(
        generate_lines(
            SyntheticSpec(geometry_type="line", distribution="grid", count=rows, seed=0)
        ).geometries
    )
    shifted = [translate(geometry, xoff=0.5, yoff=0.5) for geometry in base]
    return from_shapely_geometries(base), from_shapely_geometries(shifted)


def _elapsed(left, right, candidates, *, dispatch_mode: ExecutionMode) -> tuple[float, dict[str, int | str]]:
    started = perf_counter()
    result = classify_segment_intersections(
        left,
        right,
        candidate_pairs=candidates,
        dispatch_mode=dispatch_mode,
    )
    elapsed = perf_counter() - started
    return elapsed, {
        "selected_runtime": result.runtime_selection.selected.value,
        "candidate_pairs": int(result.candidate_pairs),
        "proper_pairs": int((result.kinds == 1).sum()),
        "touch_pairs": int((result.kinds == 2).sum()),
        "overlap_pairs": int((result.kinds == 3).sum()),
        "ambiguous_pairs": int(result.ambiguous_rows.size),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark CPU vs GPU segment intersection classification.")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--tile-size", type=int, default=512)
    args = parser.parse_args(argv)

    left, right = build_datasets(args.rows)
    candidates = generate_segment_candidates(left, right, tile_size=args.tile_size)

    cpu_elapsed, cpu_stats = _elapsed(left, right, candidates, dispatch_mode=ExecutionMode.CPU)

    result = {
        "rows": args.rows,
        "tile_size": args.tile_size,
        "candidate_pairs": int(candidates.count),
        "cpu_elapsed_seconds": cpu_elapsed,
        "cpu": cpu_stats,
    }

    if has_gpu_runtime():
        gpu_cold_elapsed, gpu_stats = _elapsed(left, right, candidates, dispatch_mode=ExecutionMode.GPU)
        gpu_warm_elapsed, _ = _elapsed(left, right, candidates, dispatch_mode=ExecutionMode.GPU)
        result["gpu"] = {
            **gpu_stats,
            "cold_elapsed_seconds": gpu_cold_elapsed,
            "warm_elapsed_seconds": gpu_warm_elapsed,
            "warm_speedup_vs_cpu": (cpu_elapsed / gpu_warm_elapsed) if gpu_warm_elapsed else None,
        }
    else:
        result["gpu"] = {
            "available": False,
        }

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
