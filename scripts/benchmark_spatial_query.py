from __future__ import annotations

import argparse
import json
from time import perf_counter

import geopandas
import numpy as np
from shapely.affinity import translate

from vibespatial.spatial_query import build_owned_spatial_index, nearest_spatial_index, query_spatial_index
from vibespatial.testing import SyntheticSpec, generate_points, generate_polygons


def build_query_inputs(rows: int, *, overlap_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    tree = np.asarray(
        list(
            generate_polygons(
                SyntheticSpec(geometry_type="polygon", distribution="regular-grid", count=rows, seed=4)
            ).geometries
        ),
        dtype=object,
    )
    query = tree.copy()
    cutoff = int(rows * overlap_ratio)
    if cutoff < rows:
        query[cutoff:] = np.asarray(
            [translate(geometry, xoff=10_000.0, yoff=10_000.0) for geometry in query[cutoff:]],
            dtype=object,
        )
    return tree, query


def build_nearest_inputs(rows: int) -> tuple[np.ndarray, np.ndarray]:
    tree = np.asarray(
        list(
            generate_points(
                SyntheticSpec(geometry_type="point", distribution="grid", count=rows, seed=0)
            ).geometries
        ),
        dtype=object,
    )
    query = np.asarray([translate(point, xoff=0.25, yoff=0.25) for point in tree], dtype=object)
    return tree, query


def build_outer_join_frames(rows: int, *, overlap_ratio: float) -> tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]:
    right = np.asarray(
        list(
            generate_polygons(
                SyntheticSpec(geometry_type="polygon", distribution="regular-grid", count=rows, seed=4)
            ).geometries
        ),
        dtype=object,
    )
    left = np.asarray([geometry.centroid for geometry in right], dtype=object)
    cutoff = int(rows * overlap_ratio)
    if cutoff < rows:
        left[cutoff:] = np.asarray(
            [translate(geometry, xoff=10_000.0, yoff=10_000.0) for geometry in left[cutoff:]],
            dtype=object,
        )
    left = geopandas.GeoDataFrame({"left_value": np.arange(rows, dtype=np.int64), "geometry": left})
    right = geopandas.GeoDataFrame({"right_value": np.arange(rows, dtype=np.int64), "geometry": right})
    return left, right


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark repo-owned spatial query assembly.")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--overlap-ratio", type=float, default=0.2)
    args = parser.parse_args()

    tree, query = build_query_inputs(args.rows, overlap_ratio=args.overlap_ratio)
    owned, flat = build_owned_spatial_index(tree)

    started = perf_counter()
    repo_indices = query_spatial_index(owned, flat, query, predicate="intersects", sort=False)
    repo_query_elapsed = perf_counter() - started

    sindex = geopandas.GeoSeries(tree).sindex
    started = perf_counter()
    shapely_indices = sindex._tree.query(query, predicate="intersects")
    shapely_query_elapsed = perf_counter() - started

    nearest_tree, nearest_query = build_nearest_inputs(min(args.rows, 50_000))
    tree_index = geopandas.GeoSeries(nearest_tree).sindex
    started = perf_counter()
    repo_nearest, _nearest_impl = nearest_spatial_index(
        nearest_tree,
        nearest_query,
        tree_query_nearest=tree_index._tree.query_nearest,
        return_all=True,
        max_distance=1.0,
        return_distance=False,
        exclusive=False,
    )
    repo_nearest_elapsed = perf_counter() - started

    started = perf_counter()
    repo_unbounded_nearest, _unbounded_nearest_impl = nearest_spatial_index(
        nearest_tree,
        nearest_query,
        tree_query_nearest=tree_index._tree.query_nearest,
        return_all=True,
        max_distance=None,
        return_distance=False,
        exclusive=False,
    )
    repo_unbounded_nearest_elapsed = perf_counter() - started

    started = perf_counter()
    shapely_unbounded_nearest = tree_index._tree.query_nearest(
        nearest_query,
        all_matches=True,
        return_distance=False,
        exclusive=False,
    )
    shapely_unbounded_nearest_elapsed = perf_counter() - started

    left_outer, right_outer = build_outer_join_frames(args.rows, overlap_ratio=args.overlap_ratio)
    geopandas.clear_dispatch_events()
    started = perf_counter()
    _outer_join_cold = geopandas.sjoin(left_outer, right_outer, how="outer", predicate="intersects")  # noqa: F841
    outer_join_cold_elapsed = perf_counter() - started
    cold_dispatch_events = geopandas.get_dispatch_events(clear=True)
    outer_impl = cold_dispatch_events[-1].implementation if cold_dispatch_events else "unknown"
    started = perf_counter()
    outer_join = geopandas.sjoin(left_outer, right_outer, how="outer", predicate="intersects")
    outer_join_elapsed = perf_counter() - started

    print(
        json.dumps(
            {
                "rows": args.rows,
                "overlap_ratio": args.overlap_ratio,
                "repo_query_pairs": int(repo_indices.shape[1]),
                "shapely_query_pairs": int(shapely_indices.shape[1]),
                "repo_query_elapsed_seconds": repo_query_elapsed,
                "shapely_query_elapsed_seconds": shapely_query_elapsed,
                "nearest_rows": int(len(nearest_query)),
                "repo_nearest_pairs": int(repo_nearest.shape[1]),
                "repo_nearest_elapsed_seconds": repo_nearest_elapsed,
                "repo_unbounded_nearest_pairs": int(repo_unbounded_nearest.shape[1]),
                "repo_unbounded_nearest_elapsed_seconds": repo_unbounded_nearest_elapsed,
                "shapely_unbounded_nearest_pairs": int(shapely_unbounded_nearest.shape[1]),
                "shapely_unbounded_nearest_elapsed_seconds": shapely_unbounded_nearest_elapsed,
                "outer_join_rows": int(len(outer_join)),
                "outer_join_cold_elapsed_seconds": outer_join_cold_elapsed,
                "outer_join_elapsed_seconds": outer_join_elapsed,
                "outer_join_dispatch_implementation": outer_impl,
                "outer_join_unmatched_left_rows": int(outer_join["index_right"].isna().sum()),
                "outer_join_unmatched_right_rows": int(outer_join["index_left"].isna().sum()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
