from __future__ import annotations

from inspect import signature

from vibespatial.api.tools import overlay as overlay_api
from vibespatial.bench import profile_rails
from vibespatial.constructive import (
    clip_rect_kernels,
    linestring_kernels,
    polygon_kernels,
    union_all,
    validity_kernels,
)
from vibespatial.io.gpu_parse import indexing_kernels as gpu_parse_indexing_kernels
from vibespatial.kernels.constructive import (
    line_merge,
    segmented_union,
    shared_paths,
    shortest_line,
)
from vibespatial.kernels.core import spatial_query_source
from vibespatial.overlay import bypass, dissolve, gpu_kernels
from vibespatial.predicates import point_relations_kernels
from vibespatial.predicates import polygon_kernels as predicate_polygon_kernels
from vibespatial.runtime.config import (
    COARSE_BOUNDS_TILE_SIZE,
    OVERLAY_BATCH_PIP_GPU_THRESHOLD,
    OVERLAY_GPU_FAILURE_THRESHOLD,
    OVERLAY_GPU_REMAINDER_THRESHOLD,
    OVERLAY_GROUPED_BOX_GPU_THRESHOLD,
    OVERLAY_PAIR_BATCH_THRESHOLD,
    OVERLAY_UNION_ALL_GPU_THRESHOLD,
    SEGMENT_TILE_SIZE,
)
from vibespatial.spatial import (
    indexing,
    query_candidates,
    segment_distance_kernels,
    segment_primitives,
)
from vibespatial.spatial import indexing_kernels as spatial_indexing_kernels


def test_spatial_epsilon_is_centralized_in_kernel_sources() -> None:
    sources = (
        point_relations_kernels._SHARED_DEVICE_HELPERS,
        predicate_polygon_kernels._POLYGON_PREDICATES_KERNEL_SOURCE,
        gpu_kernels._OVERLAY_FACE_WALK_KERNEL_SOURCE,
        gpu_kernels._OVERLAY_FACE_LABEL_KERNEL_SOURCE,
        spatial_indexing_kernels._INDEXING_KERNEL_SOURCE,
        gpu_parse_indexing_kernels._HILBERT_KERNEL_SOURCE,
        spatial_query_source._SPATIAL_QUERY_KERNEL_SOURCE,
        segment_distance_kernels._SEGMENT_DISTANCE_KERNEL_SOURCE,
        validity_kernels._IS_SIMPLE_SEGMENTS_KERNEL_SOURCE,
        validity_kernels._HOLES_IN_SHELL_KERNEL_TEMPLATE,
        polygon_kernels._POLYGON_BUFFER_KERNEL_SOURCE,
        clip_rect_kernels._SUTHERLAND_HODGMAN_KERNEL_SOURCE,
        clip_rect_kernels._LIANG_BARSKY_KERNEL_SOURCE,
        linestring_kernels._LINESTRING_BUFFER_KERNEL_SOURCE,
        shared_paths._SHARED_PATHS_KERNEL_SOURCE,
        line_merge._LINE_MERGE_KERNEL_SOURCE,
        shortest_line._SHORTEST_LINE_KERNEL_SOURCE,
    )
    for source in sources:
        assert "1e-12" not in source
        assert "VS_SPATIAL_EPSILON" in source


def test_tile_size_defaults_are_centralized() -> None:
    assert signature(query_candidates._generate_distance_pairs).parameters["tile_size"].default == COARSE_BOUNDS_TILE_SIZE
    assert signature(indexing.generate_bounds_pairs).parameters["tile_size"].default == COARSE_BOUNDS_TILE_SIZE
    assert signature(indexing.benchmark_bounds_pairs).parameters["tile_size"].default == COARSE_BOUNDS_TILE_SIZE
    assert signature(indexing.FlatSpatialIndex.query).parameters["tile_size"].default == COARSE_BOUNDS_TILE_SIZE
    assert signature(profile_rails.profile_join_kernel).parameters["tile_size"].default == COARSE_BOUNDS_TILE_SIZE

    assert signature(indexing.generate_segment_mbr_pairs).parameters["tile_size"].default == SEGMENT_TILE_SIZE
    assert signature(indexing.benchmark_segment_filter).parameters["tile_size"].default == SEGMENT_TILE_SIZE
    assert signature(segment_primitives.generate_segment_candidates).parameters["tile_size"].default == SEGMENT_TILE_SIZE
    assert signature(segment_primitives.classify_segment_intersections).parameters["tile_size"].default == SEGMENT_TILE_SIZE
    assert signature(segment_primitives.benchmark_segment_intersections).parameters["tile_size"].default == SEGMENT_TILE_SIZE
    assert signature(profile_rails.profile_overlay_kernel).parameters["tile_size"].default == SEGMENT_TILE_SIZE


def test_overlay_thresholds_are_centralized() -> None:
    assert OVERLAY_BATCH_PIP_GPU_THRESHOLD == 100
    assert OVERLAY_GROUPED_BOX_GPU_THRESHOLD == 50_000
    assert OVERLAY_UNION_ALL_GPU_THRESHOLD == 50
    assert OVERLAY_GPU_FAILURE_THRESHOLD == 3
    assert OVERLAY_PAIR_BATCH_THRESHOLD == 200_000
    assert OVERLAY_GPU_REMAINDER_THRESHOLD == 1_000

    assert not hasattr(bypass, "_BATCH_PIP_GPU_THRESHOLD")
    assert not hasattr(dissolve, "_GROUPED_BOX_GPU_THRESHOLD")
    assert not hasattr(dissolve, "_UNION_ALL_GPU_THRESHOLD")
    assert not hasattr(union_all, "_GPU_FAILURE_THRESHOLD")
    assert not hasattr(segmented_union, "_GPU_FAILURE_THRESHOLD")
    assert not hasattr(overlay_api, "_PAIR_THRESHOLD")
    assert not hasattr(overlay_api, "_GPU_REMAINDER_THRESHOLD")
