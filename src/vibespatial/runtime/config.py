from __future__ import annotations

"""Centralized numerical tolerances and runtime tuning defaults.

These values were previously duplicated across constructive, overlay,
spatial, and benchmark modules. Keep semantically distinct thresholds as
separate names even when they currently share the same numeric value.
"""

__all__ = [
    "BOUNDS_SPAN_EPSILON",
    "COARSE_BOUNDS_TILE_SIZE",
    "GEOM_EQUALS_DEFAULT_TOLERANCE",
    "LINESTRING_TWO_POINT_BUFFER_GPU_THRESHOLD",
    "OVERLAY_BATCH_PIP_GPU_THRESHOLD",
    "OVERLAY_GPU_FAILURE_THRESHOLD",
    "OVERLAY_GPU_REMAINDER_THRESHOLD",
    "OVERLAY_GROUPED_BOX_GPU_THRESHOLD",
    "OVERLAY_GROUPED_COVERAGE_EDGE_THRESHOLD",
    "OVERLAY_PAIR_BATCH_THRESHOLD",
    "OVERLAY_UNION_ALL_GPU_THRESHOLD",
    "SEGMENT_TILE_SIZE",
    "SPATIAL_EPSILON",
]


# Shared fp64 geometric boundary / degeneracy tolerance used for
# coordinate equality, collinearity, zero-span guards, and boundary
# classification in the current GPU/CPU kernels.
SPATIAL_EPSILON = 1e-12

# The generic linestring buffer crossover is intentionally conservative because
# multi-vertex linestring buffering is still more host-competitive at small
# sizes. Simple two-point segment workloads are materially cheaper on the GPU
# and need their own lower threshold to keep grid/network pipelines off the CPU.
LINESTRING_TWO_POINT_BUFFER_GPU_THRESHOLD = 512

# Span guards use the same numeric floor today but keep a distinct name
# so Morton/grid normalization can diverge later without code search.
BOUNDS_SPAN_EPSILON = SPATIAL_EPSILON

# Public geom_equals defaults to the same tolerance used by the owned
# equality kernels and normalization-based topology comparison.
GEOM_EQUALS_DEFAULT_TOLERANCE = SPATIAL_EPSILON

# Coarse bounds filters are memory-bound and benefit from smaller CPU tile
# working sets.
COARSE_BOUNDS_TILE_SIZE = 256

# Segment candidate generation/classification benefits from a larger tile
# because per-segment payload is compact and the CPU fallback amortizes loop
# overhead better at this size.
SEGMENT_TILE_SIZE = 512

# Batch point-in-polygon only pays off once there are enough candidate
# pairs to amortize kernel launch and gather/scatter overhead.
OVERLAY_BATCH_PIP_GPU_THRESHOLD = 100

# Coverage-style grouped box union stays on the CPU below this row count;
# above it the grouped GPU box path changes the runtime shape materially.
OVERLAY_GROUPED_BOX_GPU_THRESHOLD = 50_000

# Shared-edge grouped coverage dissolve has a larger fixed setup cost than the
# box fast path and only changes throughput once the grouped workload is large.
OVERLAY_GROUPED_COVERAGE_EDGE_THRESHOLD = 50_000

# Small grouped union-all reductions are cheaper through the direct CPU path.
OVERLAY_UNION_ALL_GPU_THRESHOLD = 50

# After this many consecutive GPU reduction failures, switch the rest of the
# reduction to CPU rather than repeatedly probing a likely-corrupted context.
OVERLAY_GPU_FAILURE_THRESHOLD = 3

# Overlay difference skips batching below this total pair count because the
# unbatched path is cheaper than extra gather/split overhead.
OVERLAY_PAIR_BATCH_THRESHOLD = 200_000

# Host-only installs still use this crossover for many-vs-one exact overlay
# remainders. GPU-enabled runtimes now prefer the device remainder helpers
# regardless of size and only use the host path after an explicit GPU failure.
OVERLAY_GPU_REMAINDER_THRESHOLD = 1_000
