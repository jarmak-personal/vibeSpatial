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
    "OVERLAY_BATCH_PIP_GPU_THRESHOLD",
    "OVERLAY_GPU_FAILURE_THRESHOLD",
    "OVERLAY_GPU_REMAINDER_THRESHOLD",
    "OVERLAY_GROUPED_BOX_GPU_THRESHOLD",
    "OVERLAY_PAIR_BATCH_THRESHOLD",
    "OVERLAY_UNION_ALL_GPU_THRESHOLD",
    "SEGMENT_TILE_SIZE",
    "SPATIAL_EPSILON",
]


# Shared fp64 geometric boundary / degeneracy tolerance used for
# coordinate equality, collinearity, zero-span guards, and boundary
# classification in the current GPU/CPU kernels.
SPATIAL_EPSILON = 1e-12

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

# Small grouped union-all reductions are cheaper through the direct CPU path.
OVERLAY_UNION_ALL_GPU_THRESHOLD = 50

# After this many consecutive GPU reduction failures, switch the rest of the
# reduction to CPU rather than repeatedly probing a likely-corrupted context.
OVERLAY_GPU_FAILURE_THRESHOLD = 3

# Overlay difference skips batching below this total pair count because the
# unbatched path is cheaper than extra gather/split overhead.
OVERLAY_PAIR_BATCH_THRESHOLD = 200_000

# Many-vs-one overlay remainder routing only re-enters the GPU for small
# residual batches; larger remainders stay on the CPU path intentionally.
OVERLAY_GPU_REMAINDER_THRESHOLD = 1_000
