"""Benchmark operation wrappers — importing this package triggers registration."""
from vibespatial.bench.operations import (
    constructive_ops,
    io_ops,
    overlay_ops,
    predicate_ops,
    spatial_ops,
)

__all__ = [
    "constructive_ops",
    "io_ops",
    "overlay_ops",
    "predicate_ops",
    "spatial_ops",
]
