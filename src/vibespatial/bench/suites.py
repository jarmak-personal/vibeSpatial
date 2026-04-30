"""Predefined benchmark suite definitions for vsbench CLI."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SuiteDefinition:
    """A named collection of operations, pipelines, and scales to run."""

    name: str
    description: str
    operations: tuple[str, ...]
    pipelines: tuple[str, ...]
    scales: tuple[int, ...]
    kernels: tuple[str, ...] = ()  # Tier 2 NVBench kernel benchmarks


SUITES: dict[str, SuiteDefinition] = {
    "smoke": SuiteDefinition(
        name="smoke",
        description="Quick sanity check (~30s, 1K rows)",
        operations=(
            "bounds",
            "clip-rect",
            "binary-predicates",
        ),
        pipelines=(
            "join-heavy",
            "relation-semijoin",
            "small-grouped-constructive-reduce",
            "constructive-output-native",
            "native-area-expression",
            "native-metadata-index",
            "constructive",
        ),
        scales=(1_000,),
    ),
    "ci": SuiteDefinition(
        name="ci",
        description="CI gate suite (~5min, 100K rows)",
        operations=(
            "bounds",
            "clip-rect",
            "binary-predicates",
            "spatial-query",
            "gpu-overlay",
            "io-arrow",
            "io-file",
        ),
        pipelines=(
            "join-heavy",
            "relation-semijoin",
            "small-grouped-constructive-reduce",
            "constructive-output-native",
            "constructive",
            "predicate-heavy",
            "zero-transfer",
        ),
        scales=(100_000,),
    ),
    "full": SuiteDefinition(
        name="full",
        description="Full benchmark suite (~30min, 100K + 1M rows)",
        operations=(
            "bounds",
            "clip-rect",
            "binary-predicates",
            "spatial-query",
            "gpu-overlay",
            "make-valid",
            "gpu-dissolve",
            "stroke-kernels",
            "io-arrow",
            "io-file",
        ),
        pipelines=(
            "join-heavy",
            "relation-semijoin",
            "small-grouped-constructive-reduce",
            "constructive-output-native",
            "constructive",
            "predicate-heavy",
            "predicate-heavy-geopandas",
            "zero-transfer",
            "vegetation-corridor",
            "vegetation-corridor-geopandas",
            "parcel-zoning",
            "parcel-zoning-geopandas",
            "flood-exposure",
            "flood-exposure-geopandas",
            "site-suitability",
            "site-suitability-geopandas",
            "provenance-rewrite",
        ),
        scales=(100_000, 1_000_000),
    ),
}
