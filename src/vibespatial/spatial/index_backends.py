"""Operation capabilities for coexisting spatial-index execution backends.

Layouts are reusable state; accelerators such as point grids and segment BVHs
are strategies within those layouts. The public API chooses by operation and
reports the selected implementation through normal dispatch events.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SpatialIndexBackend(StrEnum):
    FLAT = "flat-morton"
    PACKED_STR = "packed-str"
    FIXED_K = "bounded-knn"
    HOST_STR = "strtree-host"


@dataclass(frozen=True)
class IndexBackendCapability:
    operations: tuple[str, ...]
    geometry_families: tuple[str, ...]
    device: bool
    exact_ties: bool


_ALL_FAMILIES = ("point", "linestring", "polygon", "multipoint", "multilinestring", "multipolygon")
INDEX_BACKEND_CAPABILITIES = {
    SpatialIndexBackend.FLAT: IndexBackendCapability(("query", "query_relation", "query_aggregate"), _ALL_FAMILIES, True, False),
    SpatialIndexBackend.PACKED_STR: IndexBackendCapability(("nearest",), _ALL_FAMILIES, True, True),
    SpatialIndexBackend.FIXED_K: IndexBackendCapability(("nearest_k",), _ALL_FAMILIES, True, False),
    SpatialIndexBackend.HOST_STR: IndexBackendCapability(("query", "nearest"), (*_ALL_FAMILIES, "geometrycollection"), False, True),
}


def nearest_backend(*, k: int, device: bool) -> SpatialIndexBackend:
    """Choose a nearest contract; individual backends validate family admission."""
    if not device:
        return SpatialIndexBackend.HOST_STR
    return SpatialIndexBackend.PACKED_STR if k == 1 else SpatialIndexBackend.FIXED_K
