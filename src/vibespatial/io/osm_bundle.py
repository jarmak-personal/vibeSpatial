from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vibespatial.api._native_results import (
    GeometryNativeResult,
    NativeAttributeTable,
    NativeReadProvenance,
    NativeTabularResult,
    _concat_native_tabular_results,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS

_OSM_IGNORE_COMMON = frozenset(
    {
        "created_by",
        "converted_by",
        "source",
        "time",
        "ele",
        "note",
        "todo",
        "fixme",
        "FIXME",
    }
)
_OSM_IGNORE_COMMON_PREFIXES = ("openGeoDB:",)


@dataclass(frozen=True)
class _OsmAttributeProfile:
    promoted_columns: tuple[str, ...]
    ignore_keys: frozenset[str] = frozenset()
    compute_z_order: bool = False


_OSM_PARTITION_ATTRIBUTE_PROFILES = {
    "node": _OsmAttributeProfile(
        promoted_columns=(
            "name",
            "barrier",
            "highway",
            "ref",
            "address",
            "is_in",
            "place",
            "man_made",
        ),
        ignore_keys=_OSM_IGNORE_COMMON | frozenset({"attribution"}),
    ),
    "way": _OsmAttributeProfile(
        promoted_columns=(
            "name",
            "type",
            "highway",
            "waterway",
            "aerialway",
            "barrier",
            "man_made",
            "railway",
            "z_order",
            "aeroway",
            "amenity",
            "admin_level",
            "boundary",
            "building",
            "craft",
            "geological",
            "historic",
            "land_area",
            "landuse",
            "leisure",
            "military",
            "natural",
            "office",
            "place",
            "shop",
            "sport",
            "tourism",
        ),
        ignore_keys=_OSM_IGNORE_COMMON | frozenset({"area"}),
        compute_z_order=True,
    ),
    "relation": _OsmAttributeProfile(
        promoted_columns=(
            "name",
            "type",
            "aeroway",
            "amenity",
            "admin_level",
            "barrier",
            "boundary",
            "building",
            "craft",
            "geological",
            "historic",
            "land_area",
            "landuse",
            "leisure",
            "man_made",
            "military",
            "natural",
            "office",
            "place",
            "shop",
            "sport",
            "tourism",
        ),
        ignore_keys=_OSM_IGNORE_COMMON | frozenset({"area"}),
    ),
}


def _escape_osm_tag_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _osm_tag_is_ignored(key: str, ignored_keys: frozenset[str]) -> bool:
    return key in ignored_keys or any(key.startswith(prefix) for prefix in _OSM_IGNORE_COMMON_PREFIXES)


def _compute_osm_z_order(tag_map: dict[str, str]) -> int | None:
    highway = tag_map.get("highway")
    railway = tag_map.get("railway")
    bridge = tag_map.get("bridge")
    tunnel = tag_map.get("tunnel")
    layer = tag_map.get("layer")

    rank = 0
    if highway:
        highway_ranks = {
            "motorway": 9,
            "trunk": 8,
            "primary": 7,
            "secondary": 6,
            "tertiary": 5,
            "unclassified": 4,
            "residential": 3,
            "service": 2,
            "track": 1,
            "path": 0,
        }
        rank = highway_ranks.get(highway, 0)
    elif railway:
        rank = 7

    if bridge in ("yes", "true", "1"):
        rank += 10
    if tunnel in ("yes", "true", "1"):
        rank -= 10
    if layer is not None:
        try:
            rank += int(layer) * 10
        except ValueError:
            pass
    return rank


def _stable_partition_columns(*, element: str, include_ids: bool, include_tags: bool) -> tuple[str, ...]:
    columns: list[str] = ["osm_element"]
    if include_ids:
        columns.append("osm_id")
    if include_tags:
        profile = _OSM_PARTITION_ATTRIBUTE_PROFILES[element]
        columns.extend(profile.promoted_columns)
        columns.append("other_tags")
    return tuple(columns)


def _build_osm_partition_frame(
    *,
    tags: list[dict[str, str]] | None,
    row_count: int,
    element: str,
    ids,
) -> pd.DataFrame:
    profile = _OSM_PARTITION_ATTRIBUTE_PROFILES[element]
    data: dict[str, np.ndarray] = {}

    osm_element = np.empty(row_count, dtype=object)
    osm_element[:] = element
    data["osm_element"] = osm_element

    if ids is not None:
        host_ids = ids.get() if hasattr(ids, "get") else np.asarray(ids)
        data["osm_id"] = np.asarray(host_ids, copy=False)

    if tags is None:
        return pd.DataFrame(data, index=pd.RangeIndex(row_count))

    promoted = {
        column_name: np.empty(row_count, dtype=object)
        for column_name in profile.promoted_columns
    }
    other_tags = np.empty(row_count, dtype=object)
    for values in promoted.values():
        values[:] = None
    other_tags[:] = None

    for index, tag_map in enumerate(tags):
        if not tag_map:
            continue
        extras: list[str] = []
        for key, value in tag_map.items():
            if key in promoted:
                promoted[key][index] = value
            else:
                if _osm_tag_is_ignored(key, profile.ignore_keys):
                    continue
                extras.append(
                    f'"{_escape_osm_tag_text(key)}"=>"{_escape_osm_tag_text(value)}"'
                )
        if profile.compute_z_order:
            z_order = _compute_osm_z_order(tag_map)
            if z_order is not None:
                promoted["z_order"][index] = z_order
        if extras:
            other_tags[index] = ",".join(extras)

    data.update(promoted)
    data["other_tags"] = other_tags
    return pd.DataFrame(data, index=pd.RangeIndex(row_count))


def _build_osm_partition_attributes(
    *,
    element: str,
    geometry,
    ids,
    tags: list[dict[str, str]] | None,
) -> NativeAttributeTable:
    row_count = int(geometry.row_count)
    if ids is None and tags is None:
        return NativeAttributeTable(dataframe=pd.DataFrame(index=pd.RangeIndex(row_count)))

    columns = _stable_partition_columns(
        element=element,
        include_ids=ids is not None,
        include_tags=tags is not None,
    )

    def _load() -> pd.DataFrame:
        return _build_osm_partition_frame(
            tags=tags,
            row_count=row_count,
            element=element,
            ids=ids,
        )

    return NativeAttributeTable.from_loader(
        _load,
        index_override=pd.RangeIndex(row_count),
        columns=columns,
    )


def _rebuild_native_result(payload: NativeTabularResult, attributes: NativeAttributeTable) -> NativeTabularResult:
    return NativeTabularResult(
        attributes=attributes,
        geometry=payload.geometry,
        geometry_name=payload.geometry_name,
        column_order=tuple([*attributes.columns, payload.geometry_name]),
        attrs=payload.attrs,
        secondary_geometry=payload.secondary_geometry,
        provenance=payload.provenance,
    )


def _select_attribute_columns(attributes: NativeAttributeTable, keep_columns: tuple[str, ...]) -> NativeAttributeTable:
    if attributes.loader is not None:
        parent = attributes

        def _load() -> pd.DataFrame:
            frame = parent.to_pandas(copy=False)
            return frame.loc[:, list(keep_columns)]

        return NativeAttributeTable.from_loader(
            _load,
            index_override=attributes.index,
            columns=keep_columns,
            to_pandas_kwargs=attributes.to_pandas_kwargs,
        )

    if attributes.arrow_table is not None:
        return NativeAttributeTable(
            arrow_table=attributes.arrow_table.select(list(keep_columns)),
            index_override=attributes.index,
            to_pandas_kwargs=attributes.to_pandas_kwargs,
        )

    frame = attributes.to_pandas(copy=False).loc[:, list(keep_columns)]
    return NativeAttributeTable(dataframe=frame)


def _drop_attribute_columns(payload: NativeTabularResult, drop_columns: tuple[str, ...]) -> NativeTabularResult:
    keep_columns = tuple(column for column in payload.attributes.columns if column not in set(drop_columns))
    return _rebuild_native_result(
        payload,
        _select_attribute_columns(payload.attributes, keep_columns),
    )


def _rename_attribute_columns(payload: NativeTabularResult, mapping: dict[str, str]) -> NativeTabularResult:
    if not mapping:
        return payload
    attributes = payload.attributes.rename_columns(mapping)
    return NativeTabularResult(
        attributes=attributes,
        geometry=payload.geometry,
        geometry_name=payload.geometry_name,
        column_order=tuple([mapping.get(name, name) for name in payload.column_order]),
        attrs=payload.attrs,
        secondary_geometry=payload.secondary_geometry,
        provenance=payload.provenance,
    )


def _subset_partition_by_way_families(
    partition: NativeTabularResult | None,
    *,
    families: tuple[GeometryFamily, ...],
) -> NativeTabularResult | None:
    if partition is None or partition.geometry.owned is None:
        return partition
    way_tags = np.asarray(partition.geometry.owned.tags, dtype=np.int8)
    family_tags = [FAMILY_TAGS[family] for family in families]
    keep_rows = np.flatnonzero(np.isin(way_tags, family_tags))
    if keep_rows.size == 0:
        return None
    return partition.take(keep_rows)


def _compat_partition_result(
    payload: NativeTabularResult | None,
    *,
    element: str,
    normalized_layer: str,
    n_types: int,
) -> NativeTabularResult | None:
    if payload is None:
        return None

    if normalized_layer == "all":
        if n_types > 1:
            return payload
        specific_id = {
            "node": "osm_node_id",
            "way": "osm_way_id",
            "relation": "osm_relation_id",
        }[element]
        return _drop_attribute_columns(
            _rename_attribute_columns(payload, {"osm_id": specific_id}),
            ("osm_element",),
        )

    if normalized_layer in {"points", "lines", "ways", "relations"}:
        return _drop_attribute_columns(payload, ("osm_element",))

    if normalized_layer == "multipolygons":
        if element == "way":
            payload = _rename_attribute_columns(payload, {"osm_id": "osm_way_id"})
        return _drop_attribute_columns(payload, ("osm_element",))

    return payload


@dataclass(frozen=True)
class OsmNativePartition:
    element: str
    result: NativeTabularResult

    @property
    def row_count(self) -> int:
        return int(self.result.geometry.row_count)


@dataclass(frozen=True)
class OsmNativeBundle:
    crs: Any
    points: OsmNativePartition | None = None
    ways: OsmNativePartition | None = None
    relations: OsmNativePartition | None = None
    source: str | None = None

    @property
    def partitions(self) -> tuple[OsmNativePartition, ...]:
        return tuple(
            partition
            for partition in (self.points, self.ways, self.relations)
            if partition is not None and partition.row_count > 0
        )

    def full_counts(self) -> tuple[int, int, int]:
        return (
            0 if self.points is None else self.points.row_count,
            0 if self.ways is None else self.ways.row_count,
            0 if self.relations is None else self.relations.row_count,
        )

    def supported_layers(self) -> tuple[str, ...]:
        return ("points", "lines", "ways", "multipolygons", "relations", "all")

    def to_native_tabular_result(
        self,
        *,
        layer: str = "all",
        compatibility: bool = True,
    ) -> NativeTabularResult | None:
        normalized_layer = "all" if layer is None else str(layer).strip().lower()
        if normalized_layer not in {"all", "points", "lines", "ways", "multipolygons", "relations"}:
            return None

        if normalized_layer == "points":
            payload = None if self.points is None else self.points.result
            return _compat_partition_result(
                payload,
                element="node",
                normalized_layer=normalized_layer,
                n_types=1,
            ) if compatibility else payload

        if normalized_layer in {"lines", "ways"}:
            payload = None if self.ways is None else self.ways.result
            if normalized_layer == "lines":
                payload = _subset_partition_by_way_families(
                    payload,
                    families=(GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING),
                )
            return _compat_partition_result(
                payload,
                element="way",
                normalized_layer=normalized_layer,
                n_types=1,
            ) if compatibility else payload

        if normalized_layer == "relations":
            payload = None if self.relations is None else self.relations.result
            return _compat_partition_result(
                payload,
                element="relation",
                normalized_layer=normalized_layer,
                n_types=1,
            ) if compatibility else payload

        if normalized_layer == "multipolygons":
            results: list[NativeTabularResult] = []
            way_payload = None if self.ways is None else self.ways.result
            way_payload = _subset_partition_by_way_families(
                way_payload,
                families=(GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON),
            )
            if way_payload is not None:
                results.append(
                    _compat_partition_result(
                        way_payload,
                        element="way",
                        normalized_layer=normalized_layer,
                        n_types=1,
                    ) if compatibility else way_payload
                )
            relation_payload = None if self.relations is None else self.relations.result
            if relation_payload is not None:
                results.append(
                    _compat_partition_result(
                        relation_payload,
                        element="relation",
                        normalized_layer=normalized_layer,
                        n_types=1,
                    ) if compatibility else relation_payload
                )
            if not results:
                return None
            if len(results) == 1:
                return results[0]
            return _concat_native_tabular_results(
                results,
                geometry_name="geometry",
                crs=self.crs,
            )

        results: list[NativeTabularResult] = []
        if self.points is not None:
            results.append(
                _compat_partition_result(
                    self.points.result,
                    element="node",
                    normalized_layer=normalized_layer,
                    n_types=len(self.partitions),
                ) if compatibility else self.points.result
            )
        if self.ways is not None:
            results.append(
                _compat_partition_result(
                    self.ways.result,
                    element="way",
                    normalized_layer=normalized_layer,
                    n_types=len(self.partitions),
                ) if compatibility else self.ways.result
            )
        if self.relations is not None:
            results.append(
                _compat_partition_result(
                    self.relations.result,
                    element="relation",
                    normalized_layer=normalized_layer,
                    n_types=len(self.partitions),
                ) if compatibility else self.relations.result
            )
        if not results:
            return None
        if len(results) == 1:
            return results[0]
        return _concat_native_tabular_results(
            results,
            geometry_name="geometry",
            crs=self.crs,
        )


def _partition_provenance(*, element: str, source: str | None) -> NativeReadProvenance:
    return NativeReadProvenance(
        surface="vibespatial.io.osm_pbf",
        format_name="OSM PBF",
        source=source,
        backend="osm_pbf_gpu_hybrid_parser",
        planner_strategy=f"osm_partition:{element}",
    )


def _partition_from_group(
    *,
    element: str,
    geometry,
    ids,
    tags: list[dict[str, str]] | None,
    crs,
    source: str | None,
) -> OsmNativePartition | None:
    if geometry is None or int(geometry.row_count) == 0:
        return None
    attributes = _build_osm_partition_attributes(
        element=element,
        geometry=geometry,
        ids=ids,
        tags=tags,
    )
    result = NativeTabularResult(
        attributes=attributes,
        geometry=GeometryNativeResult.from_owned(geometry, crs=crs),
        geometry_name="geometry",
        column_order=tuple([*attributes.columns, "geometry"]),
        provenance=_partition_provenance(element=element, source=source),
    )
    return OsmNativePartition(element=element, result=result)


def build_osm_native_bundle(
    osm_result,
    *,
    crs,
    source: str | None = None,
) -> OsmNativeBundle:
    return OsmNativeBundle(
        crs=crs,
        points=_partition_from_group(
            element="node",
            geometry=osm_result.nodes,
            ids=osm_result.node_ids,
            tags=osm_result.node_tags,
            crs=crs,
            source=source,
        ),
        ways=_partition_from_group(
            element="way",
            geometry=osm_result.ways,
            ids=osm_result.way_ids,
            tags=osm_result.way_tags,
            crs=crs,
            source=source,
        ),
        relations=_partition_from_group(
            element="relation",
            geometry=osm_result.relations,
            ids=osm_result.relation_ids,
            tags=osm_result.relation_tags,
            crs=crs,
            source=source,
        ),
        source=source,
    )
