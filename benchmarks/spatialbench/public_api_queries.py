#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.
"""Streaming SpatialBench implementations built exclusively from public APIs.

The original GeoPandas implementation intentionally mirrors the SQL literally.
That is useful as a readable reference, but it gives an in-memory dataframe the
physical plan of a database query and cannot execute at SF100.  This module
contains the shared, hand-optimized plan used by both GeoPandas and
vibeSpatial.  Backend wrappers inject either public ``geopandas`` or public
``vibespatial`` as ``gpd``; no private backend symbols are imported here.

The important physical rules are:

* project columns before reading Parquet;
* stream fact-table record batches instead of materializing the trip table;
* keep only bounded top-k state or mergeable aggregates between batches;
* decode WKB only after non-spatial filters whenever SQL semantics permit it;
* use public GeoSeries, GeoDataFrame, spatial-index, and spatial-join methods.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pandas import DataFrame
from shapely.geometry import Point, Polygon

CRS = None  # SpatialBench SQL uses planar coordinate units, not geodesic CRS rules.
DEFAULT_BATCH_ROWS = 2_000_000


def _batch_rows() -> int:
    return int(os.environ.get("SPATIALBENCH_BATCH_ROWS", DEFAULT_BATCH_ROWS))


def _parquet_files(path: str) -> list[Path]:
    source = Path(path)
    if source.is_file():
        return [source]

    def numeric_suffix(file: Path) -> tuple[int, str]:
        try:
            return int(file.stem.rsplit(".", 1)[-1]), file.name
        except ValueError:
            return 0, file.name

    return sorted(source.glob("*.parquet"), key=numeric_suffix)


def _batches(path: str, columns: list[str]) -> Iterator[pa.RecordBatch]:
    for file in _parquet_files(path):
        yield from pq.ParquetFile(file).iter_batches(
            batch_size=_batch_rows(), columns=columns, use_threads=True
        )


def _table(path: str, columns: list[str]) -> pa.Table:
    tables = [pq.read_table(file, columns=columns) for file in _parquet_files(path)]
    if not tables:
        return pa.table({name: [] for name in columns})
    return pa.concat_tables(tables)


def _column(batch: pa.RecordBatch | pa.Table, name: str) -> pa.Array | pa.ChunkedArray:
    return batch.column(batch.schema.get_field_index(name))


def _float_array(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return pc.cast(values, pa.float64()).to_numpy(zero_copy_only=False)


def _int_array(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return pc.cast(values, pa.int64()).to_numpy(zero_copy_only=False)


def _frame(batch: pa.RecordBatch | pa.Table, columns: list[str]) -> DataFrame:
    return batch.select(columns).to_pandas()


def _geometry(gpd: Any, values: Any):
    point_xy = _fixed_point_wkb_xy(values)
    if point_xy is not None:
        x, y = point_xy
        return gpd.GeoSeries(gpd.points_from_xy(x, y, crs=CRS), crs=CRS)
    return gpd.GeoSeries.from_wkb(values, crs=CRS)


def _fixed_point_wkb_xy(values: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Return zero-copy coordinate views for homogeneous little-endian Point WKB."""
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    if not isinstance(values, (pa.BinaryArray, pa.LargeBinaryArray)):
        return None
    if len(values) == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty
    if values.null_count:
        return None

    offset_dtype = np.dtype("<i8" if pa.types.is_large_binary(values.type) else "<i4")
    offsets_buffer = values.buffers()[1]
    payload_buffer = values.buffers()[2]
    if offsets_buffer is None or payload_buffer is None:
        return None
    offsets = np.frombuffer(
        offsets_buffer,
        dtype=offset_dtype,
        count=len(values) + 1,
        offset=values.offset * offset_dtype.itemsize,
    )
    if not np.all(np.diff(offsets) == 21):
        return None

    start = int(offsets[0])
    byte_orders = np.ndarray(
        len(values),
        dtype=np.uint8,
        buffer=payload_buffer,
        offset=start,
        strides=(21,),
    )
    type_ids = np.ndarray(
        len(values),
        dtype="<u4",
        buffer=payload_buffer,
        offset=start + 1,
        strides=(21,),
    )
    if not np.all(byte_orders == 1) or not np.all(type_ids == 1):
        return None
    return (
        np.ndarray(
            len(values),
            dtype="<f8",
            buffer=payload_buffer,
            offset=start + 5,
            strides=(21,),
        ),
        np.ndarray(
            len(values),
            dtype="<f8",
            buffer=payload_buffer,
            offset=start + 13,
            strides=(21,),
        ),
    )


def _geometry_frame(
    gpd: Any,
    batch: pa.RecordBatch | pa.Table,
    *,
    wkb_column: str,
    geometry_name: str,
    columns: list[str],
):
    frame = _frame(batch, columns)
    geometry = _geometry(gpd, _column(batch, wkb_column))
    frame[geometry_name] = geometry
    return gpd.GeoDataFrame(frame, geometry=geometry_name, crs=CRS)


def _topk(
    frames: list[DataFrame],
    n: int,
    columns: list[str],
    ascending: list[bool],
) -> DataFrame:
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(columns, ascending=ascending, na_position="last")
        .head(n)
        .reset_index(drop=True)
    )


def _combine_sums(parts: list[DataFrame], keys: list[str]) -> DataFrame:
    if not parts:
        return pd.DataFrame()
    value_columns = [c for c in parts[0].columns if c not in keys]
    return (
        pd.concat(parts, ignore_index=True)
        .groupby(keys, as_index=False, sort=False)[value_columns]
        .sum()
    )


def _right_index_column(joined: DataFrame) -> str:
    candidates = [c for c in joined.columns if str(c).startswith("index_")]
    if "index_right" in candidates:
        return "index_right"
    if not candidates:
        raise KeyError("spatial join did not return a right index column")
    return candidates[0]


class PublicApiQueries:
    """Optimized query suite parameterized by a GeoPandas-compatible module."""

    def __init__(
        self,
        gpd: Any,
        *,
        dissolve_method: str = "unary",
        distance_pair_rows: int | None = None,
        scan_batch_rows: int | None = None,
    ):
        self.gpd = gpd
        self.dissolve_method = dissolve_method
        self.distance_pair_rows = distance_pair_rows
        self.scan_batch_rows = scan_batch_rows

    def q1(self, data_paths: dict[str, str]) -> DataFrame:
        center = Point(-111.7610, 34.8697)
        candidates: list[DataFrame] = []
        columns = ["t_tripkey", "t_pickuptime", "t_pickuploc"]
        for batch in _batches(data_paths["trip"], columns):
            points = _geometry(self.gpd, _column(batch, "t_pickuploc"))
            distance = points.distance(center)
            mask = distance.notna().to_numpy() & (distance.to_numpy() <= 0.45)
            if not mask.any():
                continue
            selected = batch.filter(pa.array(mask)).to_pandas()
            selected["distance_to_center"] = distance.to_numpy()[mask]
            candidates.append(
                selected.nsmallest(100, ["distance_to_center", "t_tripkey"])
            )

        result = _topk(
            candidates,
            100,
            ["distance_to_center", "t_tripkey"],
            [True, True],
        )
        if result.empty:
            return pd.DataFrame(
                columns=[
                    "t_tripkey",
                    "pickup_lon",
                    "pickup_lat",
                    "t_pickuptime",
                    "distance_to_center",
                ]
            )
        points = _geometry(self.gpd, pa.array(result.pop("t_pickuploc")))
        result["pickup_lon"] = points.x.to_numpy()
        result["pickup_lat"] = points.y.to_numpy()
        return result[
            [
                "t_tripkey",
                "pickup_lon",
                "pickup_lat",
                "t_pickuptime",
                "distance_to_center",
            ]
        ]

    def q2(self, data_paths: dict[str, str]) -> DataFrame:
        target_wkb = None
        for batch in _batches(data_paths["zone"], ["z_name", "z_boundary"]):
            names = _column(batch, "z_name")
            matches = pc.equal(names, "Coconino County")
            if pc.any(matches).as_py():
                match_index = int(pc.index(matches, True).as_py())
                target_wkb = _column(batch, "z_boundary").slice(match_index, 1)
                break
        if target_wkb is None:
            return pd.DataFrame({"trip_count_in_coconino_county": [0]})
        polygon = _geometry(self.gpd, target_wkb).iloc[0]
        count = 0
        for batch in _batches(data_paths["trip"], ["t_pickuploc"]):
            points = _geometry(self.gpd, _column(batch, "t_pickuploc"))
            count += int(points.intersects(polygon).sum())
        return pd.DataFrame({"trip_count_in_coconino_county": [count]})

    def q3(self, data_paths: dict[str, str]) -> DataFrame:
        polygon = Polygon(
            [
                (-111.9060, 34.7347),
                (-111.6160, 34.7347),
                (-111.6160, 35.0047),
                (-111.9060, 35.0047),
                (-111.9060, 34.7347),
            ]
        )
        columns = [
            "t_tripkey",
            "t_pickuptime",
            "t_dropofftime",
            "t_distance",
            "t_fare",
            "t_pickuploc",
        ]
        parts: list[DataFrame] = []
        for batch in _batches(data_paths["trip"], columns):
            points = _geometry(self.gpd, _column(batch, "t_pickuploc"))
            mask = points.distance(polygon).to_numpy() <= 0.045
            if not mask.any():
                continue
            selected = batch.filter(pa.array(mask))
            pickup = _column(selected, "t_pickuptime").to_pandas()
            dropoff = _column(selected, "t_dropofftime").to_pandas()
            frame = pd.DataFrame(
                {
                    "pickup_month": pickup.dt.to_period("M").dt.to_timestamp(),
                    "_distance": _float_array(_column(selected, "t_distance")),
                    "_fare": _float_array(_column(selected, "t_fare")),
                    "_duration": (dropoff - pickup).dt.total_seconds(),
                }
            )
            part = frame.groupby("pickup_month", as_index=False).agg(
                total_trips=("_distance", "size"),
                _distance_sum=("_distance", "sum"),
                _duration_sum=("_duration", "sum"),
                _fare_sum=("_fare", "sum"),
            )
            parts.append(part)

        totals = _combine_sums(parts, ["pickup_month"])
        if totals.empty:
            return pd.DataFrame(
                columns=[
                    "pickup_month",
                    "total_trips",
                    "avg_distance",
                    "avg_duration",
                    "avg_fare",
                ]
            )
        totals["avg_distance"] = totals.pop("_distance_sum") / totals["total_trips"]
        totals["avg_duration"] = totals.pop("_duration_sum") / totals["total_trips"]
        totals["avg_fare"] = totals.pop("_fare_sum") / totals["total_trips"]
        return totals.sort_values("pickup_month").reset_index(drop=True)

    def q4(self, data_paths: dict[str, str]) -> DataFrame:
        top_parts: list[DataFrame] = []
        columns = ["t_tripkey", "t_tip", "t_pickuploc"]
        for batch in _batches(data_paths["trip"], columns):
            frame = pd.DataFrame(
                {
                    "t_tripkey": _int_array(_column(batch, "t_tripkey")),
                    "t_tip": _float_array(_column(batch, "t_tip")),
                    "t_pickuploc": _column(batch, "t_pickuploc").to_pylist(),
                }
            )
            top_parts.append(
                frame.sort_values(
                    ["t_tip", "t_tripkey"], ascending=[False, True]
                ).head(1000)
            )
        top = _topk(top_parts, 1000, ["t_tip", "t_tripkey"], [False, True])
        top["pickup_geom"] = _geometry(self.gpd, pa.array(top.pop("t_pickuploc")))
        top_gdf = self.gpd.GeoDataFrame(top, geometry="pickup_geom", crs=CRS)

        joined_parts: list[DataFrame] = []
        zone_columns = ["z_zonekey", "z_name", "z_boundary"]
        for batch in _batches(data_paths["zone"], zone_columns):
            zones = _geometry_frame(
                self.gpd,
                batch,
                wkb_column="z_boundary",
                geometry_name="zone_geom",
                columns=["z_zonekey", "z_name"],
            )
            joined = self.gpd.sjoin(top_gdf, zones, how="inner", predicate="within")
            if not joined.empty:
                joined_parts.append(joined[["z_zonekey", "z_name"]])
        if not joined_parts:
            return pd.DataFrame(columns=["z_zonekey", "z_name", "trip_count"])
        return (
            pd.concat(joined_parts, ignore_index=True)
            .groupby(["z_zonekey", "z_name"], as_index=False)
            .size()
            .rename(columns={"size": "trip_count"})
            .sort_values(["trip_count", "z_zonekey"], ascending=[False, True])
            .reset_index(drop=True)
        )

    def q5(self, data_paths: dict[str, str]) -> DataFrame:
        count_parts: list[DataFrame] = []
        count_columns = ["t_custkey", "t_pickuptime"]
        for batch in _batches(data_paths["trip"], count_columns):
            frame = batch.to_pandas()
            frame["pickup_month"] = (
                frame["t_pickuptime"].dt.to_period("M").dt.to_timestamp()
            )
            count_parts.append(
                frame.groupby(["t_custkey", "pickup_month"], as_index=False)
                .size()
                .rename(columns={"size": "dropoff_count"})
            )
        counts = _combine_sums(count_parts, ["t_custkey", "pickup_month"])
        eligible = counts[counts["dropoff_count"] > 5].copy()
        if eligible.empty:
            return pd.DataFrame(
                columns=[
                    "c_custkey",
                    "customer_name",
                    "pickup_month",
                    "monthly_travel_hull_area",
                    "dropoff_count",
                ]
            )
        eligible_index = pd.MultiIndex.from_frame(
            eligible[["t_custkey", "pickup_month"]]
        )

        selected_parts = []
        columns = ["t_custkey", "t_pickuptime", "t_dropoffloc"]
        for batch in _batches(data_paths["trip"], columns):
            attrs = _frame(batch, ["t_custkey", "t_pickuptime"])
            attrs["pickup_month"] = (
                attrs["t_pickuptime"].dt.to_period("M").dt.to_timestamp()
            )
            keys = pd.MultiIndex.from_frame(attrs[["t_custkey", "pickup_month"]])
            mask = keys.isin(eligible_index)
            if not mask.any():
                continue
            selected = batch.filter(pa.array(mask))
            part = _frame(selected, ["t_custkey", "t_pickuptime"])
            part["pickup_month"] = (
                part.pop("t_pickuptime").dt.to_period("M").dt.to_timestamp()
            )
            part["dropoff_geom"] = _geometry(
                self.gpd, _column(selected, "t_dropoffloc")
            )
            selected_parts.append(
                self.gpd.GeoDataFrame(part, geometry="dropoff_geom", crs=CRS)
            )

        all_points = self.gpd.GeoDataFrame(
            pd.concat(selected_parts, ignore_index=True),
            geometry="dropoff_geom",
            crs=CRS,
        )
        hulls = (
            all_points.dissolve(
                by=["t_custkey", "pickup_month"],
                method=self.dissolve_method,
            )
            .geometry.convex_hull
            .area.rename("monthly_travel_hull_area")
            .reset_index()
        )
        result = eligible.merge(hulls, on=["t_custkey", "pickup_month"])
        customer = _table(data_paths["customer"], ["c_custkey", "c_name"]).to_pandas()
        result = result.merge(
            customer,
            left_on="t_custkey",
            right_on="c_custkey",
            how="inner",
        )
        return (
            result.sort_values(
                ["monthly_travel_hull_area", "c_custkey", "pickup_month"],
                ascending=[False, True, True],
            )[
                [
                    "c_custkey",
                    "c_name",
                    "pickup_month",
                    "monthly_travel_hull_area",
                    "dropoff_count",
                ]
            ]
            .rename(columns={"c_name": "customer_name"})
            .head(100)
            .reset_index(drop=True)
        )

    def _candidate_zones_for_q6(self, data_paths: dict[str, str]):
        bbox = Polygon(
            [
                (-112.2110, 34.4197),
                (-111.3110, 34.4197),
                (-111.3110, 35.3197),
                (-112.2110, 35.3197),
                (-112.2110, 34.4197),
            ]
        )
        parts = []
        columns = ["z_zonekey", "z_name", "z_boundary"]
        for batch in _batches(data_paths["zone"], columns):
            zones = _geometry_frame(
                self.gpd,
                batch,
                wkb_column="z_boundary",
                geometry_name="zone_geom",
                columns=["z_zonekey", "z_name"],
            )
            mask = zones.geometry.notna() & zones.geometry.intersects(bbox)
            if mask.any():
                parts.append(zones.loc[mask, ["z_zonekey", "z_name", "zone_geom"]])
        return self.gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True), geometry="zone_geom", crs=CRS
        )

    def q6(self, data_paths: dict[str, str]) -> DataFrame:
        zones = self._candidate_zones_for_q6(data_paths)
        parts: list[DataFrame] = []
        columns = [
            "t_tripkey",
            "t_pickuploc",
            "t_pickuptime",
            "t_dropofftime",
            "t_distance",
        ]
        for batch in _batches(data_paths["trip"], columns):
            pickups = _geometry_frame(
                self.gpd,
                batch,
                wkb_column="t_pickuploc",
                geometry_name="pickup_geom",
                columns=["t_tripkey", "t_pickuptime", "t_dropofftime", "t_distance"],
            )
            pickups["t_distance"] = pickups["t_distance"].astype(float)
            joined = self.gpd.sjoin(pickups, zones, how="inner", predicate="within")
            if joined.empty:
                continue
            joined["_duration"] = (
                joined["t_dropofftime"] - joined["t_pickuptime"]
            ).dt.total_seconds()
            parts.append(
                joined.groupby(["z_zonekey", "z_name"], as_index=False).agg(
                    total_pickups=("t_tripkey", "count"),
                    _distance_sum=("t_distance", "sum"),
                    _duration_sum=("_duration", "sum"),
                )
            )
        result = _combine_sums(parts, ["z_zonekey", "z_name"])
        result["avg_distance"] = result.pop("_distance_sum") / result["total_pickups"]
        result["avg_duration"] = result.pop("_duration_sum") / result["total_pickups"]
        return result.sort_values(
            ["total_pickups", "z_zonekey"], ascending=[False, True]
        ).reset_index(drop=True)

    def q7(self, data_paths: dict[str, str]) -> DataFrame:
        parts: list[DataFrame] = []
        columns = ["t_tripkey", "t_distance", "t_pickuploc", "t_dropoffloc"]
        for batch in _batches(data_paths["trip"], columns):
            pickup = _geometry(self.gpd, _column(batch, "t_pickuploc"))
            dropoff = _geometry(self.gpd, _column(batch, "t_dropoffloc"))
            line_distance = pickup.distance(dropoff, align=False).to_numpy() / 0.000009
            reported = _float_array(_column(batch, "t_distance"))
            ratio = np.divide(
                reported,
                line_distance,
                out=np.full_like(reported, np.nan),
                where=line_distance != 0.0,
            )
            frame = pd.DataFrame(
                {
                    "t_tripkey": _int_array(_column(batch, "t_tripkey")),
                    "reported_distance_m": reported,
                    "line_distance_m": line_distance,
                    "detour_ratio": ratio,
                }
            )
            parts.append(
                frame.sort_values(
                    ["detour_ratio", "reported_distance_m", "t_tripkey"],
                    ascending=[False, False, True],
                    na_position="last",
                ).head(100)
            )
        return _topk(
            parts,
            100,
            ["detour_ratio", "reported_distance_m", "t_tripkey"],
            [False, False, True],
        )

    def _buildings(self, data_paths: dict[str, str], include_name: bool = False):
        columns = ["b_buildingkey", "b_boundary"]
        attrs = ["b_buildingkey"]
        if include_name:
            columns.insert(1, "b_name")
            attrs.append("b_name")
        table = _table(data_paths["building"], columns)
        return _geometry_frame(
            self.gpd,
            table,
            wkb_column="b_boundary",
            geometry_name="boundary_geom",
            columns=attrs,
        )

    def q8(self, data_paths: dict[str, str]) -> DataFrame:
        buildings = self._buildings(data_paths, include_name=True)
        nearby_counts = np.zeros(len(buildings), dtype=np.int64)
        for batch in _batches(data_paths["trip"], ["t_pickuploc"]):
            pickups = _geometry_frame(
                self.gpd,
                batch,
                wkb_column="t_pickuploc",
                geometry_name="pickup_geom",
                columns=[],
            )
            pairs = buildings.sindex.query(
                pickups.geometry,
                predicate="dwithin",
                distance=0.0045,
                sort=False,
            )
            if pairs.shape[1]:
                nearby_counts += np.bincount(
                    np.asarray(pairs[1], dtype=np.int64),
                    minlength=len(buildings),
                )
        keep = nearby_counts != 0
        result = buildings.loc[keep, ["b_buildingkey", "b_name"]].reset_index(
            drop=True
        )
        result["nearby_pickup_count"] = nearby_counts[keep]
        return (
            result.sort_values(
                ["nearby_pickup_count", "b_buildingkey"],
                ascending=[False, True],
            )
            .head(100)
            .reset_index(drop=True)
        )

    def q9(self, data_paths: dict[str, str]) -> DataFrame:
        buildings = self._buildings(data_paths).rename(
            columns={"b_buildingkey": "building_key"}
        )
        pairs = self.gpd.sjoin(
            buildings, buildings, how="inner", predicate="intersects"
        )
        right_index = _right_index_column(pairs)
        left_key = next(
            c
            for c in ("building_key_left", "building_key_1", "building_key")
            if c in pairs.columns
        )
        right_key = next(
            (c for c in ("building_key_right", "building_key_2") if c in pairs),
            None,
        )
        if right_key is None:
            pairs["_right_key"] = buildings.iloc[
                pairs[right_index].to_numpy(dtype=np.int64)
            ]["building_key"].to_numpy()
            right_key = "_right_key"
        pairs = pairs.rename(
            columns={left_key: "building_1", right_key: "building_2"}
        )
        pairs = pairs[pairs["building_1"] < pairs["building_2"]].copy()
        left_geom = pairs.geometry.reset_index(drop=True)
        right_geom = buildings.geometry.iloc[
            pairs[right_index].to_numpy(dtype=np.int64)
        ].reset_index(drop=True)
        area1 = left_geom.area.to_numpy()
        area2 = right_geom.area.to_numpy()
        overlap = left_geom.intersection(right_geom, align=False).area.to_numpy()
        union = area1 + area2 - overlap
        iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union != 0)
        iou[(union == 0) & (overlap > 0)] = 1.0
        result = pd.DataFrame(
            {
                "building_1": pairs["building_1"].to_numpy(),
                "building_2": pairs["building_2"].to_numpy(),
                "area1": area1,
                "area2": area2,
                "overlap_area": overlap,
                "iou": iou,
            }
        )
        return (
            result.sort_values(
                ["iou", "building_1", "building_2"],
                ascending=[False, True, True],
            )
            .head(100)
            .reset_index(drop=True)
        )

    def _zones(self, data_paths: dict[str, str], name_column: str = "z_name"):
        table = _table(data_paths["zone"], ["z_zonekey", "z_name", "z_boundary"])
        zones = _geometry_frame(
            self.gpd,
            table,
            wkb_column="z_boundary",
            geometry_name="zone_geom",
            columns=["z_zonekey", "z_name"],
        )
        if name_column != "z_name":
            zones = zones.rename(columns={"z_name": name_column})
        return zones

    def q10(self, data_paths: dict[str, str]) -> DataFrame:
        zones = self._zones(data_paths)
        parts: list[DataFrame] = []
        columns = [
            "t_tripkey",
            "t_pickuploc",
            "t_pickuptime",
            "t_dropofftime",
            "t_distance",
        ]
        for batch in _batches(data_paths["trip"], columns):
            pickups = _geometry_frame(
                self.gpd,
                batch,
                wkb_column="t_pickuploc",
                geometry_name="pickup_geom",
                columns=["t_tripkey", "t_pickuptime", "t_dropofftime", "t_distance"],
            )
            pickups["t_distance"] = pickups["t_distance"].astype(float)
            joined = self.gpd.sjoin(pickups, zones, how="inner", predicate="within")
            if joined.empty:
                continue
            joined["_duration"] = (
                joined["t_dropofftime"] - joined["t_pickuptime"]
            ).dt.total_seconds()
            parts.append(
                joined.groupby(["z_zonekey", "z_name"], as_index=False).agg(
                    num_trips=("t_tripkey", "count"),
                    _distance_sum=("t_distance", "sum"),
                    _duration_sum=("_duration", "sum"),
                )
            )
        totals = _combine_sums(parts, ["z_zonekey", "z_name"])
        result = zones[["z_zonekey", "z_name"]].merge(
            totals, on=["z_zonekey", "z_name"], how="left"
        )
        result["avg_duration"] = result["_duration_sum"] / result["num_trips"]
        result["avg_distance"] = result["_distance_sum"] / result["num_trips"]
        result["num_trips"] = result["num_trips"].fillna(0).astype(np.int64)
        return (
            result.sort_values(
                ["avg_duration", "z_zonekey"],
                ascending=[False, True],
                na_position="last",
            )
            .rename(columns={"z_name": "pickup_zone"})[
                [
                    "z_zonekey",
                    "pickup_zone",
                    "avg_duration",
                    "avg_distance",
                    "num_trips",
                ]
            ]
            .head(100)
            .reset_index(drop=True)
        )

    def q11(self, data_paths: dict[str, str]) -> DataFrame:
        zones = self._zones(data_paths)
        count = 0
        columns = ["t_tripkey", "t_pickuploc", "t_dropoffloc"]
        for batch in _batches(data_paths["trip"], columns):
            attrs = _frame(batch, ["t_tripkey"])
            pickup = attrs.copy()
            pickup["geometry"] = _geometry(
                self.gpd, _column(batch, "t_pickuploc")
            )
            dropoff = attrs.copy()
            dropoff["geometry"] = _geometry(
                self.gpd, _column(batch, "t_dropoffloc")
            )
            pickup = self.gpd.GeoDataFrame(pickup, geometry="geometry", crs=CRS)
            dropoff = self.gpd.GeoDataFrame(dropoff, geometry="geometry", crs=CRS)
            pickup_join = self.gpd.sjoin(
                pickup, zones, how="inner", predicate="within"
            )[["t_tripkey", "z_zonekey"]]
            dropoff_join = self.gpd.sjoin(
                dropoff, zones, how="inner", predicate="within"
            )[["t_tripkey", "z_zonekey"]]
            pickup_counts = pickup_join.groupby("t_tripkey").size()
            dropoff_counts = dropoff_join.groupby("t_tripkey").size()
            total_pairs = pickup_counts.mul(dropoff_counts, fill_value=0).sum()
            same_zone = pickup_join.merge(
                dropoff_join, on=["t_tripkey", "z_zonekey"], how="inner"
            ).shape[0]
            count += int(total_pairs) - same_zone
        return pd.DataFrame({"cross_zone_trip_count": [count]})

    def _knn5_batch(self, pickups, buildings) -> np.ndarray:
        count = len(pickups)
        result = np.full(count, np.nan, dtype=np.float64)
        unresolved = np.arange(count, dtype=np.int64)
        minx, miny, maxx, maxy = buildings.total_bounds
        indexed_area = max(float(maxx - minx) * float(maxy - miny), 0.0)
        expected_fifth_radius = np.sqrt(
            5.0 * indexed_area / (np.pi * max(len(buildings), 1))
        )
        radii = np.full(
            count,
            max(expected_fifth_radius * 0.5, 0.0025),
            dtype=np.float64,
        )
        max_radius = 360.0
        while len(unresolved):
            subset = pickups.iloc[unresolved].reset_index(drop=True)
            pairs = buildings.sindex.query(
                subset.geometry,
                predicate="dwithin",
                distance=radii,
                sort=False,
            )
            if pairs.shape[1] == 0:
                radii *= 2.0
                if np.any(radii > max_radius):
                    raise RuntimeError("failed to find five buildings for a pickup")
                continue
            local_left = np.asarray(pairs[0], dtype=np.int64)
            right = np.asarray(pairs[1], dtype=np.int64)
            counts = np.bincount(local_left, minlength=len(unresolved))
            ready_local = np.flatnonzero(counts >= 5)
            ready_mask = np.isin(local_left, ready_local)
            if len(ready_local):
                selected_left = local_left[ready_mask]
                selected_right = right[ready_mask]
                pair_rows = self.distance_pair_rows or len(selected_left)
                distance_parts = []
                for start in range(0, len(selected_left), pair_rows):
                    stop = start + pair_rows
                    left_geom = self.gpd.GeoSeries(
                        subset.geometry.array.take(selected_left[start:stop]),
                        crs=CRS,
                    )
                    right_geom = self.gpd.GeoSeries(
                        buildings.geometry.array.take(selected_right[start:stop]),
                        crs=CRS,
                    )
                    distance_parts.append(
                        left_geom.distance(right_geom, align=False).to_numpy()
                    )
                distances = np.concatenate(distance_parts)
                candidates = pd.DataFrame(
                    {
                        "left": selected_left,
                        "building_key": buildings.iloc[selected_right][
                            "b_buildingkey"
                        ].to_numpy(),
                        "distance": distances,
                    }
                )
                means = (
                    candidates.sort_values(
                        ["left", "distance", "building_key"],
                        ascending=[True, True, True],
                    )
                    .groupby("left", sort=False)
                    .head(5)
                    .groupby("left")["distance"]
                    .mean()
                )
                result[unresolved[means.index.to_numpy(dtype=np.int64)]] = means.to_numpy()
            unresolved_mask = counts < 5
            unresolved = unresolved[unresolved_mask]
            radii = radii[unresolved_mask] * 2.0
            if np.any(radii > max_radius):
                raise RuntimeError("failed to find five buildings for a pickup")
        return result

    def q12(self, data_paths: dict[str, str]) -> DataFrame:
        buildings = self._buildings(data_paths)
        # Materialize the public index once and reuse it for every streamed trip batch.
        _ = buildings.sindex
        parts: list[DataFrame] = []
        columns = ["t_tripkey", "t_pickuploc"]
        for batch in _batches(data_paths["trip"], columns):
            pickups = _geometry_frame(
                self.gpd,
                batch,
                wkb_column="t_pickuploc",
                geometry_name="pickup_geom",
                columns=["t_tripkey"],
            )
            average = self._knn5_batch(pickups, buildings)
            frame = pd.DataFrame(
                {
                    "t_tripkey": pickups["t_tripkey"].to_numpy(),
                    "avg_distance_to_5_nearest": average,
                }
            )
            parts.append(
                frame.sort_values(
                    ["avg_distance_to_5_nearest", "t_tripkey"],
                    ascending=[False, True],
                ).head(100)
            )
        return _topk(
            parts,
            100,
            ["avg_distance_to_5_nearest", "t_tripkey"],
            [False, True],
        )
