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
"""Native-GeoParquet physical plans using GeoPandas-compatible public APIs."""

from __future__ import annotations

import gc
import json
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas import DataFrame

try:
    from .public_api_queries import (
        CRS,
        PublicApiQueries,
        _batches,
        _combine_sums,
        _parquet_files,
        _table,
        _topk,
    )
except ImportError:  # SpatialBench loads query entrypoints as standalone modules.
    from public_api_queries import (
        CRS,
        PublicApiQueries,
        _batches,
        _combine_sums,
        _parquet_files,
        _table,
        _topk,
    )
from shapely.geometry import Point, Polygon


def _primary_topk_positions(
    values: np.ndarray,
    k: int,
    *,
    largest: bool,
) -> np.ndarray:
    """Return every row tied at the primary-key top-k boundary in O(n)."""
    values = np.asarray(values)
    if len(values) <= k:
        return np.arange(len(values), dtype=np.int64)
    score = -values.astype(np.float64, copy=False) if largest else values
    score = np.where(np.isnan(score), np.inf, score)
    boundary = np.partition(score, k - 1)[k - 1]
    return np.flatnonzero(score <= boundary)


class GeoParquetPublicApiQueries(PublicApiQueries):
    """SQL-derived streaming plans over native GeoParquet geometry columns."""

    def _point_geoparquet_metadata(self, geometry_name: str) -> dict[str, Any]:
        """Derive portable point metadata through public GeoParquet APIs."""
        template = self.gpd.GeoDataFrame(
            {"_template": [0]},
            geometry=self.gpd.points_from_xy([0.0], [0.0], crs=CRS),
            crs=CRS,
        ).rename_geometry(geometry_name)
        buffer = BytesIO()
        template.to_parquet(
            buffer,
            geometry_encoding="geoarrow",
            index=False,
        )
        buffer.seek(0)
        metadata = dict(pq.read_schema(buffer).metadata or {})
        geo_metadata = json.loads(metadata[b"geo"])
        column_metadata = geo_metadata["columns"][geometry_name]
        # The template's [0, 0, 0, 0] bbox does not describe output shards.
        # GeoParquet permits bbox omission, so avoid publishing false pruning
        # metadata rather than scanning coordinates on the host.
        column_metadata.pop("bbox", None)
        geo_metadata["primary_column"] = geometry_name
        geo_metadata["columns"] = {geometry_name: column_metadata}
        return geo_metadata

    def _topk_frame(
        self,
        frame,
        k: int,
        *,
        by: list[str],
        ascending: list[bool],
    ):
        """Return an exact bounded ordering with the engine's public APIs."""
        positions = _primary_topk_positions(
            frame[by[0]].to_numpy(),
            k,
            largest=not ascending[0],
        )
        return frame.iloc[positions].sort_values(by, ascending=ascending).head(k)

    def _month_code(self, frame, column: str):
        values = pd.to_datetime(frame[column])
        return values.dt.year.astype(np.int64) * 12 + values.dt.month.astype(
            np.int64
        )

    @staticmethod
    def _duration_seconds(frame, start: str, end: str):
        return (
            frame[end].astype("int64") - frame[start].astype("int64")
        ) / 1000.0

    def _spatial_frames(
        self,
        path: str,
        columns: list[str],
        geometry: str,
        *,
        batch_rows: int | None = None,
    ) -> Iterator[Any]:
        batch_reader = getattr(self.gpd, "read_parquet_batches", None)
        source_files = _parquet_files(path)
        if batch_reader is not None and (
            len(source_files) >= 100 or batch_rows is not None
        ):
            resolved_batch_rows = batch_rows or self.scan_batch_rows or 2_000_000
            batches = iter(
                batch_reader(path, columns=columns, batch_rows=resolved_batch_rows)
            )
            while True:
                try:
                    source_frame = next(batches)
                except StopIteration:
                    return
                frame = source_frame.set_geometry(geometry).reset_index(drop=True)
                yield frame
                del frame, source_frame
        for file in source_files:
            frame = self.gpd.read_parquet(file, columns=columns)
            frame = frame.set_geometry(geometry)
            # Conversion preserves the source shard boundary (~3.9M fact
            # rows at SF100). read_parquet has already materialized the shard,
            # so subdividing it only repeats index construction and kernel
            # pipelines without lowering peak residency.
            yield frame.reset_index(drop=True)

    def _spatial_table(
        self,
        path: str,
        columns: list[str],
        geometry: str,
        *,
        batch_rows: int | None = None,
    ):
        parts = list(
            self._spatial_frames(
                path,
                columns,
                geometry,
                batch_rows=batch_rows,
            )
        )
        if not parts:
            return self.gpd.GeoDataFrame(columns=columns, geometry=geometry, crs=CRS)
        if len(parts) == 1:
            return parts[0]
        return self.gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True), geometry=geometry, crs=CRS
        )

    def q1(self, data_paths: dict[str, str]) -> DataFrame:
        center = Point(-111.7610, 34.8697)
        candidates: list[DataFrame] = []
        columns = ["t_tripkey", "t_pickuptime", "t_pickuploc"]
        for trips in self._spatial_frames(
            data_paths["trip"], columns, "t_pickuploc"
        ):
            selected = self._q1_shard_topk(trips, center)
            if not selected.empty:
                candidates.append(selected)
            del selected, trips
        result = _topk(
            candidates, 100, ["distance_to_center", "t_tripkey"], [True, True]
        )
        columns = [
            "t_tripkey",
            "pickup_lon",
            "pickup_lat",
            "t_pickuptime",
            "distance_to_center",
        ]
        return result.reindex(columns=columns)

    def _q1_shard_topk(self, trips, center):
        distance = trips.geometry.distance(center)
        mask = distance.notna().to_numpy() & (distance.to_numpy() <= 0.45)
        if not mask.any():
            return pd.DataFrame()
        selected = trips.loc[mask, ["t_tripkey", "t_pickuptime"]].copy()
        points = trips.geometry.loc[mask]
        selected["pickup_lon"] = points.x.to_numpy()
        selected["pickup_lat"] = points.y.to_numpy()
        selected["distance_to_center"] = distance.to_numpy()[mask]
        return selected.nsmallest(100, ["distance_to_center", "t_tripkey"])

    def q2(self, data_paths: dict[str, str]) -> DataFrame:
        target = None
        for zones in self._spatial_frames(
            data_paths["zone"], ["z_name", "z_boundary"], "z_boundary"
        ):
            matches = zones["z_name"] == "Coconino County"
            if matches.any():
                target = zones.loc[matches].geometry.iloc[0]
                break
        if target is None:
            return pd.DataFrame({"trip_count_in_coconino_county": [0]})
        count = 0
        for trips in self._spatial_frames(
            data_paths["trip"], ["t_pickuploc"], "t_pickuploc"
        ):
            count += int(trips.geometry.intersects(target).sum())
            del trips
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
            "t_pickuptime",
            "t_dropofftime",
            "t_distance",
            "t_fare",
            "t_pickuploc",
        ]
        parts: list[DataFrame] = []
        for trips in self._spatial_frames(
            data_paths["trip"], columns, "t_pickuploc"
        ):
            mask = trips.geometry.distance(polygon) <= 0.045
            if not mask.any():
                continue
            selected = trips.loc[mask]
            pickup_time = pd.Series(selected["t_pickuptime"].to_numpy())
            dropoff_time = pd.Series(selected["t_dropofftime"].to_numpy())
            frame = pd.DataFrame(
                {
                    "pickup_month": pickup_time
                    .dt.to_period("M")
                    .dt.to_timestamp(),
                    "_distance": selected["t_distance"].to_numpy(dtype=float),
                    "_fare": selected["t_fare"].to_numpy(dtype=float),
                    "_duration": (dropoff_time - pickup_time).dt.total_seconds(),
                }
            )
            parts.append(
                frame.groupby("pickup_month", as_index=False).agg(
                    total_trips=("_distance", "size"),
                    _distance_sum=("_distance", "sum"),
                    _duration_sum=("_duration", "sum"),
                    _fare_sum=("_fare", "sum"),
                )
            )
            del dropoff_time, frame, mask, pickup_time, selected, trips
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
        top_parts = []
        columns = ["t_tripkey", "t_tip", "t_pickuploc"]
        for trips in self._spatial_frames(
            data_paths["trip"], columns, "t_pickuploc"
        ):
            # Do not retain a GeoDataFrame slice here. Device-backed geometry
            # slices may share the complete source-shard coordinate allocation,
            # turning a logical 1,000-row top-k into 154 live shard buffers.
            # The SQL result needs geometry only after the global top-k, so keep
            # the bounded scalar payload and reconstruct those final points.
            rows = self._topk_frame(
                trips,
                1000,
                by=["t_tip", "t_tripkey"],
                ascending=[False, True],
            )
            top_parts.append(
                pd.DataFrame(
                    {
                        "t_tripkey": rows["t_tripkey"].to_numpy(),
                        "t_tip": rows["t_tip"].to_numpy(),
                        "_x": rows.geometry.x.to_numpy(),
                        "_y": rows.geometry.y.to_numpy(),
                    }
                )
            )
            del rows, trips
        top = pd.concat(top_parts, ignore_index=True).sort_values(
            ["t_tip", "t_tripkey"], ascending=[False, True]
        ).head(1000)
        top = self.gpd.GeoDataFrame(
            top.drop(columns=["_x", "_y"]),
            geometry=self.gpd.points_from_xy(top["_x"], top["_y"], crs=CRS),
            crs=CRS,
        )
        parts = []
        columns = ["z_zonekey", "z_name", "z_boundary"]
        for zones in self._spatial_frames(
            data_paths["zone"], columns, "z_boundary"
        ):
            joined = self.gpd.sjoin(top, zones, how="inner", predicate="within")
            if not joined.empty:
                parts.append(joined[["z_zonekey", "z_name"]])
        if not parts:
            return pd.DataFrame(columns=["z_zonekey", "z_name", "trip_count"])
        return (
            pd.concat(parts, ignore_index=True)
            .groupby(["z_zonekey", "z_name"], as_index=False)
            .size()
            .rename(columns={"size": "trip_count"})
            .sort_values(["trip_count", "z_zonekey"], ascending=[False, True])
            .reset_index(drop=True)
        )

    def q5(self, data_paths: dict[str, str]) -> DataFrame:
        customer = _table(
            data_paths["customer"], ["c_custkey", "c_name"]
        ).to_pandas()
        max_customer = int(customer["c_custkey"].max()) if len(customer) else -1
        month_width = 128
        group_domain = (max_customer + 1) * month_width
        dense_count = getattr(self.gpd, "dense_count", None)
        numeric_take = getattr(self.gpd, "numeric_take", None)
        use_native_dense_count = dense_count is not None and numeric_take is not None
        group_counts = (
            None
            if use_native_dense_count
            else np.zeros(group_domain, dtype=np.uint16)
        )
        min_month = np.iinfo(np.int64).max
        max_month = np.iinfo(np.int64).min
        if use_native_dense_count:
            count_frames = self._spatial_frames(
                data_paths["trip"],
                ["t_custkey", "t_pickuptime", "t_dropoffloc"],
                "t_dropoffloc",
            )
        else:
            # The count-only pass has no spatial semantics. Decoding 384M
            # dropoff points into Shapely objects retains tens of GiB in the
            # host allocator across shards and can OOM before the selective
            # geometry pass begins. Public Arrow batches preserve the two
            # required attribute dtypes without constructing geometry.
            count_frames = (
                batch.to_pandas()
                for batch in _batches(
                    data_paths["trip"],
                    ["t_custkey", "t_pickuptime"],
                )
            )
        for trips in count_frames:
            month_code = self._month_code(trips, "t_pickuptime")
            min_month = min(min_month, int(month_code.min()))
            max_month = max(max_month, int(month_code.max()))
            packed = trips["t_custkey"] * month_width + month_code % month_width
            if use_native_dense_count:
                batch_counts = dense_count(
                    packed,
                    size=group_domain,
                    dtype=np.uint32,
                    name="dropoff_count",
                )
                group_counts = (
                    batch_counts
                    if group_counts is None
                    else group_counts + batch_counts
                )
            else:
                packed_values = packed.to_numpy(dtype=np.int64, copy=False)
                unique_codes, batch_counts = np.unique(
                    packed_values,
                    return_counts=True,
                )
                updated = group_counts[unique_codes].astype(np.uint32) + batch_counts
                if int(updated.max(initial=0)) > np.iinfo(np.uint16).max:
                    raise RuntimeError("Q5 group count exceeds uint16 capacity")
                group_counts[unique_codes] = updated.astype(np.uint16)
                del packed_values, unique_codes, updated
            del batch_counts, month_code, packed, trips

        if max_month - min_month >= month_width:
            raise RuntimeError("Q5 month packing width is smaller than the data span")
        eligible_count = int((group_counts > 5).sum())
        if eligible_count == 0:
            return pd.DataFrame(
                columns=[
                    "c_custkey",
                    "customer_name",
                    "pickup_month",
                    "monthly_travel_hull_area",
                    "dropoff_count",
                ]
            )
        groups_per_partition = 1_000_000
        required_partitions = max(
            1,
            int(np.ceil(eligible_count / groups_per_partition)),
        )
        partition_count = 1 << int(np.ceil(np.log2(required_partitions)))
        partition_count = min(partition_count, 64)
        point_geo_metadata = self._point_geoparquet_metadata("t_dropoffloc")
        columns = ["t_custkey", "t_pickuptime", "t_dropoffloc"]
        candidate_parts: list[DataFrame] = []
        with TemporaryDirectory(prefix="spatialbench-q5-") as temporary:
            temporary_path = Path(temporary)
            batch_number = 0
            for trips in self._spatial_frames(
                data_paths["trip"], columns, "t_dropoffloc"
            ):
                month_code = self._month_code(trips, "t_pickuptime")
                trips = trips.assign(
                    _month_code=month_code,
                )
                packed = (
                    trips["t_custkey"] * month_width
                    + trips["_month_code"] % month_width
                )
                if use_native_dense_count:
                    row_counts = numeric_take(group_counts, packed)
                    mask = row_counts > 5
                    selected = trips.loc[mask]
                else:
                    packed_values = packed.to_numpy(dtype=np.int64, copy=False)
                    mask = group_counts[packed_values] > 5
                    if not mask.any():
                        del mask, month_code, packed, packed_values, trips
                        continue
                    selected = trips.loc[mask]
                    del packed_values
                if selected.empty:
                    del mask, month_code, packed, selected, trips
                    if use_native_dense_count:
                        del row_counts
                    continue
                selected = selected.assign(
                    _q5_partition=selected["t_custkey"] % partition_count,
                )
                arrow_table = pa.table(
                    selected.to_arrow(index=False, geometry_encoding="geoarrow")
                ).select(
                    [
                        "t_custkey",
                        "_month_code",
                        "t_dropoffloc",
                        "_q5_partition",
                    ]
                )
                metadata = dict(arrow_table.schema.metadata or {})
                metadata[b"geo"] = json.dumps(point_geo_metadata).encode()
                arrow_table = arrow_table.replace_schema_metadata(metadata)
                pq.write_to_dataset(
                    arrow_table,
                    root_path=temporary_path,
                    partition_cols=["_q5_partition"],
                    basename_template=f"batch-{batch_number:03d}-{{i}}.parquet",
                )
                batch_number += 1
                del (
                    arrow_table,
                    mask,
                    month_code,
                    packed,
                    selected,
                    trips,
                )
                if use_native_dense_count:
                    del row_counts

            partition_columns = [
                "t_custkey",
                "_month_code",
                "t_dropoffloc",
            ]
            for partition_path in sorted(temporary_path.glob("_q5_partition=*")):
                partition_frame = self._spatial_table(
                    str(partition_path),
                    partition_columns,
                    "t_dropoffloc",
                    batch_rows=16_000_000,
                )
                dissolved = partition_frame.dissolve(
                    by=["t_custkey", "_month_code"],
                    method="unary",
                )
                rank_source = dissolved.assign(
                    monthly_travel_hull_area=dissolved.geometry.convex_hull.area,
                ).drop(columns=dissolved.geometry.name)
                ranked = rank_source.nlargest(100, ["monthly_travel_hull_area"])
                candidate_parts.append(
                    pd.DataFrame(
                        {
                            "t_custkey": ranked.index.get_level_values(
                                "t_custkey"
                            ).to_numpy(dtype=np.int64),
                            "pickup_month_code": ranked.index.get_level_values(
                                "_month_code"
                            ).to_numpy(dtype=np.int64),
                            "monthly_travel_hull_area": ranked[
                                "monthly_travel_hull_area"
                            ].to_numpy(dtype=np.float64),
                        }
                    )
                )
                del dissolved, partition_frame, ranked

        if not candidate_parts:
            return pd.DataFrame(
                columns=[
                    "c_custkey",
                    "customer_name",
                    "pickup_month",
                    "monthly_travel_hull_area",
                    "dropoff_count",
                ]
            )
        result = (
            pd.concat(candidate_parts, ignore_index=True)
            .sort_values(
                ["monthly_travel_hull_area", "t_custkey", "pickup_month_code"],
                ascending=[False, True, True],
            )
            .head(100)
        )
        month_codes = result.pop("pickup_month_code").to_numpy(dtype=np.int64)
        customer_keys = result["t_custkey"].to_numpy(dtype=np.int64)
        result["t_custkey"] = customer_keys
        top_group_codes = pd.Series(
            customer_keys * month_width + month_codes % month_width,
        )
        if use_native_dense_count:
            result["dropoff_count"] = numeric_take(
                group_counts,
                top_group_codes,
            ).to_numpy(dtype=np.int64)
        else:
            result["dropoff_count"] = group_counts[
                top_group_codes.to_numpy(dtype=np.int64, copy=False)
            ].astype(np.int64)
        years, months = np.divmod(month_codes - 1, 12)
        result["pickup_month"] = pd.to_datetime(
            pd.DataFrame({"year": years, "month": months + 1, "day": 1})
        ).to_numpy()
        customer_names = customer.set_index("c_custkey")["c_name"]
        result["c_custkey"] = customer_keys
        result["c_name"] = customer_names.reindex(customer_keys).to_numpy()
        result = result[result["c_name"].notna()]
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
        zones = self._zones(data_paths)
        mask = zones.geometry.notna() & zones.geometry.intersects(bbox)
        return zones.loc[mask].reset_index(drop=True)

    @staticmethod
    def _zone_pairs(points, zones) -> tuple[np.ndarray, np.ndarray]:
        """Return point-row and zone-row arrays without materializing a join frame."""
        # ST_Within(point, polygon) is equivalent to ST_Contains(polygon, point).
        # Index the high-cardinality points and query with the polygons so exact
        # refinement can prepare each polygon once, matching the sjoin planner.
        pairs = points.sindex.query(
            zones.geometry, predicate="contains", sort=False
        )
        return (
            np.asarray(pairs[1], dtype=np.int64),
            np.asarray(pairs[0], dtype=np.int64),
        )

    def _zone_aggregates(self, points, zones, distances, durations) -> DataFrame:
        """Reduce point-in-zone matches through the engine's public API."""
        query_aggregate = getattr(points.sindex, "query_aggregate", None)
        if query_aggregate is not None:
            return query_aggregate(
                zones.geometry,
                {
                    "total_pickups": "size",
                    "distance_sum": (distances, "sum"),
                    "duration_sum": (durations, "sum"),
                },
                predicate="contains",
            )

        point_rows, zone_rows = self._zone_pairs(points, zones)
        return pd.DataFrame(
            {
                "total_pickups": np.bincount(
                    zone_rows,
                    minlength=len(zones),
                ),
                "distance_sum": np.bincount(
                    zone_rows,
                    weights=np.asarray(distances)[point_rows],
                    minlength=len(zones),
                ),
                "duration_sum": np.bincount(
                    zone_rows,
                    weights=np.asarray(durations)[point_rows],
                    minlength=len(zones),
                ),
            },
            index=zones.index,
        )

    @staticmethod
    def _zone_pair_aggregates(pickup, dropoff, zones) -> DataFrame:
        """Reduce aligned endpoint memberships through public engine APIs."""
        pickup_index = pickup.sindex
        query_pair_aggregate = getattr(
            pickup_index,
            "query_pair_aggregate",
            None,
        )
        if query_pair_aggregate is not None:
            return query_pair_aggregate(
                dropoff.sindex,
                zones.geometry,
                predicate="contains",
            )

        pickup_rows, pickup_zones = GeoParquetPublicApiQueries._zone_pairs(
            pickup,
            zones,
        )
        dropoff_rows, dropoff_zones = GeoParquetPublicApiQueries._zone_pairs(
            dropoff,
            zones,
        )
        row_count = len(pickup)
        zone_count = len(zones)
        if zone_count == 0:
            zeros = np.zeros(row_count, dtype=np.int64)
            return pd.DataFrame(
                {
                    "left_count": zeros,
                    "right_count": zeros.copy(),
                    "shared_count": zeros.copy(),
                }
            )
        pickup_counts = np.bincount(
            pickup_rows,
            minlength=row_count,
        ).astype(np.int64, copy=False)
        dropoff_counts = np.bincount(
            dropoff_rows,
            minlength=row_count,
        ).astype(np.int64, copy=False)
        pickup_codes = pickup_rows * zone_count + pickup_zones
        dropoff_codes = dropoff_rows * zone_count + dropoff_zones
        shared_codes = np.intersect1d(
            pickup_codes,
            dropoff_codes,
            assume_unique=True,
        )
        shared_counts = np.bincount(
            shared_codes // zone_count,
            minlength=row_count,
        ).astype(np.int64, copy=False)
        return pd.DataFrame(
            {
                "left_count": pickup_counts,
                "right_count": dropoff_counts,
                "shared_count": shared_counts,
            }
        )

    @staticmethod
    def _add_zone_aggregates(
        accumulated: DataFrame | None,
        current: DataFrame,
    ) -> DataFrame:
        """Add eager public aggregate columns without forcing array export."""
        if accumulated is None:
            return current
        return pd.DataFrame(
            {
                column: accumulated[column] + current[column]
                for column in current.columns
            },
            index=current.index,
        )

    def q6(self, data_paths: dict[str, str]) -> DataFrame:
        zones = self._candidate_zones_for_q6(data_paths)
        zone_keys = zones["z_zonekey"].to_numpy(dtype=np.int64)
        zone_names = zones["z_name"].to_numpy()
        aggregate_total = None
        columns = [
            "t_tripkey",
            "t_pickuploc",
            "t_pickuptime",
            "t_dropofftime",
            "t_distance",
        ]
        for trips in self._spatial_frames(
            data_paths["trip"],
            columns,
            "t_pickuploc",
            batch_rows=8_000_000,
        ):
            trips["t_distance"] = trips["t_distance"].astype(float)
            durations = self._duration_seconds(
                trips,
                "t_pickuptime",
                "t_dropofftime",
            )
            aggregates = self._zone_aggregates(
                trips,
                zones,
                trips["t_distance"],
                durations,
            )
            aggregate_total = self._add_zone_aggregates(
                aggregate_total,
                aggregates,
            )
            del durations, trips
            gc.collect()
        if aggregate_total is None:
            total_pickups = np.zeros(len(zones), dtype=np.int64)
            distance_sum = np.zeros(len(zones), dtype=np.float64)
            duration_sum = np.zeros(len(zones), dtype=np.float64)
        else:
            total_pickups = aggregate_total["total_pickups"].to_numpy(
                dtype=np.int64,
                copy=False,
            )
            distance_sum = aggregate_total["distance_sum"].to_numpy(
                dtype=np.float64,
                copy=False,
            )
            duration_sum = aggregate_total["duration_sum"].to_numpy(
                dtype=np.float64,
                copy=False,
            )
        keep = total_pickups != 0
        result = pd.DataFrame(
            {
                "z_zonekey": zone_keys[keep],
                "z_name": zone_names[keep],
                "total_pickups": total_pickups[keep],
                "avg_distance": distance_sum[keep] / total_pickups[keep],
                "avg_duration": duration_sum[keep] / total_pickups[keep],
            }
        )
        return result.sort_values(
            ["total_pickups", "z_zonekey"], ascending=[False, True]
        ).reset_index(drop=True)

    def _q7_shard_topk(self, trips):
        pickup = trips.geometry
        dropoff = trips.set_geometry("t_dropoffloc").geometry
        line_distance = pickup.distance(dropoff, align=False).to_numpy() / 0.000009
        reported = trips["t_distance"].to_numpy(dtype=float)
        ratio = np.divide(
            reported,
            line_distance,
            out=np.full_like(reported, np.nan),
            where=line_distance != 0.0,
        )
        frame = pd.DataFrame(
            {
                "t_tripkey": trips["t_tripkey"].to_numpy(dtype=np.int64),
                "reported_distance_m": reported,
                "line_distance_m": line_distance,
                "detour_ratio": ratio,
            }
        )
        positions = _primary_topk_positions(ratio, 100, largest=True)
        return (
            frame.iloc[positions]
            .sort_values(
                ["detour_ratio", "reported_distance_m", "t_tripkey"],
                ascending=[False, False, True],
                na_position="last",
            )
            .head(100)
        )

    def q7(self, data_paths: dict[str, str]) -> DataFrame:
        parts: list[DataFrame] = []
        columns = ["t_tripkey", "t_distance", "t_pickuploc", "t_dropoffloc"]
        for trips in self._spatial_frames(
            data_paths["trip"], columns, "t_pickuploc"
        ):
            part = self._q7_shard_topk(trips)
            parts.append(part)
            del part, trips
        return _topk(
            parts,
            100,
            ["detour_ratio", "reported_distance_m", "t_tripkey"],
            [False, False, True],
        )

    def _buildings(self, data_paths: dict[str, str], include_name: bool = False):
        columns = ["b_buildingkey", "b_boundary"]
        if include_name:
            columns.insert(1, "b_name")
        return self._spatial_table(data_paths["building"], columns, "b_boundary")

    def q8(self, data_paths: dict[str, str]) -> DataFrame:
        buildings = self._buildings(data_paths, include_name=True)
        nearby_counts = np.zeros(len(buildings), dtype=np.int64)
        for pickups in self._spatial_frames(
            data_paths["trip"], ["t_pickuploc"], "t_pickuploc"
        ):
            pairs = buildings.sindex.query(
                pickups.geometry,
                predicate="dwithin",
                distance=0.0045,
                sort=False,
            )
            if pairs.shape[1]:
                nearby_counts += np.bincount(
                    np.asarray(pairs[1], dtype=np.int64), minlength=len(buildings)
                )
            del pickups
        keep = nearby_counts != 0
        result = buildings.loc[keep, ["b_buildingkey", "b_name"]].reset_index(
            drop=True
        )
        result["nearby_pickup_count"] = nearby_counts[keep]
        return (
            result.sort_values(
                ["nearby_pickup_count", "b_buildingkey"], ascending=[False, True]
            )
            .head(100)
            .reset_index(drop=True)
        )

    def _zones(self, data_paths: dict[str, str], name_column: str = "z_name"):
        zones = self._spatial_table(
            data_paths["zone"], ["z_zonekey", "z_name", "z_boundary"], "z_boundary"
        )
        if name_column != "z_name":
            zones = zones.rename(columns={"z_name": name_column})
        return zones

    def _zone_frames(self, data_paths: dict[str, str]):
        """Keep the large zone geometry in public row-group partitions."""
        return list(
            self._spatial_frames(
                data_paths["zone"],
                ["z_zonekey", "z_name", "z_boundary"],
                "z_boundary",
                batch_rows=250_000,
            )
        )

    def q10(self, data_paths: dict[str, str]) -> DataFrame:
        zone_frames = self._zone_frames(data_paths)
        zone_count = sum(len(zones) for zones in zone_frames)
        zone_keys = np.concatenate(
            [zones["z_zonekey"].to_numpy(dtype=np.int64) for zones in zone_frames]
        )
        zone_names = np.concatenate(
            [zones["z_name"].to_numpy() for zones in zone_frames]
        )
        num_trips = np.zeros(zone_count, dtype=np.int64)
        distance_sum = np.zeros(zone_count, dtype=np.float64)
        duration_sum = np.zeros(zone_count, dtype=np.float64)
        aggregate_totals: list[DataFrame | None] = [None] * len(zone_frames)
        columns = [
            "t_tripkey",
            "t_pickuploc",
            "t_pickuptime",
            "t_dropofftime",
            "t_distance",
        ]
        for trips in self._spatial_frames(
            data_paths["trip"],
            columns,
            "t_pickuploc",
            batch_rows=4_000_000,
        ):
            trips["t_distance"] = trips["t_distance"].astype(float)
            durations = self._duration_seconds(
                trips,
                "t_pickuptime",
                "t_dropofftime",
            )
            for zone_index, zones in enumerate(zone_frames):
                aggregates = self._zone_aggregates(
                    trips,
                    zones,
                    trips["t_distance"],
                    durations,
                )
                aggregate_totals[zone_index] = self._add_zone_aggregates(
                    aggregate_totals[zone_index],
                    aggregates,
                )
            del durations, trips
            gc.collect()
        zone_offset = 0
        for zones, aggregates in zip(zone_frames, aggregate_totals, strict=True):
            zone_stop = zone_offset + len(zones)
            if aggregates is not None:
                num_trips[zone_offset:zone_stop] = aggregates[
                    "total_pickups"
                ].to_numpy(dtype=np.int64, copy=False)
                distance_sum[zone_offset:zone_stop] = aggregates[
                    "distance_sum"
                ].to_numpy(dtype=np.float64, copy=False)
                duration_sum[zone_offset:zone_stop] = aggregates[
                    "duration_sum"
                ].to_numpy(dtype=np.float64, copy=False)
            zone_offset = zone_stop
        result = pd.DataFrame(
            {
                "z_zonekey": zone_keys,
                "z_name": zone_names,
                "avg_duration": np.divide(
                    duration_sum,
                    num_trips,
                    out=np.full(zone_count, np.nan, dtype=np.float64),
                    where=num_trips != 0,
                ),
                "avg_distance": np.divide(
                    distance_sum,
                    num_trips,
                    out=np.full(zone_count, np.nan, dtype=np.float64),
                    where=num_trips != 0,
                ),
                "num_trips": num_trips,
            }
        )
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
        zone_frames = self._zone_frames(data_paths)
        count = 0
        columns = ["t_tripkey", "t_pickuploc", "t_dropoffloc"]
        for trips in self._spatial_frames(
            data_paths["trip"],
            columns,
            "t_pickuploc",
            batch_rows=4_000_000,
        ):
            pickup = trips.set_geometry("t_pickuploc").geometry
            dropoff = trips.set_geometry("t_dropoffloc").geometry
            aggregate_total = None
            for zones in zone_frames:
                aggregates = self._zone_pair_aggregates(
                    pickup,
                    dropoff,
                    zones,
                )
                aggregate_total = self._add_zone_aggregates(
                    aggregate_total,
                    aggregates,
                )
            if aggregate_total is not None:
                count += int(
                    (
                        aggregate_total["left_count"]
                        * aggregate_total["right_count"]
                    ).sum()
                    - aggregate_total["shared_count"].sum()
                )
            del (
                aggregate_total,
                dropoff,
                pickup,
                trips,
            )
            gc.collect()
        return pd.DataFrame({"cross_zone_trip_count": [count]})

    def _knn5_relation(self, pickups, buildings):
        spatial_index = buildings.sindex
        unresolved = np.arange(len(pickups), dtype=np.int64)
        minx, miny, maxx, maxy = buildings.total_bounds
        area = max(float(maxx - minx) * float(maxy - miny), 0.0)
        initial = max(
            0.5 * np.sqrt(5.0 * area / (np.pi * max(len(buildings), 1))),
            0.0025,
        )
        radii = np.full(len(pickups), initial, dtype=np.float64)
        left_parts = []
        right_parts = []
        distance_parts = []
        while len(unresolved):
            subset = self.gpd.GeoSeries(
                pickups.geometry.array.take(unresolved),
                crs=CRS,
            )
            pairs = spatial_index.query(
                subset,
                predicate="dwithin",
                distance=radii,
                sort=False,
            )
            local_left = np.asarray(pairs[0], dtype=np.int64)
            right = np.asarray(pairs[1], dtype=np.int64)
            counts = np.bincount(local_left, minlength=len(unresolved))
            ready = counts >= 5
            ready_pairs = ready[local_left]
            if ready.any():
                selected_left = local_left[ready_pairs]
                selected_right = right[ready_pairs]
                left_geometry = self.gpd.GeoSeries(
                    subset.array.take(selected_left),
                    crs=CRS,
                )
                right_geometry = self.gpd.GeoSeries(
                    buildings.geometry.array.take(selected_right),
                    crs=CRS,
                )
                distances = left_geometry.distance(
                    right_geometry,
                    align=False,
                ).to_numpy()
                candidates = pd.DataFrame(
                    {
                        "left": selected_left,
                        "right": selected_right,
                        "building_key": buildings["b_buildingkey"].to_numpy()[
                            selected_right
                        ],
                        "distance": distances,
                    }
                )
                selected = (
                    candidates.sort_values(
                        ["left", "distance", "building_key"],
                        ascending=[True, True, True],
                    )
                    .groupby("left", sort=False)
                    .head(5)
                )
                left_parts.append(
                    unresolved[selected["left"].to_numpy(dtype=np.int64)]
                )
                right_parts.append(selected["right"].to_numpy(dtype=np.int64))
                distance_parts.append(selected["distance"].to_numpy())
            unresolved = unresolved[~ready]
            radii = radii[~ready] * 2.0
            if len(radii) and np.any(radii > 360.0):
                raise RuntimeError("failed to find five buildings for a pickup")
        return (
            np.concatenate(left_parts),
            np.concatenate(right_parts),
            np.concatenate(distance_parts),
        )

    def _q12_grid_representatives(self, buildings):
        minx, miny, maxx, maxy = buildings.total_bounds
        grid_x = 64
        grid_y = 64
        width = max(float(maxx - minx), np.finfo(np.float64).eps)
        height = max(float(maxy - miny), np.finfo(np.float64).eps)
        fractions = np.minimum((np.arange(grid_x) + 0.25) / (grid_x - 1), 1.0)
        x = minx + fractions * width
        y = miny + fractions * height
        anchors = buildings.geometry.representative_point()
        anchor_x = anchors.x.to_numpy()
        anchor_y = anchors.y.to_numpy()
        representatives = np.empty((grid_x * grid_y, 5), dtype=np.int64)
        center_x = np.tile(x, grid_y)
        center_y = np.repeat(y, grid_x)
        centers = self.gpd.GeoSeries(
            self.gpd.points_from_xy(center_x, center_y, crs=CRS),
            crs=CRS,
        )
        codes = centers.hilbert_distance(
            total_bounds=(minx, miny, maxx, maxy),
            level=6,
        ).to_numpy(dtype=np.int64)
        if len(np.unique(codes)) != grid_x * grid_y:
            raise RuntimeError("Hilbert representative grid did not cover every cell")
        for row in range(grid_y):
            for column in range(grid_x):
                distance = np.square(anchor_x - x[column]) + np.square(
                    anchor_y - y[row]
                )
                selected = np.argpartition(distance, 4)[:5]
                representatives[codes[row * grid_x + column]] = selected[
                    np.lexsort((selected, distance[selected]))
                ]
        return (
            self._q12_anchor_lookup(anchor_x, anchor_y, representatives),
            (minx, miny, maxx, maxy),
        )

    def _q12_anchor_lookup(self, anchor_x, anchor_y, representatives):
        lookup = self.gpd.GeoDataFrame(
            {
                f"representative_{position}": self.gpd.GeoSeries(
                    self.gpd.points_from_xy(
                        anchor_x[representatives[:, position]],
                        anchor_y[representatives[:, position]],
                        crs=CRS,
                    ),
                    crs=CRS,
                )
                for position in range(5)
            },
            geometry="representative_0",
            crs=CRS,
        )
        # A tiny in-memory GeoParquet roundtrip establishes one public,
        # device-native frame containing all five geometry columns.
        buffer = BytesIO()
        lookup.to_parquet(
            buffer,
            geometry_encoding="geoarrow",
            index=False,
        )
        buffer.seek(0)
        return self.gpd.read_parquet(buffer)

    def _q12_upper_metrics(self, pickups, anchor_lookup, total_bounds):
        codes = pickups.geometry.hilbert_distance(
            total_bounds=total_bounds,
            level=6,
        )
        distances = [
            pickups.geometry.distance(
                anchor_lookup[f"representative_{position}"].take(codes),
                align=False,
            )
            for position in range(5)
        ]
        # The fixed epsilon keeps the computed mean conservatively above the
        # five exact point-to-building distances despite floating-point divide.
        upper = sum(distances[1:], distances[0]) / 5.0 + 1.0e-12
        return pickups.assign(__q12_upper_bound=upper)

    def _q12_device_point_frame(self, frame):
        buffer = BytesIO()
        frame.to_parquet(buffer, geometry_encoding="geoarrow", index=False)
        buffer.seek(0)
        return self.gpd.read_parquet(buffer).set_geometry("t_pickuploc")

    def _q12_exact_rows(self, pickups, buildings, rows):
        subset = self.gpd.GeoDataFrame(
            {
                "t_tripkey": pickups["t_tripkey"].to_numpy()[rows],
                "t_pickuploc": self.gpd.GeoSeries(
                    pickups.geometry.array.take(rows),
                    crs=CRS,
                ),
            },
            geometry="t_pickuploc",
            crs=CRS,
        )
        means = self._knn5_batch(subset, buildings)
        return pd.DataFrame(
            {
                "t_tripkey": subset["t_tripkey"].to_numpy(),
                "avg_distance_to_5_nearest": means,
            }
        )

    def q12(self, data_paths: dict[str, str]) -> DataFrame:
        buildings = self._buildings(data_paths)
        _ = buildings.sindex
        anchor_lookup, total_bounds = self._q12_grid_representatives(buildings)
        # A representative point lies on its building, so pickup-to-anchor
        # distance is an upper bound on pickup-to-building distance. Any five
        # anchors therefore preserve the exact branch-and-bound proof while
        # avoiding repeated multi-ring polygon buffers in the row-aligned pass.
        retained_parts = []
        shard_state = []
        columns = ["t_tripkey", "t_pickuploc"]
        for batch_index, pickups in enumerate(
            self._spatial_frames(data_paths["trip"], columns, "t_pickuploc")
        ):
            metrics = self._q12_upper_metrics(
                pickups,
                anchor_lookup,
                total_bounds,
            )
            selected = self._topk_frame(
                metrics,
                1_000,
                by=["__q12_upper_bound", "t_tripkey"],
                ascending=[False, True],
            )
            tripkeys = selected["t_tripkey"].to_numpy()
            upper = selected["__q12_upper_bound"].to_numpy()
            retained_parts.append(
                pd.DataFrame(
                    {
                        "t_tripkey": tripkeys,
                        "x": selected.geometry.x.to_numpy(),
                        "y": selected.geometry.y.to_numpy(),
                        "upper": upper,
                    }
                )
            )
            discarded_bound = (
                -np.inf
                if len(selected) == len(metrics)
                else float(np.min(upper))
            )
            shard_state.append((batch_index, discarded_bound, frozenset(tripkeys)))
            del metrics, selected, pickups

        retained = pd.concat(retained_parts, ignore_index=True).sort_values(
            ["upper", "t_tripkey"], ascending=[False, True]
        ).reset_index(drop=True)
        exact_parts = []
        position = 0
        threshold = -np.inf
        while position < len(retained) and (
            position < 100 or retained["upper"].iloc[position] >= threshold
        ):
            if exact_parts:
                potential_end = int(
                    np.searchsorted(
                        -retained["upper"].to_numpy(), -threshold, side="right"
                    )
                )
            else:
                potential_end = len(retained)
            stop = min(
                len(retained),
                max(position + 1_000, min(potential_end, position + 100_000)),
            )
            candidates = retained.iloc[position:stop]
            candidate_points = self.gpd.GeoDataFrame(
                {
                    "t_tripkey": candidates["t_tripkey"].to_numpy(),
                    "t_pickuploc": self.gpd.points_from_xy(
                        candidates["x"].to_numpy(),
                        candidates["y"].to_numpy(),
                        crs=CRS,
                    ),
                },
                geometry="t_pickuploc",
                crs=CRS,
            )
            candidate_points = self._q12_device_point_frame(candidate_points)
            exact_parts.append(
                self._q12_exact_rows(
                    candidate_points,
                    buildings,
                    np.arange(len(candidate_points), dtype=np.int64),
                )
            )
            position = stop
            current = _topk(
                exact_parts,
                100,
                ["avg_distance_to_5_nearest", "t_tripkey"],
                [False, True],
            )
            if len(current) == 100:
                threshold = float(current["avg_distance_to_5_nearest"].iloc[-1])

        revisit = {
            batch_index: retained_keys
            for batch_index, discarded_bound, retained_keys in shard_state
            if discarded_bound >= threshold
        }
        for batch_index, pickups in enumerate(
            self._spatial_frames(data_paths["trip"], columns, "t_pickuploc")
        ):
            if batch_index not in revisit:
                continue
            metrics = self._q12_upper_metrics(pickups, anchor_lookup, total_bounds)
            possible = metrics.loc[metrics["__q12_upper_bound"] >= threshold]
            keys = possible["t_tripkey"].to_numpy()
            rows = np.flatnonzero(~np.isin(keys, list(revisit[batch_index])))
            if len(rows):
                exact_parts.append(self._q12_exact_rows(possible, buildings, rows))
                current = _topk(
                    exact_parts,
                    100,
                    ["avg_distance_to_5_nearest", "t_tripkey"],
                    [False, True],
                )
                threshold = float(
                    current["avg_distance_to_5_nearest"].iloc[-1]
                )
        return current
