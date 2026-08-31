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
"""Optimized SpatialBench implementation using public vibeSpatial APIs."""

import gc
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

try:
    from .geoparquet_public_api_queries import CRS, GeoParquetPublicApiQueries
except ImportError:  # SpatialBench loads this entrypoint as a standalone module.
    from geoparquet_public_api_queries import CRS, GeoParquetPublicApiQueries

import vibespatial as gpd

gpd.set_execution_mode(gpd.ExecutionMode.AUTO)


class VibeSpatialQueries(GeoParquetPublicApiQueries):
    """Public hybrid plan selected by physical workload shape."""

    _Q5_NATIVE_SPILL_GROUP_DOMAIN_THRESHOLD = 1_000_000_000

    @classmethod
    def _q5_uses_native_spill(cls, group_domain: int) -> bool:
        return int(group_domain) >= cls._Q5_NATIVE_SPILL_GROUP_DOMAIN_THRESHOLD

    def _candidate_zones_for_q6(self, data_paths):
        """Apply Q6's selective zone predicate before table concatenation."""
        bbox = Polygon(
            [
                (-112.2110, 34.4197),
                (-111.3110, 34.4197),
                (-111.3110, 35.3197),
                (-112.2110, 35.3197),
                (-112.2110, 34.4197),
            ]
        )
        accumulated = None
        for zones in self._spatial_frames(
            data_paths["zone"],
            ["z_zonekey", "z_name", "z_boundary"],
            "z_boundary",
            batch_rows=250_000,
        ):
            mask = zones.geometry.notna() & zones.geometry.intersects(bbox)
            selected = zones.loc[mask].reset_index(drop=True)
            if not selected.empty:
                accumulated = (
                    selected
                    if accumulated is None
                    else self.gpd.GeoDataFrame(
                        pd.concat([accumulated, selected], ignore_index=True),
                        geometry="z_boundary",
                        crs=CRS,
                    )
                )
            del mask, selected, zones
            gc.collect()
        if accumulated is None:
            return self.gpd.GeoDataFrame(
                columns=["z_zonekey", "z_name", "z_boundary"],
                geometry="z_boundary",
                crs=CRS,
            )
        return accumulated.reset_index(drop=True)

    def q2(self, data_paths):
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
            data_paths["trip"],
            ["t_pickuploc"],
            "t_pickuploc",
            batch_rows=8_000_000,
        ):
            count += int(
                trips.sindex.query(
                    target,
                    predicate="intersects",
                    sort=False,
                ).size
            )
        return pd.DataFrame({"trip_count_in_coconino_county": [count]})

    def _month_code(self, frame, column):
        return (
            frame.datetime_component(column, "year") * 12
            + frame.datetime_component(column, "month")
        )

    def q5(self, data_paths):
        """Run Q5 with a bounded, device-native partition-clustered spill.

        The public workload stays on ordinary frame, sort, GeoParquet write,
        filtered GeoParquet read, and dissolve APIs.  On vibeSpatial those
        operations retain Native* state, so the spill uses device WKB columns
        instead of the inherited Arrow-host partition writer.  One clustered
        file is written per bounded scan batch and Parquet statistics prune
        every unrelated partition during readback.
        """
        customer = pd.read_parquet(
            data_paths["customer"],
            columns=["c_custkey", "c_name"],
        )
        max_customer = int(customer["c_custkey"].max()) if len(customer) else -1
        month_width = 128
        group_domain = (max_customer + 1) * month_width
        if not self._q5_uses_native_spill(group_domain):
            # At SF100 (384M dense group slots), the existing Arrow-clustered
            # spill is still materially faster. The native sink becomes the
            # bounded plan once the group domain enters the scale where host
            # externalization and allocator churn dominated the saved run.
            return super().q5(data_paths)
        group_counts = None
        min_month = np.iinfo(np.int64).max
        max_month = np.iinfo(np.int64).min

        for trips in self._spatial_frames(
            data_paths["trip"],
            ["t_custkey", "t_pickuptime", "t_dropoffloc"],
            "t_dropoffloc",
        ):
            month_code = self._month_code(trips, "t_pickuptime")
            min_month = min(min_month, int(month_code.min()))
            max_month = max(max_month, int(month_code.max()))
            packed = trips["t_custkey"] * month_width + month_code % month_width
            batch_counts = self.gpd.dense_count(
                packed,
                size=group_domain,
                dtype=np.uint32,
                name="dropoff_count",
            )
            group_counts = (
                batch_counts if group_counts is None else group_counts + batch_counts
            )
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

        groups_per_partition = 4_000_000
        required_partitions = max(
            1,
            int(np.ceil(eligible_count / groups_per_partition)),
        )
        partition_count = min(
            1 << int(np.ceil(np.log2(required_partitions))),
            64,
        )
        source_columns = ["t_custkey", "t_pickuptime", "t_dropoffloc"]
        spill_columns = [
            "t_custkey",
            "_month_code",
            "t_dropoffloc",
            "_q5_partition",
        ]
        candidate_parts = []

        with TemporaryDirectory(prefix="spatialbench-q5-native-") as temporary:
            temporary_path = Path(temporary)
            spill_path = temporary_path / "q5-clustered.parquet"

            def _selected_batches():
                for trips in self._spatial_frames(
                    data_paths["trip"], source_columns, "t_dropoffloc"
                ):
                    month_code = self._month_code(trips, "t_pickuptime")
                    trips = trips.assign(_month_code=month_code)
                    packed = (
                        trips["t_custkey"] * month_width
                        + trips["_month_code"] % month_width
                    )
                    row_counts = self.gpd.numeric_take(group_counts, packed)
                    mask = row_counts > 5
                    selected = trips.loc[mask]
                    if selected.empty:
                        del mask, month_code, packed, row_counts, selected, trips
                        continue
                    selected = selected.assign(
                        _q5_partition=selected["t_custkey"] % partition_count,
                    )[spill_columns].reset_index(drop=True)
                    yield selected
                    del mask, month_code, packed, row_counts, selected, trips

            self.gpd.write_geoparquet(
                _selected_batches(),
                spill_path,
                index=False,
                geometry_encoding="WKB",
                partition_column="_q5_partition",
                partition_count=partition_count,
                max_row_group_rows=1_000_000,
                row_group_size=1_000_000,
            )

            partition_columns = [
                "t_custkey",
                "_month_code",
                "t_dropoffloc",
            ]
            for partition in range(partition_count):
                partition_frame = self.gpd.read_parquet(
                    spill_path,
                    columns=partition_columns,
                    filters=[("_q5_partition", "=", partition)],
                ).set_geometry("t_dropoffloc")
                if partition_frame.empty:
                    del partition_frame
                    continue
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
        top_group_codes = pd.Series(
            customer_keys * month_width + month_codes % month_width,
        )
        result["dropoff_count"] = self.gpd.numeric_take(
            group_counts,
            top_group_codes,
        ).to_numpy(dtype=np.int64)
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

    def _topk_frame(self, frame, k, *, by, ascending):
        # Encode ascending numeric keys as descending public expressions so
        # nlargest can lower to one bounded pylibcudf top-k rowset.
        assignments = {}
        rank_columns = []
        derived_columns = []
        for position, (column, is_ascending) in enumerate(
            zip(by, ascending, strict=True)
        ):
            expression = frame[column]
            if not pd.api.types.is_numeric_dtype(expression.dtype):
                cast_column = f"__vibespatial_topk_cast_{position}"
                expression = expression.astype(float)
                assignments[cast_column] = expression
                derived_columns.append(cast_column)
            else:
                cast_column = column
            if not is_ascending:
                rank_columns.append(cast_column)
                continue
            derived = f"__vibespatial_topk_tie_{position}"
            assignments[derived] = 0 - expression
            rank_columns.append(derived)
            derived_columns.append(derived)
        ranked = frame.assign(**assignments) if assignments else frame
        selected = ranked.nlargest(k, rank_columns)
        return selected.drop(columns=derived_columns) if derived_columns else selected

    def _q7_shard_topk(self, trips):
        pickup = trips.geometry
        dropoff = trips.set_geometry("t_dropoffloc").geometry
        line_distance = pickup.distance(dropoff, align=False) / 0.000009
        reported = trips["t_distance"].astype(float)
        detour_ratio = (reported / line_distance) * (line_distance / line_distance)
        metrics = trips.drop(columns=["t_dropoffloc"]).assign(
            reported_distance_m=reported,
            line_distance_m=line_distance,
            detour_ratio=detour_ratio,
            __vibespatial_topk_tie_2=0 - trips["t_tripkey"],
        )
        selected = metrics.nlargest(
            100,
            [
                "detour_ratio",
                "reported_distance_m",
                "__vibespatial_topk_tie_2",
            ],
        )
        columns = [
            "t_tripkey",
            "reported_distance_m",
            "line_distance_m",
            "detour_ratio",
        ]
        return pd.DataFrame(
            {column: selected[column].to_numpy() for column in columns}
        )

    def _q1_shard_topk(self, trips, center):
        distance = trips.geometry.distance(center)
        metrics = trips.assign(distance_to_center=distance)
        selected = self._topk_frame(
            metrics.loc[distance <= 0.45],
            100,
            by=["distance_to_center", "t_tripkey"],
            ascending=[True, True],
        )
        return pd.DataFrame(
            {
                "t_tripkey": selected["t_tripkey"].to_numpy(),
                "pickup_lon": selected.geometry.x.to_numpy(),
                "pickup_lat": selected.geometry.y.to_numpy(),
                "t_pickuptime": selected["t_pickuptime"].to_numpy(),
                "distance_to_center": selected["distance_to_center"].to_numpy(),
            }
        )

    def _knn5_batch(self, pickups, buildings):
        import numpy as np

        result = np.full(len(pickups), np.nan, dtype=np.float64)
        unresolved = np.arange(len(pickups), dtype=np.int64)
        minx, miny, maxx, maxy = buildings.total_bounds
        indexed_area = max(float(maxx - minx) * float(maxy - miny), 0.0)
        radius = max(
            np.sqrt(5.0 * indexed_area / (np.pi * max(len(buildings), 1))) * 0.5,
            0.0025,
        )
        while len(unresolved):
            subset = pickups.iloc[unresolved].reset_index(drop=True)
            indices, distances = buildings.sindex.nearest(
                subset.geometry,
                return_all=False,
                max_distance=radius,
                return_distance=True,
                k=5,
            )
            local_rows = np.asarray(indices[0], dtype=np.int64)
            counts = np.bincount(local_rows, minlength=len(unresolved))
            ready = counts >= 5
            if ready.any():
                sums = np.bincount(
                    local_rows,
                    weights=np.asarray(distances, dtype=np.float64),
                    minlength=len(unresolved),
                )
                result[unresolved[ready]] = sums[ready] / counts[ready]
            unresolved = unresolved[~ready]
            radius *= 2.0
            if len(unresolved) and radius > 360.0:
                raise RuntimeError("failed to find five buildings for a pickup")
        return result


_queries = VibeSpatialQueries(
    gpd,
    dissolve_method="coverage",
    distance_pair_rows=1_000_000,
    scan_batch_rows=32_000_000,
)

q1 = _queries.q1
q2 = _queries.q2
q3 = _queries.q3
q4 = _queries.q4
q5 = _queries.q5
q6 = _queries.q6
q7 = _queries.q7
q8 = _queries.q8
q9 = _queries.q9
q10 = _queries.q10
q11 = _queries.q11
q12 = _queries.q12
