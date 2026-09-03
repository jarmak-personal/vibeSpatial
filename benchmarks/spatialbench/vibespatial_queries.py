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
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

try:
    from .geoparquet_public_api_queries import CRS, GeoParquetPublicApiQueries
except ImportError:  # SpatialBench loads this entrypoint as a standalone module.
    from geoparquet_public_api_queries import CRS, GeoParquetPublicApiQueries

import vibespatial as gpd
from vibespatial.api.tabular import _streaming_topk

gpd.set_execution_mode(gpd.ExecutionMode.AUTO)


class VibeSpatialQueries(GeoParquetPublicApiQueries):
    """Public hybrid plan selected by physical workload shape."""

    _Q5_NATIVE_SPILL_GROUP_DOMAIN_THRESHOLD = 1_000_000_000
    # pandas' nanosecond datetime range fits below this absolute year*12+month
    # code. Keeping the whole legal domain dense avoids per-batch extrema
    # exports while bounding each merged reduction vector to 256 KiB.
    _DATETIME_MONTH_CODE_DOMAIN = 32_768

    @classmethod
    def _q5_uses_native_spill(cls, group_domain: int) -> bool:
        return int(group_domain) >= cls._Q5_NATIVE_SPILL_GROUP_DOMAIN_THRESHOLD

    @staticmethod
    @contextmanager
    def _terminal_export(*, rows: int, columns: int):
        """Record one bounded terminal-export phase for a completed query."""
        from vibespatial.runtime.hotpath_trace import (
            attach_work_amplification,
            hotpath_stage,
            hotpath_timing_enabled,
        )

        timing = hotpath_timing_enabled()
        if timing:
            from vibespatial.cuda._runtime import get_cuda_runtime

            get_cuda_runtime().synchronize()
        with hotpath_stage(
            "spatialbench.terminal_export",
            category="emit",
            metadata={"terminal_export": True},
        ) as stage_metadata:
            try:
                yield
            finally:
                if timing:
                    get_cuda_runtime().synchronize()
                attach_work_amplification(
                    stage_metadata,
                    operation="spatialbench_terminal_export",
                    metric_family="materialization",
                    sums={
                        "output_rows": int(rows),
                        "output_columns": int(columns),
                        "public_frame_materializations": 1,
                        "diagnostic_synchronizations": 2 if timing else 0,
                    },
                    maxima={"output_rows": int(rows)},
                    physical_shape="bounded terminal result export",
                    consumer_kind="SpatialBench result serializer",
                    semantic_contract={
                        "selected_rows_bulk_exported": False,
                        "single_terminal_export_phase": True,
                    },
                )

    @staticmethod
    @contextmanager
    def _profile_stage(name: str, *, category: str):
        """Fence a diagnostic-only stage so queued GPU work cannot overlap it."""
        from vibespatial.runtime.hotpath_trace import (
            hotpath_stage,
            hotpath_timing_enabled,
        )

        timing = hotpath_timing_enabled()
        if timing:
            from vibespatial.cuda._runtime import get_cuda_runtime

            get_cuda_runtime().synchronize()
        with hotpath_stage(name, category=category) as stage_metadata:
            try:
                yield stage_metadata
            finally:
                if timing:
                    get_cuda_runtime().synchronize()

    def _spatial_frames(self, path, columns, geometry, *, batch_rows=None):
        """Expose a bounded scan/decode stage without claiming a GDS transport."""
        from vibespatial.runtime.hotpath_trace import (
            attach_work_amplification,
            hotpath_stage,
            hotpath_timing_enabled,
        )

        # Always select the budgeted dataset reader when it is available.  The
        # inherited compatibility plan switches at 100 source files, which
        # creates a physical-shape cliff at intermediate scale tiers.  Passing
        # the configured upper row target keeps every tier on the same
        # metadata- and live-memory-admitted chunk equations.
        resolved_batch_rows = batch_rows or self.scan_batch_rows or 2_000_000
        batches = iter(
            super()._spatial_frames(
                path,
                columns,
                geometry,
                batch_rows=resolved_batch_rows,
            )
        )
        while True:
            if hotpath_timing_enabled():
                from vibespatial.cuda._runtime import get_cuda_runtime

                get_cuda_runtime().synchronize()
            try:
                with hotpath_stage(
                    "spatialbench.scan_decode",
                    category="setup",
                    metadata={"projected_columns": len(columns)},
                ) as stage_metadata:
                    frame = next(batches)
                    if hotpath_timing_enabled():
                        from vibespatial.cuda._runtime import get_cuda_runtime

                        get_cuda_runtime().synchronize()
                    attach_work_amplification(
                        stage_metadata,
                        operation="spatialbench_scan_decode",
                        metric_family="io",
                        sums={"input_batches": 1, "input_rows": len(frame)},
                        maxima={
                            "batch_rows": len(frame),
                            "projected_columns": len(columns),
                        },
                        physical_shape="bounded GeoParquet scan and GeoArrow decode",
                        consumer_kind="native query batch",
                        semantic_contract={
                            "transport": "backend-reported-only",
                            "direct_gds_inferred": False,
                        },
                    )
            except StopIteration:
                return
            yield frame

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

    def q3(self, data_paths):
        """Reduce filtered trip metrics into device-resident month vectors."""
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
        accumulator = None
        group_count = self._DATETIME_MONTH_CODE_DOMAIN

        for trips in self._spatial_frames(
            data_paths["trip"],
            columns,
            "t_pickuploc",
        ):
            selected = trips.loc[trips.geometry.distance(polygon) <= 0.045]
            month_code = self._month_code(selected, "t_pickuptime")
            duration = selected.datetime_difference_seconds(
                "t_pickuptime",
                "t_dropofftime",
            )
            batch = self.gpd.dense_grouped_reduce(
                month_code,
                size=group_count,
                count_name="total_trips",
                sums={
                    "_distance_sum": selected["t_distance"].astype(float),
                    "_duration_sum": duration,
                    "_fare_sum": selected["t_fare"].astype(float),
                },
                out=accumulator,
            )
            accumulator = batch
            del (
                batch,
                duration,
                month_code,
                selected,
                trips,
            )

        if accumulator is None:
            return pd.DataFrame(
                columns=[
                    "pickup_month",
                    "total_trips",
                    "avg_distance",
                    "avg_duration",
                    "avg_fare",
                ]
            )

        # This is the single terminal host-export phase. Each transferred vector
        # is either fixed datetime-domain capacity or observed-month capacity;
        # no selected source rows cross it.
        with self._terminal_export(rows=group_count, columns=5):
            total_trips = accumulator["total_trips"]
            distance_sum = accumulator["_distance_sum"]
            duration_sum = accumulator["_duration_sum"]
            fare_sum = accumulator["_fare_sum"]
            count_values = total_trips.to_numpy(dtype=np.uint64, copy=False)
            observed = np.flatnonzero(count_values)
            distance_values = self.gpd.numeric_take(
                distance_sum,
                observed,
            ).to_numpy(dtype=np.float64, copy=False)
            duration_values = self.gpd.numeric_take(
                duration_sum,
                observed,
            ).to_numpy(dtype=np.float64, copy=False)
            fare_values = self.gpd.numeric_take(
                fare_sum,
                observed,
            ).to_numpy(dtype=np.float64, copy=False)
            observed_counts = count_values[observed]
            years, months = np.divmod(observed - 1, 12)
            return pd.DataFrame(
                {
                    "pickup_month": pd.to_datetime(
                        {"year": years, "month": months + 1, "day": 1}
                    ),
                    "total_trips": observed_counts,
                    "avg_distance": distance_values / observed_counts,
                    "avg_duration": duration_values / observed_counts,
                    "avg_fare": fare_values / observed_counts,
                }
            ).reset_index(drop=True)

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
            if group_counts is None:
                group_counts = self.gpd.dense_count(
                    packed,
                    size=group_domain,
                    dtype=np.uint32,
                    name="dropoff_count",
                )
            else:
                group_counts = self.gpd.dense_count(
                    packed,
                    size=group_domain,
                    dtype=np.uint32,
                    out=group_counts,
                )
            del month_code, packed, trips

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

    def q1(self, data_paths):
        """Keep distance-filtered winners resident across bounded scan batches."""
        center = Point(-111.7610, 34.8697)
        accumulator = None
        source_columns = ["t_tripkey", "t_pickuptime", "t_pickuploc"]
        rank_columns = ["__distance_rank", "__tripkey_rank"]
        for trips in self._spatial_frames(
            data_paths["trip"], source_columns, "t_pickuploc"
        ):
            from vibespatial.runtime.hotpath_trace import attach_work_amplification

            with self._profile_stage(
                "spatialbench.q1.scalar_distance",
                category="refine",
            ) as stage_metadata:
                distance = trips.geometry.distance(center)
                attach_work_amplification(
                    stage_metadata,
                    operation="spatialbench_q1_scalar_distance",
                    metric_family="metric",
                    sums={"input_rows": len(trips)},
                    maxima={"batch_rows": len(trips)},
                    physical_shape="point rows -> scalar distance expression",
                    consumer_kind="Q1 threshold predicate",
                )
            with self._profile_stage(
                "spatialbench.q1.threshold_filter",
                category="filter",
            ) as stage_metadata:
                metrics = trips.assign(
                    distance_to_center=distance,
                    __distance_rank=0 - distance,
                    __tripkey_rank=0 - trips["t_tripkey"],
                )
                candidates = metrics.loc[distance <= 0.45]
                attach_work_amplification(
                    stage_metadata,
                    operation="spatialbench_q1_threshold_filter",
                    metric_family="filter",
                    sums={
                        "input_rows": len(trips),
                        "selected_rows": len(candidates),
                    },
                    maxima={
                        "batch_rows": len(trips),
                        "selected_rows": len(candidates),
                    },
                    physical_shape="distance expression -> bounded candidate rowset",
                    consumer_kind="batch-local top-k",
                )
            accumulator = _streaming_topk(
                candidates,
                100,
                rank_columns,
                largest=True,
                out=accumulator,
            )
            del candidates, distance, metrics, trips

        columns = [
            "t_tripkey",
            "pickup_lon",
            "pickup_lat",
            "t_pickuptime",
            "distance_to_center",
        ]
        if accumulator is None:
            return pd.DataFrame(columns=columns)
        with self._profile_stage(
            "spatialbench.q1.result_take",
            category="emit",
        ) as stage_metadata:
            result = accumulator.drop(columns=rank_columns)
            payload = {
                "t_tripkey": result["t_tripkey"].to_numpy(),
                "pickup_lon": result.geometry.x.to_numpy(),
                "pickup_lat": result.geometry.y.to_numpy(),
                "t_pickuptime": result["t_pickuptime"].to_numpy(),
                "distance_to_center": result["distance_to_center"].to_numpy(),
            }
            attach_work_amplification(
                stage_metadata,
                operation="spatialbench_q1_result_take",
                metric_family="materialization",
                sums={
                    "output_rows": len(result),
                    "output_columns": len(columns),
                },
                maxima={"output_rows": len(result)},
                physical_shape="bounded winner state -> terminal column vectors",
                consumer_kind="Q1 terminal public frame",
            )
        with self._terminal_export(rows=len(result), columns=len(columns)):
            return pd.DataFrame(payload, columns=columns)

    def _q7_shard_topk(self, trips):
        """Return one bounded public candidate set for the inherited final merge."""
        pickup = trips.geometry
        dropoff = trips.set_geometry("t_dropoffloc").geometry
        line_distance = pickup.distance(dropoff, align=False) / 0.000009
        reported = trips["t_distance"].astype(float)
        metrics = trips.drop(columns=["t_dropoffloc"]).assign(
            reported_distance_m=reported,
            line_distance_m=line_distance,
        )
        metrics = metrics.loc[line_distance != 0.0]
        metrics = metrics.assign(
            detour_ratio=(
                metrics["reported_distance_m"] / metrics["line_distance_m"]
            ),
            __vibespatial_topk_tie_2=0 - metrics["t_tripkey"],
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
