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

import pandas as pd

try:
    from .geoparquet_public_api_queries import GeoParquetPublicApiQueries
except ImportError:  # SpatialBench loads this entrypoint as a standalone module.
    from geoparquet_public_api_queries import GeoParquetPublicApiQueries

import vibespatial as gpd

gpd.set_execution_mode(gpd.ExecutionMode.AUTO)


class VibeSpatialQueries(GeoParquetPublicApiQueries):
    """Public hybrid plan selected by physical workload shape."""

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
