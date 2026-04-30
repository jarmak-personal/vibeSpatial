import numpy as np
import pandas as pd

from vibespatial.api import GeoDataFrame
from vibespatial.api._native_results import (
    RelationIndexResult,
    RelationJoinExportResult,
    RelationJoinResult,
)
from vibespatial.api._native_state import get_native_state
from vibespatial.api.geometry_array import _check_crs, _crs_mismatch_warn
from vibespatial.api.tools._pair_cache import cache_intersection_pairs
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.provenance import (
    _r1_preconditions_met,
    provenance_rewrites_enabled,
    record_rewrite_event,
)
from vibespatial.runtime.residency import Residency
from vibespatial.spatial.indexing import build_flat_spatial_index
from vibespatial.spatial.query import (
    SpatialQueryExecution,
    build_owned_spatial_index,
    query_spatial_index,
    supports_owned_spatial_input,
)


def sjoin(
    left_df,
    right_df,
    how="inner",
    predicate="intersects",
    lsuffix="left",
    rsuffix="right",
    distance=None,
    on_attribute=None,
    **kwargs,
):
    """Spatial join of two GeoDataFrames.

    See the User Guide page :doc:`../../user_guide/mergingdata` for details.


    Parameters
    ----------
    left_df, right_df : GeoDataFrames
    how : string, default 'inner'
        The type of join:

        * 'left': use keys from left_df; retain only left_df geometry column
        * 'right': use keys from right_df; retain only right_df geometry column
        * 'inner': use intersection of keys from both dfs; retain only
          left_df geometry column
        * 'outer': use the union of keys from both dfs; retain a single
          active geometry column by preferring left geometries and filling
          unmatched right-only rows from the right geometry column
    predicate : string, default 'intersects'
        Binary predicate. Valid values are determined by the spatial index used.
        You can check the valid values in left_df or right_df as
        ``left_df.sindex.valid_query_predicates`` or
        ``right_df.sindex.valid_query_predicates``

        Available predicates include:

        * ``'intersects'``: True if geometries intersect (boundaries and interiors)
        * ``'within'``: True if left geometry is completely within right geometry
        * ``'contains'``: True if left geometry completely contains right geometry
        * ``'contains_properly'``: True if left geometry contains right geometry
          and their boundaries do not touch
        * ``'overlaps'``: True if geometries overlap but neither contains the other
        * ``'crosses':`` True if geometries cross (interiors intersect but neither
          contains the other, with intersection dimension less than max dimension)
        * ``'touches'``: True if geometries touch at boundaries but interiors don't
        * ``'covers'``: True if left geometry covers right geometry (every point of
          right is a point of left)
        * ``'covered_by'``: True if left geometry is covered by right geometry
        * ``'dwithin'``: True if geometries are within specified distance (requires
          distance parameter)
    lsuffix : string, default 'left'
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string, default 'right'
        Suffix to apply to overlapping column names (right GeoDataFrame).
    distance : number or array_like, optional
        Distance(s) around each input geometry within which to query the tree
        for the 'dwithin' predicate. If array_like, must be
        one-dimesional with length equal to length of left GeoDataFrame.
        Required if ``predicate='dwithin'``.
    on_attribute : string, list or tuple
        Column name(s) to join on as an additional join restriction on top
        of the spatial predicate. These must be found in both DataFrames.
        If set, observations are joined only if the predicate applies
        and values in specified columns match.

    Examples
    --------
    >>> import geodatasets
    >>> chicago = geopandas.read_file(
    ...     geodatasets.get_path("geoda.chicago_health")
    ... )
    >>> groceries = geopandas.read_file(
    ...     geodatasets.get_path("geoda.groceries")
    ... ).to_crs(chicago.crs)

    >>> chicago.head()  # doctest: +SKIP
        ComAreaID  ...                                           geometry
    0         35  ...  POLYGON ((-87.60914 41.84469, -87.60915 41.844...
    1         36  ...  POLYGON ((-87.59215 41.81693, -87.59231 41.816...
    2         37  ...  POLYGON ((-87.62880 41.80189, -87.62879 41.801...
    3         38  ...  POLYGON ((-87.60671 41.81681, -87.60670 41.816...
    4         39  ...  POLYGON ((-87.59215 41.81693, -87.59215 41.816...
    [5 rows x 87 columns]

    >>> groceries.head()  # doctest: +SKIP
        OBJECTID     Ycoord  ...  Category                         geometry
    0        16  41.973266  ...       NaN  MULTIPOINT (-87.65661 41.97321)
    1        18  41.696367  ...       NaN  MULTIPOINT (-87.68136 41.69713)
    2        22  41.868634  ...       NaN  MULTIPOINT (-87.63918 41.86847)
    3        23  41.877590  ...       new  MULTIPOINT (-87.65495 41.87783)
    4        27  41.737696  ...       NaN  MULTIPOINT (-87.62715 41.73623)
    [5 rows x 8 columns]

    >>> groceries_w_communities = geopandas.sjoin(groceries, chicago)
    >>> groceries_w_communities.head()  # doctest: +SKIP
       OBJECTID       community                           geometry
    0        16          UPTOWN  MULTIPOINT ((-87.65661 41.97321))
    1        18     MORGAN PARK  MULTIPOINT ((-87.68136 41.69713))
    2        22  NEAR WEST SIDE  MULTIPOINT ((-87.63918 41.86847))
    3        23  NEAR WEST SIDE  MULTIPOINT ((-87.65495 41.87783))
    4        27         CHATHAM  MULTIPOINT ((-87.62715 41.73623))
    [5 rows x 95 columns]

    See Also
    --------
    overlay : overlay operation resulting in a new geometry
    GeoDataFrame.sjoin : equivalent method

    Notes
    -----
    Every operation in GeoPandas is planar, i.e. the potential third
    dimension is not taken into account.
    """
    if kwargs:
        first = next(iter(kwargs.keys()))
        raise TypeError(f"sjoin() got an unexpected keyword argument '{first}'")

    on_attribute = _maybe_make_list(on_attribute)

    _basic_checks(
        left_df,
        right_df,
        how,
        lsuffix,
        rsuffix,
        on_attribute=on_attribute,
        allowed_hows=["left", "right", "inner", "outer"],
    )

    export_result, query_implementation, query_execution = _sjoin_export_result(
        left_df,
        right_df,
        how,
        predicate,
        distance,
        lsuffix,
        rsuffix,
        on_attribute=on_attribute,
        return_device=True,
    )
    # Use the actual execution mode from the query engine when available.
    if query_execution is not None:
        sjoin_selected = query_execution.selected
    else:
        sjoin_selected = ExecutionMode.CPU
    record_dispatch_event(
        surface="geopandas.tools.sjoin",
        operation="sjoin",
        implementation=query_implementation,
        reason=_sjoin_dispatch_reason(query_implementation),
        detail=f"how={how}, predicate={predicate}, rows_left={len(left_df)}, rows_right={len(right_df)}",
        selected=sjoin_selected,
    )

    return export_result.to_geodataframe()


def _maybe_make_list(obj):
    if isinstance(obj, tuple):
        return list(obj)
    if obj is not None and not isinstance(obj, list):
        return [obj]
    return obj


def _sjoin_relation_result(
    left_df,
    right_df,
    predicate,
    distance,
    *,
    on_attribute=None,
    return_device: bool = False,
):
    """Build the native relation result before any DataFrame materialization."""
    indices, query_implementation, query_execution = _geom_predicate_query(
        left_df,
        right_df,
        predicate,
        distance,
        on_attribute=on_attribute,
        return_device=return_device,
    )
    return RelationJoinResult(RelationIndexResult(*indices)), query_implementation, query_execution


def _sjoin_export_result(
    left_df,
    right_df,
    how,
    predicate,
    distance,
    lsuffix,
    rsuffix,
    *,
    on_attribute=None,
    return_device: bool = False,
):
    """Build the deferred public sjoin export result before GeoDataFrame assembly."""
    native_result, query_implementation, query_execution = _sjoin_relation_result(
        left_df,
        right_df,
        predicate,
        distance,
        on_attribute=on_attribute,
        return_device=return_device,
    )
    if predicate == "intersects" and distance is None and not on_attribute:
        cache_intersection_pairs(
            left_df,
            right_df,
            native_result.relation.left_indices,
            native_result.relation.right_indices,
        )
    return (
        RelationJoinExportResult(
            relation_result=native_result,
            left_df=left_df,
            right_df=right_df,
            how=how,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            on_attribute=on_attribute,
            predicate=predicate,
        ),
        query_implementation,
        query_execution,
    )


def _basic_checks(left_df, right_df, how, lsuffix, rsuffix, on_attribute=None, allowed_hows=None):
    """Check the validity of join input parameters.

    `how` must be one of the valid options.
    `'index_'` concatenated with `lsuffix` or `rsuffix` must not already
    exist as columns in the left or right data frames.

    Parameters
    ----------
    left_df : GeoDataFrame
    right_df : GeoData Frame
    how : str, one of 'left', 'right', 'inner', 'outer'
        join type
    lsuffix : str
        left index suffix
    rsuffix : str
        right index suffix
    on_attribute : list, default None
        list of column names to merge on along with geometry
    """
    if not isinstance(left_df, GeoDataFrame):
        raise ValueError(f"'left_df' should be GeoDataFrame, got {type(left_df)}")

    if not isinstance(right_df, GeoDataFrame):
        raise ValueError(f"'right_df' should be GeoDataFrame, got {type(right_df)}")

    if allowed_hows is None:
        allowed_hows = ["left", "right", "inner"]
    if how not in allowed_hows:
        raise ValueError(f'`how` was "{how}" but is expected to be in {allowed_hows}')

    if not _check_crs(left_df, right_df):
        _crs_mismatch_warn(left_df, right_df, stacklevel=4)

    if on_attribute:
        for attr in on_attribute:
            if (attr not in left_df) and (attr not in right_df):
                raise ValueError(
                    f"Expected column {attr} is missing from both of the dataframes."
                )
            if attr not in left_df:
                raise ValueError(
                    f"Expected column {attr} is missing from the left dataframe."
                )
            if attr not in right_df:
                raise ValueError(
                    f"Expected column {attr} is missing from the right dataframe."
                )
            if attr in (left_df.geometry.name, right_df.geometry.name):
                raise ValueError(
                    "Active geometry column cannot be used as an input "
                    "for on_attribute parameter."
                )


def _sjoin_dispatch_reason(query_implementation: str) -> str:
    if query_implementation == "native_spatial_index":
        return "NativeSpatialIndex produced relation pairs for sjoin"
    if query_implementation == "owned_spatial_query":
        return "repo-owned spatial query assembled candidate pairs for sjoin"
    return "Shapely STRtree remains the current host query path for this sjoin surface"


def _query_with_native_spatial_index(
    left_df,
    right_df,
    predicate,
    distance,
    *,
    return_device: bool = False,
):
    """Run sjoin through attached NativeFrameState -> NativeSpatialIndex.

    Physical shape: candidate/predicate pair generation over a reusable native
    index.  Native input carriers are the attached left/right
    ``NativeFrameState`` objects and a transient ``NativeSpatialIndex`` over the
    right frame.  The native output carrier is ``NativeRelation``; public sjoin
    callers receive the same index-pair contract after the explicit export
    boundary.
    """
    if predicate not in {
        None,
        "contains",
        "contains_properly",
        "covered_by",
        "covers",
        "crosses",
        "dwithin",
        "intersects",
        "overlaps",
        "touches",
        "within",
    }:
        return None

    left_state = get_native_state(left_df)
    right_state = get_native_state(right_df)
    if left_state is None or right_state is None:
        return None

    left_owned = getattr(left_state.geometry, "owned", None)
    right_owned = getattr(right_state.geometry, "owned", None)
    if left_owned is None or right_owned is None:
        return None

    native_index = _native_spatial_index_for_sjoin_right(
        right_df,
        right_state,
        right_owned,
    )
    relation, execution = native_index.query_relation(
        left_state,
        predicate=predicate,
        sort=False,
        distance=distance,
        query_token=left_state.lineage_token,
        return_device=return_device,
        return_metadata=True,
    )
    return (relation.left_indices, relation.right_indices), execution


def _native_spatial_index_for_sjoin_right(right_df, right_state, right_owned):
    """Return reusable native index state for right-side sjoin geometry.

    Physical shape: reusable spatial-index execution state feeding relation
    consume.  Native input carriers are the right ``NativeFrameState`` and, when
    already valid, the cached public ``SpatialIndex``/``FlatSpatialIndex``.  The
    native output carrier is ``NativeSpatialIndex`` with the right lineage token.
    """
    geometry_values = getattr(getattr(right_df, "geometry", None), "values", None)
    cached_public_sindex = getattr(geometry_values, "_sindex", None)
    if (
        cached_public_sindex is not None
        and hasattr(cached_public_sindex, "query_relation")
        and getattr(geometry_values, "_owned", None) is right_owned
    ):
        return cached_public_sindex._native_spatial_index_for_query(
            source_token=right_state.lineage_token,
        )

    flat_index = getattr(geometry_values, "_owned_flat_sindex", None)
    if (
        flat_index is not None
        and getattr(flat_index, "geometry_array", None) is right_owned
        and hasattr(flat_index, "to_native_spatial_index")
    ):
        return flat_index.to_native_spatial_index(
            source_token=right_state.lineage_token,
        )

    selection = plan_dispatch_selection(
        kernel_name="flat_index_build",
        kernel_class=KernelClass.COARSE,
        row_count=right_state.row_count,
        requested_mode=ExecutionMode.AUTO,
        gpu_available=has_gpu_runtime(),
        current_residency=(
            Residency.DEVICE
            if right_owned.residency is Residency.DEVICE
            else Residency.HOST
        ),
    )
    flat_index = build_flat_spatial_index(
        right_owned,
        runtime_selection=selection.runtime_selection,
    )
    return flat_index.to_native_spatial_index(
        source_token=right_state.lineage_token,
    )


def _query_with_owned_spatial_index(
    left_df,
    right_df,
    predicate,
    distance,
    *,
    return_device: bool = False,
):
    """Run the owned spatial query path for sjoin.

    ``return_device`` is intentionally opt-in so public ``sjoin`` preserves
    the exact GeoPandas-shaped host boundary while private ADR-0044 callers can
    keep admitted relation-pair exports device-resident.
    """
    tree_geometry = right_df.geometry
    query_geometry = left_df.geometry
    tree_positions = None
    query_positions = None

    def _non_empty_positions(geometry) -> np.ndarray:
        import shapely

        values = getattr(geometry, "values", geometry)
        data = np.asarray(getattr(values, "_data", values), dtype=object)
        if data.size == 0:
            return np.empty(0, dtype=np.intp)
        non_null = np.asarray([geom is not None for geom in data], dtype=bool)
        non_empty = np.zeros(data.size, dtype=bool)
        if np.any(non_null):
            non_empty[non_null] = ~np.asarray(shapely.is_empty(data[non_null]), dtype=bool)
        return np.flatnonzero(non_empty).astype(np.intp, copy=False)

    if hasattr(tree_geometry, "values") and hasattr(tree_geometry.values, "supports_owned_spatial_input"):
        tree_supported = tree_geometry.values.supports_owned_spatial_input()
    else:
        tree_supported = supports_owned_spatial_input(tree_geometry)
    if hasattr(query_geometry, "values") and hasattr(query_geometry.values, "supports_owned_spatial_input"):
        query_supported = query_geometry.values.supports_owned_spatial_input()
    else:
        query_supported = supports_owned_spatial_input(query_geometry)
    if not tree_supported or not query_supported:
        tree_positions = _non_empty_positions(tree_geometry)
        query_positions = _non_empty_positions(query_geometry)
        if tree_positions.size == 0 or query_positions.size == 0:
            empty = np.asarray([], dtype=np.intp)
            execution = SpatialQueryExecution(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU,
                implementation="owned_empty_geometry_filter",
                reason="empty geometries were filtered before owned spatial query",
            )
            return (empty, empty), execution
        if tree_positions.size != len(tree_geometry):
            tree_geometry = tree_geometry.iloc[tree_positions]
        else:
            tree_positions = None
        if query_positions.size != len(query_geometry):
            query_geometry = query_geometry.iloc[query_positions]
        else:
            query_positions = None
        if hasattr(tree_geometry, "values") and hasattr(tree_geometry.values, "supports_owned_spatial_input"):
            tree_supported = tree_geometry.values.supports_owned_spatial_input()
        else:
            tree_supported = supports_owned_spatial_input(tree_geometry)
        if hasattr(query_geometry, "values") and hasattr(query_geometry.values, "supports_owned_spatial_input"):
            query_supported = query_geometry.values.supports_owned_spatial_input()
        else:
            query_supported = supports_owned_spatial_input(query_geometry)
    if not tree_supported or not query_supported:
        return None
    if predicate not in {
        None,
        "contains",
        "contains_properly",
        "covered_by",
        "covers",
        "crosses",
        "dwithin",
        "intersects",
        "overlaps",
        "touches",
        "within",
    }:
        return None

    if hasattr(tree_geometry, "values") and hasattr(tree_geometry.values, "owned_flat_sindex"):
        tree_owned, flat_index = tree_geometry.values.owned_flat_sindex()
    else:
        if hasattr(tree_geometry, "values") and hasattr(tree_geometry.values, "_data"):
            tree_values = np.asarray(tree_geometry.values._data, dtype=object)
        elif hasattr(tree_geometry, "_data"):
            tree_values = np.asarray(tree_geometry._data, dtype=object)
        else:
            tree_values = np.asarray(tree_geometry, dtype=object)
        tree_owned, flat_index = build_owned_spatial_index(tree_values)
    if hasattr(query_geometry, "values") and hasattr(query_geometry.values, "to_owned"):
        query_input = query_geometry.values.to_owned()
    elif hasattr(query_geometry, "to_owned"):
        query_input = query_geometry.to_owned()
    else:
        query_input = query_geometry
    indices, execution = query_spatial_index(
        tree_owned,
        flat_index,
        query_input,
        predicate=predicate,
        sort=False,
        distance=distance,
        output_format="indices",
        return_metadata=True,
        return_device=return_device,
    )
    if hasattr(indices, "d_left_idx") and hasattr(indices, "d_right_idx"):
        left_idx = indices.d_left_idx
        right_idx = indices.d_right_idx
        if query_positions is not None or tree_positions is not None:
            import cupy as cp

            if query_positions is not None:
                left_idx = cp.asarray(query_positions, dtype=left_idx.dtype)[left_idx]
            if tree_positions is not None:
                right_idx = cp.asarray(tree_positions, dtype=right_idx.dtype)[right_idx]
        return (left_idx, right_idx), execution
    if indices.ndim == 1:
        empty = np.asarray([], dtype=np.intp)
        right_idx = indices.astype(np.intp, copy=False)
        if tree_positions is not None:
            right_idx = tree_positions[right_idx]
        return (empty, right_idx), execution
    left_idx = indices[0].astype(np.intp, copy=False)
    right_idx = indices[1].astype(np.intp, copy=False)
    if query_positions is not None:
        left_idx = query_positions[left_idx]
    if tree_positions is not None:
        right_idx = tree_positions[right_idx]
    return (left_idx, right_idx), execution


def _is_device_array(values) -> bool:
    return hasattr(values, "__cuda_array_interface__")


def _is_pylibcudf_column(values) -> bool:
    type_ = type(values)
    return bool(
        type_.__module__.startswith("pylibcudf.")
        and type_.__name__ == "Column"
        and hasattr(values, "size")
        and hasattr(values, "type")
    )


def _is_admitted_on_attribute_arrow_type(dtype) -> bool:
    import pyarrow as pa

    return bool(
        pa.types.is_integer(dtype)
        or pa.types.is_floating(dtype)
        or pa.types.is_boolean(dtype)
        or pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
        or pa.types.is_dictionary(dtype)
        or pa.types.is_temporal(dtype)
    )


def _device_columns_from_native_attributes(attributes, on_attribute):
    from vibespatial.api._native_result_core import NativeAttributeTable

    try:
        table = NativeAttributeTable.from_value(attributes)
    except TypeError:
        return None
    if table.device_table is None or not hasattr(table.device_table, "columns"):
        return None

    requested = tuple(dict.fromkeys(on_attribute))
    column_positions = {name: index for index, name in enumerate(tuple(table.columns))}
    if len(column_positions) != len(tuple(table.columns)) or any(
        column not in column_positions for column in requested
    ):
        return None

    policies = table.device_column_policies(requested)
    if any(
        (policy := policies.get(column)) is None
        or not policy.can_project_take
        or int(policy.null_count) != 0
        for column in requested
    ):
        return None

    source_columns = table.device_table.columns()
    return {
        column: source_columns[column_positions[column]]
        for column in requested
    }


def _device_columns_from_public_frame(frame, on_attribute):
    requested = tuple(dict.fromkeys(on_attribute))
    try:
        import pyarrow as pa
        import pylibcudf as plc
    except ModuleNotFoundError:
        return None

    arrays = []
    for column in requested:
        if column not in frame:
            return None
        series = frame[column]
        if not isinstance(series, pd.Series):
            return None
        if series.hasnans:
            return None
        try:
            arrow_array = pa.array(series, from_pandas=True)
        except (pa.ArrowException, TypeError, ValueError):
            return None
        if arrow_array.null_count != 0 or not _is_admitted_on_attribute_arrow_type(
            arrow_array.type
        ):
            return None
        arrays.append(arrow_array)

    physical_names = [f"__vibespatial_on_attribute_{index}" for index in range(len(arrays))]
    try:
        device_table = plc.Table.from_arrow(
            pa.Table.from_arrays(arrays, names=physical_names)
        )
    except (TypeError, ValueError):
        return None
    source_columns = device_table.columns()
    return {
        column: source_columns[index]
        for index, column in enumerate(requested)
    }


def _shared_attribute_device_columns(frame, state, on_attribute):
    if state is not None:
        columns = _device_columns_from_native_attributes(
            state.attributes,
            on_attribute,
        )
        if columns is not None:
            return columns
    return _device_columns_from_public_frame(frame, on_attribute)


def _native_shared_attribute_columns(
    left_df,
    right_df,
    on_attribute,
    *,
    require_device: bool,
):
    """Return admitted native attribute columns for a shared-attribute join."""
    if not on_attribute:
        return None
    left_state = get_native_state(left_df)
    right_state = get_native_state(right_df)
    if left_state is None or right_state is None:
        if require_device:
            left_columns = _device_columns_from_public_frame(left_df, on_attribute)
            right_columns = _device_columns_from_public_frame(right_df, on_attribute)
            if left_columns is not None and right_columns is not None:
                return left_columns, right_columns
        return None

    if require_device:
        left_columns = _shared_attribute_device_columns(
            left_df,
            left_state,
            on_attribute,
        )
        right_columns = _shared_attribute_device_columns(
            right_df,
            right_state,
            on_attribute,
        )
        if left_columns is None or right_columns is None:
            return None
        if (
            any(not _is_pylibcudf_column(values) for values in left_columns.values())
            or any(not _is_pylibcudf_column(values) for values in right_columns.values())
        ):
            return None
        return left_columns, right_columns

    from vibespatial.api._native_result_core import NativeAttributeTable

    left_columns = NativeAttributeTable.from_value(
        left_state.attributes
    ).numeric_column_arrays(on_attribute)
    right_columns = NativeAttributeTable.from_value(
        right_state.attributes
    ).numeric_column_arrays(on_attribute)
    if left_columns is None or right_columns is None:
        return None
    return left_columns, right_columns


def _filter_relation_by_native_shared_attributes(
    relation,
    left_df,
    right_df,
    on_attribute,
    *,
    require_device: bool,
    attribute_columns=None,
):
    columns = attribute_columns
    if columns is None:
        columns = _native_shared_attribute_columns(
            left_df,
            right_df,
            on_attribute,
            require_device=require_device,
        )
    if columns is None:
        return None
    if require_device and (
        not _is_device_array(relation.left_indices)
        or not _is_device_array(relation.right_indices)
        or (
            relation.distances is not None
            and not _is_device_array(relation.distances)
        )
    ):
        return None
    left_columns, right_columns = columns
    try:
        return relation.filter_by_equal_columns(left_columns, right_columns)
    except ValueError:
        return None


def _filter_indices_by_native_shared_attributes(
    left_df,
    right_df,
    left_indices,
    right_indices,
    on_attribute,
    *,
    attribute_columns=None,
):
    from vibespatial.api._native_relation import NativeRelation

    relation = NativeRelation(
        left_indices=left_indices,
        right_indices=right_indices,
        left_row_count=len(left_df),
        right_row_count=len(right_df),
    )
    filtered = _filter_relation_by_native_shared_attributes(
        relation,
        left_df,
        right_df,
        on_attribute,
        require_device=True,
        attribute_columns=attribute_columns,
    )
    if filtered is None:
        return None
    return filtered.left_indices, filtered.right_indices


def _geom_predicate_query(
    left_df,
    right_df,
    predicate,
    distance,
    on_attribute=None,
    *,
    return_device: bool = False,
):
    """Compute geometric comparisons and get matching indices.

    Parameters
    ----------
    left_df : GeoDataFrame
    right_df : GeoDataFrame
    predicate : string
        Binary predicate to query.
    on_attribute: list, default None
        list of column names to merge on along with geometry


    Returns
    -------
    DataFrame
        DataFrame with matching indices in
        columns named `_key_left` and `_key_right`.
    """
    original_predicate = predicate

    def _rewrite_buffer_intersects_to_dwithin(df, *, side: str):
        ga = df.geometry.values
        tag = getattr(ga, "_provenance", None)
        if tag is None or tag.operation != "buffer" or not _r1_preconditions_met(tag):
            return None
        buf_distance = tag.get_param("distance")
        source_ga = tag.source_array
        if source_ga is None or len(source_ga) != len(ga):
            return None

        from vibespatial.api.geoseries import GeoSeries

        rewritten = df.copy()
        rewritten[df.geometry.name] = GeoSeries(
            source_ga,
            index=df.index,
            crs=df.crs,
        )
        record_rewrite_event(
            rule_name="R2_sjoin_buffer_intersects_to_dwithin",
            surface="geopandas.tools.sjoin",
            original_operation="sjoin(buffer, intersects)",
            rewritten_operation="sjoin(dwithin)",
            reason="sjoin(buffer(r, X), Y, intersects) == sjoin(X, Y, dwithin, r)",
            detail=f"buffer_distance={buf_distance}, side={side}, rows={len(df)}",
        )
        return rewritten, buf_distance

    # R2: sjoin(buffer(r, X), Y, "intersects") -> sjoin(X, Y, "dwithin", r)
    #
    # This rewrite is currently limited to the left side. A symmetric
    # right-buffer rewrite looks valid algebraically, but the existing
    # polygon-left / point-right dwithin engine under-selects pairs on
    # real workloads (for example redevelopment_screening at 10k), so we
    # leave the right-buffer case on the proven intersects path until the
    # dwithin query orientation is fixed end to end.
    if (
        provenance_rewrites_enabled()
        and predicate == "intersects"
        and distance is None
    ):
        left_rewrite = _rewrite_buffer_intersects_to_dwithin(left_df, side="left")
        if left_rewrite is not None:
            left_df, distance = left_rewrite
            predicate = "dwithin"

    # Prefer attached native frame/index state when both sides have a valid
    # NativeFrameState.  Otherwise, keep the older owned spatial-query path.
    native_attribute_columns = None
    if on_attribute and return_device:
        native_attribute_columns = _native_shared_attribute_columns(
            left_df,
            right_df,
            on_attribute,
            require_device=True,
        )
        if native_attribute_columns is None:
            record_fallback_event(
                surface="geopandas.sjoin",
                requested=ExecutionMode.GPU,
                selected=ExecutionMode.CPU,
                reason=(
                    "native on_attribute filtering requires all-valid "
                    "device-compatible columns"
                ),
                detail=f"on_attribute={on_attribute!r}",
                pipeline="spatial_join/native_on_attribute",
                d2h_transfer=True,
            )
    query_return_device = return_device and (
        not on_attribute or native_attribute_columns is not None
    )
    native_result = _query_with_native_spatial_index(
        left_df,
        right_df,
        predicate,
        distance,
        return_device=query_return_device,
    )
    if native_result is not None:
        (l_idx, r_idx), native_execution = native_result
        query_implementation = "native_spatial_index"
        query_execution = native_execution
    else:
        # Always try owned spatial query before STRtree for all join types, not
        # just outer.  The owned engine handles 'within' directly so we pass the
        # current predicate (which may have been rewritten by R2 above).
        owned_result = _query_with_owned_spatial_index(
            left_df,
            right_df,
            predicate,
            distance,
            return_device=query_return_device,
        )
        if owned_result is not None:
            (l_idx, r_idx), owned_execution = owned_result
            query_implementation = "owned_spatial_query"
            query_execution = owned_execution
        else:
            query_execution = None
            # Owned path unavailable – fall back to sindex.query.
            if predicate == "within":
                # within is implemented as the inverse of contains
                # contains is a faster predicate
                # see discussion at https://github.com/geopandas/geopandas/pull/1421
                predicate = "contains"
                sindex = left_df.sindex
                input_geoms = right_df.geometry
            else:
                sindex = right_df.sindex
                input_geoms = left_df.geometry

            if sindex:
                l_idx, r_idx = sindex.query(
                    input_geoms, predicate=predicate, sort=False, distance=distance
                )
                query_implementation = "strtree_host"
            else:
                l_idx, r_idx = np.array([], dtype=np.intp), np.array([], dtype=np.intp)
                query_implementation = "owned_spatial_query"

            if (
                original_predicate == "within"
                and query_implementation != "owned_spatial_query"
            ):
                # within is implemented as the inverse of contains
                # flip back the results
                r_idx, l_idx = l_idx, r_idx
                indexer = np.lexsort((r_idx, l_idx))
                l_idx = l_idx[indexer]
                r_idx = r_idx[indexer]

    if on_attribute:
        filtered_indices = None
        if native_attribute_columns is not None:
            filtered_indices = _filter_indices_by_native_shared_attributes(
                left_df,
                right_df,
                l_idx,
                r_idx,
                on_attribute,
                attribute_columns=native_attribute_columns,
            )
        if filtered_indices is not None:
            l_idx, r_idx = filtered_indices
        else:
            if native_attribute_columns is not None and return_device:
                record_fallback_event(
                    surface="geopandas.sjoin",
                    requested=ExecutionMode.GPU,
                    selected=ExecutionMode.CPU,
                    reason="native on_attribute relation filtering failed",
                    detail=f"on_attribute={on_attribute!r}",
                    pipeline="spatial_join/native_on_attribute",
                    d2h_transfer=True,
                )
            for attr in on_attribute:
                (l_idx, r_idx), _ = _filter_shared_attribute(
                    left_df, right_df, l_idx, r_idx, attr
                )

    return (l_idx, r_idx), query_implementation, query_execution


def _frame_join_from_relation_result(
    left_df,
    right_df,
    relation_result: RelationIndexResult,
    distances,
    how,
    lsuffix,
    rsuffix,
    predicate,
    on_attribute=None,
):
    """Compatibility wrapper around the ADR-0042 native join export seam."""
    native_result = RelationJoinResult(relation_result, distances=distances)
    return native_result.materialize(
        left_df,
        right_df,
        how=how,
        lsuffix=lsuffix,
        rsuffix=rsuffix,
        on_attribute=on_attribute,
    )


def _frame_join(
    left_df,
    right_df,
    indices,
    distances,
    how,
    lsuffix,
    rsuffix,
    predicate,
    on_attribute=None,
):
    """Compatibility wrapper around native relation-join materialization."""
    native_result = RelationJoinResult(
        RelationIndexResult(*indices),
        distances=distances,
    )
    return native_result.materialize(
        left_df,
        right_df,
        how=how,
        lsuffix=lsuffix,
        rsuffix=rsuffix,
        on_attribute=on_attribute,
    )


def _nearest_query(
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    max_distance: float,
    how: str,
    return_distance: bool,
    exclusive: bool,
    on_attribute: list | None = None,
):
    # use the opposite of the join direction for the index
    use_left_as_sindex = how == "right"
    if use_left_as_sindex:
        sindex = left_df.sindex
        query = right_df.geometry
    else:
        sindex = right_df.sindex
        query = left_df.geometry
    nearest_selected = ExecutionMode.CPU
    if sindex:
        res, nearest_selected = sindex.nearest(
            query,
            return_all=True,
            max_distance=max_distance,
            return_distance=return_distance,
            exclusive=exclusive,
            _return_execution_mode=True,
        )
        if return_distance:
            (input_idx, tree_idx), distances = res
        else:
            (input_idx, tree_idx) = res
            distances = None
        if use_left_as_sindex:
            l_idx, r_idx = tree_idx, input_idx
            sort_order = np.argsort(l_idx, kind="stable")
            l_idx, r_idx = l_idx[sort_order], r_idx[sort_order]
            if distances is not None:
                distances = distances[sort_order]
        else:
            l_idx, r_idx = input_idx, tree_idx
    else:
        # when sindex is empty / has no valid geometries
        l_idx, r_idx = np.array([], dtype=np.intp), np.array([], dtype=np.intp)
        if return_distance:
            distances = np.array([], dtype=np.float64)
        else:
            distances = None

    if on_attribute:
        for attr in on_attribute:
            (l_idx, r_idx), shared_attribute_rows = _filter_shared_attribute(
                left_df, right_df, l_idx, r_idx, attr
            )
            distances = distances[shared_attribute_rows]

    return (l_idx, r_idx), distances, nearest_selected


def _stable_sort_nearest_pairs_by_left(left_indices, right_indices, distances):
    """Preserve GeoPandas right-nearest ordering without host materialization."""
    if hasattr(left_indices, "__cuda_array_interface__"):
        import cupy as cp

        order = cp.argsort(left_indices, kind="stable")
    else:
        order = np.argsort(left_indices, kind="stable")
    left_indices = left_indices[order]
    right_indices = right_indices[order]
    if distances is not None:
        distances = distances[order]
    return left_indices, right_indices, distances


def _nearest_native_relation_result(
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    max_distance: float,
    how: str,
    exclusive: bool,
    *,
    on_attribute: list | None = None,
):
    """Build an admitted nearest ``NativeRelation`` without public pair export."""
    use_left_as_sindex = how == "right"
    sindex = left_df.sindex if use_left_as_sindex else right_df.sindex
    if not sindex or not hasattr(sindex, "nearest_relation"):
        return None, ExecutionMode.CPU

    left_state = get_native_state(left_df)
    right_state = get_native_state(right_df)
    if on_attribute and _native_shared_attribute_columns(
        left_df,
        right_df,
        on_attribute,
        require_device=True,
    ) is None:
        return None, ExecutionMode.CPU

    query_geometry = right_df.geometry if use_left_as_sindex else left_df.geometry
    source_state = left_state if use_left_as_sindex else right_state
    query_state = right_state if use_left_as_sindex else left_state
    source_df = left_df if use_left_as_sindex else right_df
    query_df = right_df if use_left_as_sindex else left_df
    native_relation, selected = sindex.nearest_relation(
        query_geometry,
        return_all=True,
        max_distance=max_distance,
        exclusive=exclusive,
        source_token=(
            source_state.lineage_token
            if source_state is not None
            else f"gdf:{id(source_df)}"
        ),
        query_token=(
            query_state.lineage_token
            if query_state is not None
            else f"gdf:{id(query_df)}"
        ),
        query_row_count=query_state.row_count if query_state is not None else len(query_df),
    )
    if native_relation is None:
        return None, selected

    left_indices = native_relation.left_indices
    right_indices = native_relation.right_indices
    distances = native_relation.distances
    if use_left_as_sindex:
        left_indices, right_indices, distances = _stable_sort_nearest_pairs_by_left(
            native_relation.right_indices,
            native_relation.left_indices,
            native_relation.distances,
        )

    from vibespatial.api._native_relation import NativeRelation

    relation = NativeRelation(
        left_indices=left_indices,
        right_indices=right_indices,
        left_token=left_state.lineage_token if left_state is not None else None,
        right_token=right_state.lineage_token if right_state is not None else None,
        predicate="nearest",
        distances=distances,
        left_row_count=left_state.row_count if left_state is not None else len(left_df),
        right_row_count=(
            right_state.row_count if right_state is not None else len(right_df)
        ),
        sorted_by_left=use_left_as_sindex
        or bool(getattr(native_relation, "sorted_by_left", False)),
    )
    if on_attribute:
        filtered_relation = _filter_relation_by_native_shared_attributes(
            relation,
            left_df,
            right_df,
            on_attribute,
            require_device=True,
        )
        if filtered_relation is None:
            return None, ExecutionMode.CPU
        relation = filtered_relation

    return (
        RelationJoinResult(
            RelationIndexResult(
                relation.left_indices,
                relation.right_indices,
            ),
            distances=relation.distances,
        ),
        selected,
    )


def _filter_shared_attribute(left_df, right_df, l_idx, r_idx, attribute):
    """Return the indices for the left and right dataframe that share the same entry
    in the attribute column.

    Also returns a Boolean `shared_attribute_rows` for rows with the same entry.
    """
    shared_attribute_rows = (
        left_df[attribute].iloc[l_idx].values == right_df[attribute].iloc[r_idx].values
    )

    l_idx = l_idx[shared_attribute_rows]
    r_idx = r_idx[shared_attribute_rows]
    return (l_idx, r_idx), shared_attribute_rows


def _sjoin_nearest_relation_result(
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    max_distance: float,
    how: str,
    return_distance: bool,
    exclusive: bool,
    *,
    on_attribute: list | None = None,
):
    """Build the native nearest-join relation result before host export."""
    native_result, selected = _nearest_native_relation_result(
        left_df,
        right_df,
        max_distance,
        how,
        exclusive,
        on_attribute=on_attribute,
    )
    if native_result is not None:
        return native_result, selected

    indices, distances, nearest_selected = _nearest_query(
        left_df,
        right_df,
        max_distance,
        how,
        return_distance,
        exclusive,
        on_attribute=on_attribute,
    )
    return RelationJoinResult(
        RelationIndexResult(*indices),
        distances=distances,
    ), nearest_selected


def sjoin_nearest(
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    how: str = "inner",
    max_distance: float | None = None,
    lsuffix: str = "left",
    rsuffix: str = "right",
    distance_col: str | None = None,
    exclusive: bool = False,
) -> GeoDataFrame:
    """Spatial join of two GeoDataFrames based on the distance between their geometries.

    Results will include multiple output records for a single input record
    where there are multiple equidistant nearest or intersected neighbors.

    Distance is calculated in CRS units and can be returned using the
    `distance_col` parameter.

    See the User Guide page
    https://geopandas.readthedocs.io/en/latest/docs/user_guide/mergingdata.html
    for more details.


    Parameters
    ----------
    left_df, right_df : GeoDataFrames
    how : string, default 'inner'
        The type of join:

        * 'left': use keys from left_df; retain only left_df geometry column
        * 'right': use keys from right_df; retain only right_df geometry column
        * 'inner': use intersection of keys from both dfs; retain only
          left_df geometry column
    max_distance : float, default None
        Maximum distance within which to query for nearest geometry.
        Must be greater than 0.
        The max_distance used to search for nearest items in the tree may have a
        significant impact on performance by reducing the number of input
        geometries that are evaluated for nearest items in the tree.
    lsuffix : string, default 'left'
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string, default 'right'
        Suffix to apply to overlapping column names (right GeoDataFrame).
    distance_col : string, default None
        If set, save the distances computed between matching geometries under a
        column of this name in the joined GeoDataFrame.
    exclusive : bool, default False
        If True, the nearest geometries that are equal to the input geometry
        will not be returned, default False.

    Examples
    --------
    >>> import geodatasets
    >>> groceries = geopandas.read_file(
    ...     geodatasets.get_path("geoda.groceries")
    ... )
    >>> chicago = geopandas.read_file(
    ...     geodatasets.get_path("geoda.chicago_health")
    ... ).to_crs(groceries.crs)

    >>> chicago.head()  # doctest: +SKIP
       ComAreaID  ...                                           geometry
    0         35  ...  POLYGON ((-87.60914 41.84469, -87.60915 41.844...
    1         36  ...  POLYGON ((-87.59215 41.81693, -87.59231 41.816...
    2         37  ...  POLYGON ((-87.62880 41.80189, -87.62879 41.801...
    3         38  ...  POLYGON ((-87.60671 41.81681, -87.60670 41.816...
    4         39  ...  POLYGON ((-87.59215 41.81693, -87.59215 41.816...
    [5 rows x 87 columns]

    >>> groceries.head()  # doctest: +SKIP
       OBJECTID     Ycoord  ...  Category                           geometry
    0        16  41.973266  ...       NaN  MULTIPOINT ((-87.65661 41.97321))
    1        18  41.696367  ...       NaN  MULTIPOINT ((-87.68136 41.69713))
    2        22  41.868634  ...       NaN  MULTIPOINT ((-87.63918 41.86847))
    3        23  41.877590  ...       new  MULTIPOINT ((-87.65495 41.87783))
    4        27  41.737696  ...       NaN  MULTIPOINT ((-87.62715 41.73623))
    [5 rows x 8 columns]

    >>> groceries_w_communities = geopandas.sjoin_nearest(groceries, chicago)
    >>> groceries_w_communities[["Chain", "community", "geometry"]].head(2)
                   Chain    community                                geometry
    0     VIET HOA PLAZA       UPTOWN   MULTIPOINT ((1168268.672 1933554.35))
    1  COUNTY FAIR FOODS  MORGAN PARK  MULTIPOINT ((1162302.618 1832900.224))


    To include the distances:

    >>> groceries_w_communities = geopandas.sjoin_nearest(groceries, chicago, \
distance_col="distances")
    >>> groceries_w_communities[["Chain", "community", \
"distances"]].head(2)
                   Chain    community  distances
    0     VIET HOA PLAZA       UPTOWN        0.0
    1  COUNTY FAIR FOODS  MORGAN PARK        0.0

    In the following example, we get multiple groceries for Uptown because all
    results are equidistant (in this case zero because they intersect).
    In fact, we get 4 results in total:

    >>> chicago_w_groceries = geopandas.sjoin_nearest(groceries, chicago, \
distance_col="distances", how="right")
    >>> uptown_results = \
chicago_w_groceries[chicago_w_groceries["community"] == "UPTOWN"]
    >>> uptown_results[["Chain", "community"]]
                Chain community
    30  VIET HOA PLAZA    UPTOWN
    30      JEWEL OSCO    UPTOWN
    30          TARGET    UPTOWN
    30       Mariano's    UPTOWN

    See Also
    --------
    sjoin : binary predicate joins
    GeoDataFrame.sjoin_nearest : equivalent method

    Notes
    -----
    Since this join relies on distances, results will be inaccurate
    if your geometries are in a geographic CRS.

    Every operation in GeoPandas is planar, i.e. the potential third
    dimension is not taken into account.
    """
    _basic_checks(left_df, right_df, how, lsuffix, rsuffix)

    left_df.geometry.values.check_geographic_crs(stacklevel=1)
    right_df.geometry.values.check_geographic_crs(stacklevel=1)

    export_result, nearest_selected = _sjoin_nearest_export_result(
        left_df,
        right_df,
        how,
        max_distance,
        lsuffix,
        rsuffix,
        distance_col,
        exclusive,
    )
    # Dispatch event for sjoin_nearest surface -- sindex.nearest already
    # records its own event; this gives visibility at the sjoin layer.
    record_dispatch_event(
        surface="geopandas.tools.sjoin_nearest",
        operation="sjoin_nearest",
        implementation="sindex_nearest_delegate",
        reason="sjoin_nearest delegates to sindex.nearest for query work",
        detail=f"how={how}, max_distance={max_distance!r}, exclusive={exclusive}",
        selected=nearest_selected,
    )
    return export_result.to_geodataframe()


def _sjoin_nearest_export_result(
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    how: str,
    max_distance: float | None,
    lsuffix: str,
    rsuffix: str,
    distance_col: str | None,
    exclusive: bool,
) -> tuple[RelationJoinExportResult, ExecutionMode]:
    """Build the deferred public nearest-join export result before GeoDataFrame assembly."""
    native_result, nearest_selected = _sjoin_nearest_relation_result(
        left_df,
        right_df,
        max_distance,
        how,
        distance_col is not None,
        exclusive,
    )
    return (
        RelationJoinExportResult(
            relation_result=native_result,
            left_df=left_df,
            right_df=right_df,
            how=how,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            distance_col=distance_col,
            predicate="nearest",
        ),
        nearest_selected,
    )
