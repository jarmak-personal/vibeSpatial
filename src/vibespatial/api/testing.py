"""Testing functionality for geopandas objects."""

import warnings

import numpy as np
import pandas as pd
import shapely

from vibespatial.api import GeoDataFrame, GeoSeries
from vibespatial.api.geo_base import _is_geometry_like_dtype

_PUBLIC_EQUALITY_GEOM_TYPES = frozenset(
    {
        "Point",
        "LineString",
        "Polygon",
        "MultiPoint",
        "MultiLineString",
        "MultiPolygon",
    }
)


def _isna(this):
    """Version of isna that works for both scalars and (Geo)Series."""
    with warnings.catch_warnings():
        # GeoSeries.isna will raise a warning about no longer returning True
        # for empty geometries. This helper is used below always in combination
        # with an is_empty check to preserve behaviour, and thus we ignore the
        # warning here to avoid it bubbling up to the user
        warnings.filterwarnings(
            "ignore", r"GeoSeries.isna\(\) previously returned", UserWarning
        )
        if hasattr(this, "isna"):
            return this.isna()
        elif hasattr(this, "isnull"):
            return this.isnull()
        else:
            return pd.isnull(this)


def _geom_equals_mask(this, that):
    """
    Test for geometric equality. Empty or missing geometries are considered
    equal.

    Parameters
    ----------
    this, that : arrays of Geo objects (or anything that has an `is_empty`
                 attribute)

    Returns
    -------
    Series
        boolean Series, True if geometries in left equal geometries in right
    """
    this_values, index = _comparison_values_and_index(this)
    that_values = _comparison_values(that)
    equals_mask = _public_geom_equals_mask(this, that)
    if equals_mask is None:
        equals_mask = _comparison_mask(shapely.equals(this_values, that_values), index)
    return equals_mask | (_empty_mask(this, index) & _empty_mask(that, index)) | (
        _na_mask(this, index) & _na_mask(that, index)
    )


def geom_equals(this, that):
    """
    Test for geometric equality. Empty or missing geometries are considered
    equal.

    Parameters
    ----------
    this, that : arrays of Geo objects (or anything that has an `is_empty`
                 attribute)

    Returns
    -------
    bool
        True if all geometries in left equal geometries in right
    """
    return _geom_equals_mask(this, that).all()


def _geom_almost_equals_mask(this, that):
    """
    Test for 'almost' geometric equality. Empty or missing geometries
    considered equal.

    This method allows small difference in the coordinates, but this
    requires coordinates be in the same order for all components of a geometry.

    Parameters
    ----------
    this, that : arrays of Geo objects

    Returns
    -------
    Series
        boolean Series, True if geometries in left almost equal geometries in right
    """
    this_values, index = _comparison_values_and_index(this)
    that_values = _comparison_values(that)
    equals_mask = _public_geom_almost_equals_mask(this, that)
    if equals_mask is None:
        equals_mask = _comparison_mask(
            shapely.equals_exact(this_values, that_values, tolerance=0.5 * 10 ** (-6)),
            index,
        )
    return equals_mask | (_empty_mask(this, index) & _empty_mask(that, index)) | (
        _na_mask(this, index) & _na_mask(that, index)
    )


def _public_geom_equals_mask(this, that):
    this_series, that_series = _series_pair_for_public_comparison(this, that)
    if this_series is None or that_series is None:
        return None
    return this_series.geom_equals(that_series, align=False)


def _public_geom_almost_equals_mask(this, that):
    this_series, that_series = _series_pair_for_public_comparison(this, that)
    if this_series is None or that_series is None:
        return None
    return this_series.geom_equals_exact(that_series, tolerance=0.5 * 10 ** (-6), align=False)


def _series_pair_for_public_comparison(this, that):
    if isinstance(this, GeoDataFrame):
        this = this.geometry
    if isinstance(that, GeoDataFrame):
        that = that.geometry
    if not isinstance(this, GeoSeries) or not isinstance(that, GeoSeries):
        return None, None
    if _contains_unsupported_public_equality_family(this) or _contains_unsupported_public_equality_family(that):
        return None, None
    return this, that


def _contains_unsupported_public_equality_family(obj) -> bool:
    values = _comparison_values(obj)
    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        array = np.asarray([values], dtype=object)
    missing = shapely.is_missing(array)
    for geometry in array[~missing]:
        geom_type = getattr(geometry, "geom_type", None)
        if geom_type not in _PUBLIC_EQUALITY_GEOM_TYPES:
            return True
    return False


def _comparison_values(obj):
    if isinstance(obj, GeoDataFrame):
        obj = obj.geometry
    if isinstance(obj, GeoSeries):
        return np.asarray(obj.array, dtype=object)
    if _is_geometry_like_dtype(getattr(obj, "dtype", None)):
        return np.asarray(obj, dtype=object)
    if isinstance(obj, (list, tuple)):
        return np.asarray(obj, dtype=object)
    return obj


def _comparison_values_and_index(obj):
    if isinstance(obj, GeoDataFrame):
        obj = obj.geometry
    if isinstance(obj, GeoSeries):
        return np.asarray(obj.array, dtype=object), obj.index
    values = _comparison_values(obj)
    if np.ndim(values) == 0 or not hasattr(values, "__len__"):
        return values, pd.RangeIndex(1)
    return values, pd.RangeIndex(len(values))


def _comparison_mask(values, index):
    array = np.asarray(values, dtype=bool)
    if array.ndim == 0:
        array = np.repeat(array, len(index))
    return pd.Series(array, index=index)


def _empty_mask(obj, index):
    if isinstance(obj, GeoDataFrame):
        obj = obj.geometry
    if isinstance(obj, GeoSeries):
        return pd.Series(np.asarray(obj.is_empty, dtype=bool), index=index)
    values = _comparison_values(obj)
    array = np.asarray(shapely.is_empty(values), dtype=bool)
    if array.ndim == 0:
        array = np.repeat(array, len(index))
    return pd.Series(array, index=index)


def _na_mask(obj, index):
    if isinstance(obj, GeoDataFrame):
        obj = obj.geometry
    mask = _isna(obj)
    array = np.asarray(mask, dtype=bool)
    if array.ndim == 0:
        array = np.repeat(array, len(index))
    return pd.Series(array, index=index)


def geom_almost_equals(this, that):
    """
    Test for 'almost' geometric equality. Empty or missing geometries
    considered equal.

    This method allows small difference in the coordinates, but this
    requires coordinates be in the same order for all components of a geometry.

    Parameters
    ----------
    this, that : arrays of Geo objects (or anything that has an `is_empty`
                 property)

    Returns
    -------
    bool
        True if all geometries in left almost equal geometries in right
    """
    if isinstance(this, GeoDataFrame) and isinstance(that, GeoDataFrame):
        this = this.geometry
        that = that.geometry

    return _geom_almost_equals_mask(this, that).all()


def assert_geoseries_equal(
    left,
    right,
    check_dtype=True,
    check_index_type=False,
    check_series_type=True,
    check_less_precise=False,
    check_geom_type=False,
    check_crs=True,
    normalize=False,
):
    """
    Test util for checking that two GeoSeries are equal.

    Parameters
    ----------
    left, right : two GeoSeries
    check_dtype : bool, default False
        If True, check geo dtype [only included so it's a drop-in replacement
        for assert_series_equal].
    check_index_type : bool, default False
        Check that index types are equal.
    check_series_type : bool, default True
        Check that both are same type (*and* are GeoSeries). If False,
        will attempt to convert both into GeoSeries.
    check_less_precise : bool, default False
        If True, use geom_equals_exact with relative error of 0.5e-6.
        If False, use geom_equals.
    check_geom_type : bool, default False
        If True, check that all the geom types are equal.
    check_crs: bool, default True
        If `check_series_type` is True, then also check that the
        crs matches.
    normalize: bool, default False
        If True, normalize the geometries before comparing equality.
        Typically useful with ``check_less_precise=True``, which uses
        ``geom_equals_exact`` and requires exact coordinate order.
    """
    assert len(left) == len(right), f"{len(left)} != {len(right)}"

    if check_dtype:
        msg = "dtype should be a GeometryDtype, got {0}"
        assert _is_geometry_like_dtype(left.dtype), msg.format(left.dtype)
        assert _is_geometry_like_dtype(right.dtype), msg.format(right.dtype)

    if check_index_type:
        assert isinstance(left.index, type(right.index))

    if check_series_type:
        assert isinstance(left, GeoSeries)
        assert isinstance(left, type(right))

        if check_crs:
            assert left.crs == right.crs
    else:
        if not isinstance(left, GeoSeries):
            left = GeoSeries(left)
        if not isinstance(right, GeoSeries):
            right = GeoSeries(right, index=left.index)

    assert left.index.equals(right.index), f"index: {left.index} != {right.index}"

    if check_geom_type:
        assert (left.geom_type == right.geom_type).all(), (
            f"type: {left.geom_type} != {right.geom_type}"
        )

    if normalize:
        left = GeoSeries(left.array.normalize())
        right = GeoSeries(right.array.normalize())

    if not check_crs:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "CRS mismatch", UserWarning)
            _check_equality(left, right, check_less_precise)
    else:
        _check_equality(left, right, check_less_precise)


def _truncated_string(geom):
    """Truncate WKT repr of geom."""
    s = str(geom)
    if len(s) > 100:
        return s[:100] + "..."
    else:
        return s


def _check_equality(left, right, check_less_precise):
    assert_error_message = (
        "{0} out of {1} geometries are not {3}equal.\n"
        "Indices where geometries are not {3}equal: {2} \n"
        "The first not {3}equal geometry:\n"
        "Left: {4}\n"
        "Right: {5}\n"
    )
    if check_less_precise:
        precise = "almost "
        equal = _geom_almost_equals_mask(left, right)
    else:
        precise = ""
        equal = _geom_equals_mask(left, right)

    if not equal.all():
        unequal_left_geoms = left[~equal]
        unequal_right_geoms = right[~equal]
        raise AssertionError(
            assert_error_message.format(
                len(unequal_left_geoms),
                len(left),
                unequal_left_geoms.index.to_list(),
                precise,
                _truncated_string(unequal_left_geoms.iloc[0]),
                _truncated_string(unequal_right_geoms.iloc[0]),
            )
        )


def assert_geodataframe_equal(
    left,
    right,
    check_dtype=True,
    check_index_type="equiv",
    check_column_type="equiv",
    check_frame_type=True,
    check_like=False,
    check_less_precise=False,
    check_geom_type=False,
    check_crs=True,
    normalize=False,
):
    """Check that two GeoDataFrames are equal.

    Parameters
    ----------
    left, right : two GeoDataFrames
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type, check_column_type : bool, default 'equiv'
        Check that index types are equal.
    check_frame_type : bool, default True
        Check that both are same type (*and* are GeoDataFrames). If False,
        will attempt to convert both into GeoDataFrame.
    check_like : bool, default False
        If true, ignore the order of rows & columns
    check_less_precise : bool, default False
        If True, use geom_equals_exact. if False, use geom_equals.
    check_geom_type : bool, default False
        If True, check that all the geom types are equal.
    check_crs: bool, default True
        If `check_frame_type` is True, then also check that the
        crs matches.
    normalize: bool, default False
        If True, normalize the geometries before comparing equality.
        Typically useful with ``check_less_precise=True``, which uses
        ``geom_equals_exact`` and requires exact coordinate order.
    """
    try:
        # added from pandas 0.20
        from pandas.testing import assert_frame_equal, assert_index_equal
    except ImportError:
        from pandas.util.testing import assert_frame_equal, assert_index_equal

    # instance validation
    if check_frame_type:
        assert isinstance(left, GeoDataFrame)
        assert isinstance(left, type(right))

        if check_crs:
            # allow if neither left and right has an active geometry column
            if (
                left._geometry_column_name is None
                and right._geometry_column_name is None
            ):
                pass
            elif (
                left._geometry_column_name not in left.columns
                and right._geometry_column_name not in right.columns
            ):
                pass
            # no crs can be either None or {}
            elif not left.crs and not right.crs:
                pass
            else:
                assert left.crs == right.crs
    else:
        if not isinstance(left, GeoDataFrame):
            left = GeoDataFrame(left)
        if not isinstance(right, GeoDataFrame):
            right = GeoDataFrame(right)

    # shape comparison
    assert left.shape == right.shape, (
        f"GeoDataFrame shape mismatch, left: {left.shape!r}, right: {right.shape!r}.\n"
        f"Left columns: {left.columns!r}, right columns: {right.columns!r}"
    )

    if check_like:
        left = left.reindex_like(right)

    # column comparison
    assert_index_equal(
        left.columns, right.columns, exact=check_column_type, obj="GeoDataFrame.columns"
    )

    # geometry comparison
    for col, dtype in left.dtypes.items():
        if _is_geometry_like_dtype(dtype):
            assert_geoseries_equal(
                left[col],
                right[col],
                normalize=normalize,
                check_dtype=check_dtype,
                check_less_precise=check_less_precise,
                check_geom_type=check_geom_type,
                check_crs=check_crs,
            )

    # ensure the active geometry column is the same
    assert left._geometry_column_name == right._geometry_column_name

    # drop geometries and check remaining columns
    left2 = left.select_dtypes(exclude="geometry")
    right2 = right.select_dtypes(exclude="geometry")
    assert_frame_equal(
        left2,
        right2,
        check_dtype=check_dtype,
        check_index_type=check_index_type,
        check_column_type=check_column_type,
        obj="GeoDataFrame",
    )
