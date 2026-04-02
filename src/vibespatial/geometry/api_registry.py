from __future__ import annotations

from collections.abc import Callable
from typing import Any

_make_transform_func: Callable[[Any, Any], Any] | None = None
_transform_geometry_array: Callable[[Any, Any], Any] | None = None
_device_spatial_index_factory: Callable[[Any], Any] | None = None


def register_host_transform_helpers(
    *,
    make_transform_func: Callable[[Any, Any], Any],
    transform_geometry_array: Callable[[Any, Any], Any],
) -> None:
    global _make_transform_func, _transform_geometry_array
    _make_transform_func = make_transform_func
    _transform_geometry_array = transform_geometry_array


def build_host_transform_func(src_crs, dst_crs):
    if _make_transform_func is None:
        raise RuntimeError("Host transform helpers are not registered")
    return _make_transform_func(src_crs, dst_crs)


def apply_host_transform(data, func):
    if _transform_geometry_array is None:
        raise RuntimeError("Host geometry transform helper is not registered")
    return _transform_geometry_array(data, func)


def register_device_spatial_index_factory(
    factory: Callable[[Any], Any],
) -> None:
    global _device_spatial_index_factory
    _device_spatial_index_factory = factory


def build_device_spatial_index(device_geometry_array):
    if _device_spatial_index_factory is None:
        raise RuntimeError("Device spatial-index factory is not registered")
    return _device_spatial_index_factory(device_geometry_array)
