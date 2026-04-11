from __future__ import annotations

from functools import wraps


def install_shapely_make_valid_dispatch() -> None:
    import shapely as _shapely

    if getattr(_shapely.make_valid, "_vibespatial_dispatch", False):
        return

    from vibespatial.api.geometry_array import GeometryArray
    from vibespatial.geometry.device_array import DeviceGeometryArray

    _original_make_valid = _shapely.make_valid

    @wraps(_original_make_valid)
    def _vibespatial_make_valid_dispatch(geometry, *args, **kwargs):
        if isinstance(geometry, (GeometryArray, DeviceGeometryArray)):
            return geometry.make_valid(*args, **kwargs)
        return _original_make_valid(geometry, *args, **kwargs)

    _vibespatial_make_valid_dispatch._vibespatial_dispatch = True
    _vibespatial_make_valid_dispatch.__wrapped__ = _original_make_valid
    _shapely.make_valid = _vibespatial_make_valid_dispatch
