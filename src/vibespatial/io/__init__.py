"""vibespatial.io -- IO codecs and GPU-accelerated readers.

Lazy-exports GPU reader functions and spatial index utilities so that
power users can import them directly::

    from vibespatial.io import read_geojson_gpu
    from vibespatial.io import build_spatial_index, GpuSpatialIndex

Imports are deferred via ``__getattr__`` to avoid pulling in heavy CUDA
dependencies (CuPy, NVRTC) at package import time.
"""
from __future__ import annotations


def __getattr__(name: str):
    _LAZY_IMPORTS = {
        "read_geojson_gpu": ("vibespatial.io.geojson_gpu", "read_geojson_gpu"),
        "read_wkt_gpu": ("vibespatial.io.wkt_gpu", "read_wkt_gpu"),
        "read_csv_gpu": ("vibespatial.io.csv_gpu", "read_csv_gpu"),
        "read_kml_gpu": ("vibespatial.io.kml_gpu", "read_kml_gpu"),
        "read_dbf_gpu": ("vibespatial.io.dbf_gpu", "read_dbf_gpu"),
        "read_shp_gpu": ("vibespatial.io.shp_gpu", "read_shp_gpu"),
        "read_fgb_gpu": ("vibespatial.io.fgb_gpu", "read_fgb_gpu"),
        "read_osm_pbf_nodes": ("vibespatial.io.osm_gpu", "read_osm_pbf_nodes"),
        "build_spatial_index": (
            "vibespatial.io.gpu_parse.indexing",
            "build_spatial_index",
        ),
        "GpuSpatialIndex": (
            "vibespatial.io.gpu_parse.indexing",
            "GpuSpatialIndex",
        ),
        "read_postgis_gpu": (
            "vibespatial.io.postgis_gpu",
            "read_postgis_gpu",
        ),
        "to_postgis_gpu": (
            "vibespatial.io.postgis_gpu",
            "to_postgis_gpu",
        ),
    }
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'vibespatial.io' has no attribute {name!r}")


__all__ = [
    "read_geojson_gpu",
    "read_wkt_gpu",
    "read_csv_gpu",
    "read_kml_gpu",
    "read_dbf_gpu",
    "read_shp_gpu",
    "read_fgb_gpu",
    "read_osm_pbf_nodes",
    "build_spatial_index",
    "GpuSpatialIndex",
    "read_postgis_gpu",
    "to_postgis_gpu",
]
