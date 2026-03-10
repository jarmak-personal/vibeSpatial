from __future__ import annotations

from vibespatial import IOFormat, IOOperation, IOPathKind, plan_io_support


def test_geoarrow_and_geoparquet_are_canonical_gpu_formats() -> None:
    geoarrow = plan_io_support(IOFormat.GEOARROW, IOOperation.READ)
    geoparquet = plan_io_support(IOFormat.GEOPARQUET, IOOperation.SCAN)

    assert geoarrow.selected_path is IOPathKind.GPU_NATIVE
    assert geoparquet.selected_path is IOPathKind.GPU_NATIVE
    assert geoarrow.canonical_gpu is True
    assert geoparquet.canonical_gpu is True


def test_wkb_geojson_and_shapefile_stay_noncanonical() -> None:
    wkb = plan_io_support("wkb", "decode")
    geojson = plan_io_support("geojson", "read")
    shapefile = plan_io_support("shapefile", "write")

    assert wkb.selected_path is IOPathKind.HYBRID
    assert geojson.selected_path is IOPathKind.HYBRID
    assert shapefile.selected_path is IOPathKind.HYBRID
    assert wkb.canonical_gpu is False


def test_gdal_legacy_stays_explicit_fallback() -> None:
    plan = plan_io_support(IOFormat.GDAL_LEGACY, IOOperation.READ)

    assert plan.selected_path is IOPathKind.FALLBACK
    assert "fallback" in plan.reason.lower() or "legacy" in plan.reason.lower()
