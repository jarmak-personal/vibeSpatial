from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString

from vibespatial.constructive.linestring import linestring_buffer_owned_array
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.residency import Residency


def _buffer_matches_shapely(gpu_geom, shapely_geom, *, tol: float = 1e-6) -> bool:
    """Check geometric equivalence using area-based comparison."""
    if gpu_geom is None or shapely_geom is None:
        return gpu_geom is shapely_geom
    sym_diff = gpu_geom.symmetric_difference(shapely_geom)
    return sym_diff.area < tol * max(gpu_geom.area, shapely_geom.area, 1e-12)


def _buffer_matches_shapely_normalized_exact(
    gpu_geom,
    shapely_geom,
    *,
    tol: float = 1e-12,
) -> bool:
    if gpu_geom is None or shapely_geom is None:
        return gpu_geom is shapely_geom
    return bool(
        shapely.normalize(gpu_geom).equals_exact(
            shapely.normalize(shapely_geom),
            tolerance=tol,
        )
    )


@pytest.mark.gpu
def test_linestring_buffer_gpu_simple_horizontal_line() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    lines = from_shapely_geometries([LineString([(0, 0), (10, 0)])])
    gpu = linestring_buffer_owned_array(lines, 1.0, quad_segs=8, dispatch_mode=ExecutionMode.GPU)

    expected = shapely.buffer(LineString([(0, 0), (10, 0)]), 1.0, quad_segs=8)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_linestring_buffer_gpu_right_angle_line() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    line = LineString([(0, 0), (10, 0), (10, 10)])
    lines = from_shapely_geometries([line])
    gpu = linestring_buffer_owned_array(lines, 1.0, quad_segs=8, dispatch_mode=ExecutionMode.GPU)

    expected = shapely.buffer(line, 1.0, quad_segs=8)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_linestring_buffer_gpu_zigzag() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    line = LineString([(0, 0), (5, 5), (10, 0), (15, 5)])
    lines = from_shapely_geometries([line])
    gpu = linestring_buffer_owned_array(lines, 2.0, quad_segs=4, dispatch_mode=ExecutionMode.GPU)

    expected = shapely.buffer(line, 2.0, quad_segs=4)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_linestring_buffer_gpu_multiple_rows() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    geoms = [
        LineString([(0, 0), (10, 0)]),
        LineString([(0, 0), (5, 5), (10, 0)]),
        LineString([(0, 0), (0, 10)]),
    ]
    lines = from_shapely_geometries(geoms)
    radii = np.asarray([1.0, 2.0, 0.5])
    gpu = linestring_buffer_owned_array(lines, radii, quad_segs=4, dispatch_mode=ExecutionMode.GPU)

    for i, (result, geom, r) in enumerate(zip(gpu.to_shapely(), geoms, radii)):
        expected = shapely.buffer(geom, r, quad_segs=4)
        assert _buffer_matches_shapely(result, expected), (
            f"row {i} area diff: {result.symmetric_difference(expected).area}"
        )


@pytest.mark.gpu
def test_linestring_buffer_gpu_collinear_segments() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    line = LineString([(0, 0), (5, 0), (10, 0)])
    lines = from_shapely_geometries([line])
    gpu = linestring_buffer_owned_array(lines, 1.0, quad_segs=4, dispatch_mode=ExecutionMode.GPU)

    expected = shapely.buffer(line, 1.0, quad_segs=4)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_linestring_buffer_gpu_quad_segs_1() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    line = LineString([(0, 0), (10, 0)])
    lines = from_shapely_geometries([line])
    gpu = linestring_buffer_owned_array(lines, 1.0, quad_segs=1, dispatch_mode=ExecutionMode.GPU)

    expected = shapely.buffer(line, 1.0, quad_segs=1)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_linestring_buffer_gpu_keeps_metadata_lazy(strict_device_guard) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    line = LineString([(0, 0), (10, 0)])
    lines = from_shapely_geometries([line], residency=Residency.DEVICE)
    gpu = linestring_buffer_owned_array(lines, 1.0, quad_segs=4, dispatch_mode=ExecutionMode.GPU)

    assert gpu.residency is Residency.DEVICE
    assert gpu._validity is None
    assert gpu._tags is None
    assert gpu._family_row_offsets is None
    assert gpu.device_state is not None

    poly_buf = gpu.device_state.families[GeometryFamily.POLYGON]
    assert isinstance(poly_buf.geometry_offsets, cp.ndarray)
    assert isinstance(poly_buf.ring_offsets, cp.ndarray)


@pytest.mark.gpu
def test_linestring_buffer_gpu_scatter_does_not_force_runtime_sync(monkeypatch) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()

    def _fail_sync():
        raise AssertionError("linestring buffer scatter should not force runtime.synchronize()")

    monkeypatch.setattr(runtime, "synchronize", _fail_sync)

    lines = from_shapely_geometries(
        [LineString([(0, 0), (10, 0)])],
        residency=Residency.DEVICE,
    )
    gpu = linestring_buffer_owned_array(
        lines,
        1.0,
        quad_segs=4,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert gpu.device_state is not None
    assert gpu.device_state.families[GeometryFamily.POLYGON].x.size > 0


# --- Phase 3: cap/join style variants ---


@pytest.mark.gpu
@pytest.mark.parametrize("cap_style", ["round", "flat", "square"])
@pytest.mark.parametrize("join_style", ["round", "mitre", "bevel"])
def test_linestring_buffer_gpu_cap_join_combinations(cap_style: str, join_style: str) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    line = LineString([(0, 0), (10, 0), (10, 10)])
    lines = from_shapely_geometries([line])
    gpu = linestring_buffer_owned_array(
        lines, 1.0, quad_segs=4,
        cap_style=cap_style, join_style=join_style,
        dispatch_mode=ExecutionMode.GPU,
    )

    expected = shapely.buffer(line, 1.0, quad_segs=4, cap_style=cap_style, join_style=join_style)
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"cap={cap_style} join={join_style}: area diff={result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("join_style", ["round", "mitre", "bevel"])
def test_linestring_buffer_gpu_square_cap_elbow_matches_shapely_exactly(join_style: str) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    line = LineString([(0, 0), (10, 0), (10, 10)])
    lines = from_shapely_geometries([line])
    gpu = linestring_buffer_owned_array(
        lines,
        1.0,
        quad_segs=4,
        cap_style="square",
        join_style=join_style,
        dispatch_mode=ExecutionMode.GPU,
    )

    expected = shapely.buffer(
        line,
        1.0,
        quad_segs=4,
        cap_style="square",
        join_style=join_style,
    )
    result = gpu.to_shapely()[0]
    assert len(result.exterior.coords) == len(expected.exterior.coords)
    assert _buffer_matches_shapely_normalized_exact(result, expected), (
        f"cap=square join={join_style}: normalized coords differ"
    )


@pytest.mark.gpu
def test_linestring_buffer_gpu_mitre_limit_exceeded_falls_back_to_bevel() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    # Near-parallel segments have high mitre ratio; mitre_limit=1.0 forces bevel fallback
    line = LineString([(0, 0), (10, 0), (10, 10)])
    lines = from_shapely_geometries([line])
    gpu = linestring_buffer_owned_array(
        lines, 1.0, quad_segs=4,
        join_style="mitre", mitre_limit=1.0,
        dispatch_mode=ExecutionMode.GPU,
    )

    # mitre_limit=1.0 < sqrt(2) ≈ 1.414 for a right-angle turn, so bevel kicks in
    expected = shapely.buffer(line, 1.0, quad_segs=4, join_style="bevel")
    result = gpu.to_shapely()[0]
    # GPU uses bevel when mitre exceeded; Shapely's clipped mitre may differ slightly
    assert _buffer_matches_shapely(result, expected, tol=0.02), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_linestring_buffer_gpu_flat_cap_zigzag() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    line = LineString([(0, 0), (5, 5), (10, 0), (15, 5)])
    lines = from_shapely_geometries([line])
    gpu = linestring_buffer_owned_array(
        lines, 1.5, quad_segs=4, cap_style="flat",
        dispatch_mode=ExecutionMode.GPU,
    )

    expected = shapely.buffer(line, 1.5, quad_segs=4, cap_style="flat")
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_linestring_buffer_gpu_square_cap_straight_line() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    line = LineString([(0, 0), (10, 0)])
    lines = from_shapely_geometries([line])
    gpu = linestring_buffer_owned_array(
        lines, 1.0, quad_segs=4, cap_style="square",
        dispatch_mode=ExecutionMode.GPU,
    )

    expected = shapely.buffer(line, 1.0, quad_segs=4, cap_style="square")
    result = gpu.to_shapely()[0]
    assert _buffer_matches_shapely(result, expected), (
        f"area diff: {result.symmetric_difference(expected).area}"
    )


@pytest.mark.gpu
def test_linestring_buffer_gpu_bevel_multiple_rows() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    geoms = [
        LineString([(0, 0), (10, 0), (10, 10)]),
        LineString([(0, 0), (5, 5), (10, 0)]),
    ]
    lines = from_shapely_geometries(geoms)
    radii = np.asarray([1.0, 2.0])
    gpu = linestring_buffer_owned_array(
        lines, radii, quad_segs=4, join_style="bevel", cap_style="flat",
        dispatch_mode=ExecutionMode.GPU,
    )

    for i, (result, geom, r) in enumerate(zip(gpu.to_shapely(), geoms, radii)):
        expected = shapely.buffer(geom, r, quad_segs=4, join_style="bevel", cap_style="flat")
        assert _buffer_matches_shapely(result, expected), (
            f"row {i} area diff: {result.symmetric_difference(expected).area}"
        )
