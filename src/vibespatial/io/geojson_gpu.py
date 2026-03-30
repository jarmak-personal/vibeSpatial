"""GPU byte-classification GeoJSON parser.

Parses GeoJSON FeatureCollection files on GPU using NVRTC kernels for
byte classification, structural scanning, coordinate extraction, and
ASCII-to-float64 parsing.  Property extraction stays on CPU (hybrid
design per vibeSpatial GPU memory policy).

Supports homogeneous and mixed Point, LineString, Polygon, MultiPoint,
MultiLineString, and MultiPolygon files.  GeometryCollection and chunked
processing for files exceeding GPU memory are deferred.

Shared parsing primitives (quote parity, bracket depth, number boundary
detection, ASCII float parsing, span masking) are imported from
``vibespatial.io.gpu_parse``.  GeoJSON-specific kernels (coordinate key
search, type detection, ring counting, offset scattering, feature
boundaries) remain in this module.
"""
from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    _device_gather_offset_slices,
)
from vibespatial.io.geojson_gpu_kernels import (
    _CLASSIFY_TYPE_NAMES,
    _CLASSIFY_TYPE_SOURCE,
    _COORD_KEY_NAMES,
    _COORD_KEY_SOURCE,
    _COORD_SPAN_END_NAMES,
    _COORD_SPAN_END_SOURCE,
    _FEATURE_BOUNDARY_NAMES,
    _FEATURE_BOUNDARY_SOURCE,
    _MPOLY_COUNT_NAMES,
    _MPOLY_COUNT_SOURCE,
    _MPOLY_SCATTER_NAMES,
    _MPOLY_SCATTER_SOURCE,
    _RING_COUNT_NAMES,
    _RING_COUNT_SOURCE,
    _SCATTER_COORDS_NAMES,
    _SCATTER_COORDS_SOURCE,
    _TYPE_KEY_NAMES,
    _TYPE_KEY_SOURCE,
)
from vibespatial.io.gpu_parse.numeric import (
    extract_number_positions,
    number_boundaries,
    parse_ascii_floats,
)
from vibespatial.io.gpu_parse.pattern import mark_spans
from vibespatial.io.gpu_parse.structural import bracket_depth, quote_parity

from .pylibcudf import _build_device_mixed_owned, _build_device_single_family_owned

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

# ctypes for kernel params not in cuda_runtime.py
KERNEL_PARAM_I64 = ctypes.c_longlong
KERNEL_PARAM_I8 = ctypes.c_int8

# ---------------------------------------------------------------------------
# NVRTC warmup (ADR-0034 Level 2)
# ---------------------------------------------------------------------------
# Only GeoJSON-specific kernels are warmed up here.  Shared primitives
# (quote parity, bracket depth, number boundaries, float parsing, span
# marking) register their own warmup in their respective gpu_parse modules.
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("geojson-coord-key", _COORD_KEY_SOURCE, _COORD_KEY_NAMES),
    ("geojson-coord-span-end", _COORD_SPAN_END_SOURCE, _COORD_SPAN_END_NAMES),
    ("geojson-ring-count", _RING_COUNT_SOURCE, _RING_COUNT_NAMES),
    ("geojson-mpoly-count", _MPOLY_COUNT_SOURCE, _MPOLY_COUNT_NAMES),
    ("geojson-mpoly-scatter", _MPOLY_SCATTER_SOURCE, _MPOLY_SCATTER_NAMES),
    ("geojson-scatter-coords", _SCATTER_COORDS_SOURCE, _SCATTER_COORDS_NAMES),
    ("geojson-feature-boundary", _FEATURE_BOUNDARY_SOURCE, _FEATURE_BOUNDARY_NAMES),
    ("geojson-type-key", _TYPE_KEY_SOURCE, _TYPE_KEY_NAMES),
    ("geojson-classify-type", _CLASSIFY_TYPE_SOURCE, _CLASSIFY_TYPE_NAMES),
])


# ---------------------------------------------------------------------------
# GeoJSON-specific kernel compilation helpers
# ---------------------------------------------------------------------------

def _coord_key_kernels():
    return compile_kernel_group("geojson-coord-key", _COORD_KEY_SOURCE, _COORD_KEY_NAMES)


def _coord_span_end_kernels():
    return compile_kernel_group("geojson-coord-span-end", _COORD_SPAN_END_SOURCE, _COORD_SPAN_END_NAMES)


def _ring_count_kernels():
    return compile_kernel_group("geojson-ring-count", _RING_COUNT_SOURCE, _RING_COUNT_NAMES)


def _mpoly_count_kernels():
    return compile_kernel_group("geojson-mpoly-count", _MPOLY_COUNT_SOURCE, _MPOLY_COUNT_NAMES)


def _mpoly_scatter_kernels():
    return compile_kernel_group("geojson-mpoly-scatter", _MPOLY_SCATTER_SOURCE, _MPOLY_SCATTER_NAMES)


def _scatter_coords_kernels():
    return compile_kernel_group("geojson-scatter-coords", _SCATTER_COORDS_SOURCE, _SCATTER_COORDS_NAMES)


def _feature_boundary_kernels():
    return compile_kernel_group("geojson-feature-boundary", _FEATURE_BOUNDARY_SOURCE, _FEATURE_BOUNDARY_NAMES)


def _type_key_kernels():
    return compile_kernel_group("geojson-type-key", _TYPE_KEY_SOURCE, _TYPE_KEY_NAMES)


def _classify_type_kernels():
    return compile_kernel_group("geojson-classify-type", _CLASSIFY_TYPE_SOURCE, _CLASSIFY_TYPE_NAMES)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GeoJSONGpuResult:
    owned: OwnedGeometryArray
    n_features: int
    host_bytes: np.ndarray
    feature_starts: np.ndarray
    feature_ends: np.ndarray

    def properties_loader(self) -> Callable[[], list[dict[str, object]]]:
        host_bytes = self.host_bytes
        feature_starts = self.feature_starts
        feature_ends = self.feature_ends

        def _load() -> list[dict[str, object]]:
            return _extract_properties_cpu(host_bytes, feature_starts, feature_ends)
        return _load

    def extract_properties_dataframe(self):
        import pandas as pd
        props = _extract_properties_cpu(self.host_bytes, self.feature_starts, self.feature_ends)
        if not props:
            return pd.DataFrame()
        return pd.DataFrame(props)


# ---------------------------------------------------------------------------
# CPU property extraction
# ---------------------------------------------------------------------------

def _extract_properties_cpu(
    host_bytes: np.ndarray,
    feature_starts: np.ndarray,
    feature_ends: np.ndarray,
) -> list[dict[str, object]]:
    from .geojson import _fast_json_loads

    raw = host_bytes.tobytes()
    result: list[dict[str, object]] = []
    for i in range(len(feature_starts)):
        start = int(feature_starts[i])
        end = int(feature_ends[i])
        feature_bytes = raw[start:end]
        try:
            feature = _fast_json_loads(feature_bytes)
            props = feature.get("properties") or {}
            result.append(dict(props))
        except Exception:
            result.append({})
    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _launch_kernel(runtime, kernel, n, params):
    grid, block = runtime.launch_config(kernel, int(n))
    runtime.launch(kernel, grid=grid, block=block, params=params)


def read_geojson_gpu(
    path: Path,
    *,
    target_crs: str | None = None,
) -> GeoJSONGpuResult:
    """Parse a GeoJSON file using GPU byte-classification pipeline.

    Parameters
    ----------
    path : Path
        Path to the GeoJSON file.
    target_crs : str, optional
        Target CRS to reproject coordinates into (e.g. ``"EPSG:3857"``).
        GeoJSON is EPSG:4326 by spec (RFC 7946).  When provided, a fused
        GPU coordinate transform runs between parsing and geometry assembly,
        with zero host round-trips.

    Returns a GeoJSONGpuResult with device-resident OwnedGeometryArray
    and host data for lazy CPU property extraction.
    """
    runtime = get_cuda_runtime()

    # S0: Read file to device via kvikio (parallel POSIX with pinned
    # bounce buffers) or cp.asarray fallback.
    from .kvikio_reader import read_file_to_device

    file_size = path.stat().st_size
    result = read_file_to_device(path, file_size)
    d_bytes = result.device_bytes
    if result.host_bytes is not None:
        # Fallback path: host_bytes already read, reuse them.
        host_bytes = result.host_bytes
    else:
        # kvikio path: buffered POSIX read populated the OS page cache,
        # so this np.fromfile hits warm cache (~memcpy speed).
        host_bytes = np.fromfile(str(path), dtype=np.uint8)
    n = len(d_bytes)
    if len(host_bytes) != n:
        raise OSError(
            f"File size changed between reads: device has {n} bytes, "
            f"host has {len(host_bytes)} bytes"
        )
    n_i64 = np.int64(n)
    ptr = runtime.pointer

    # S1b: Quote parity (0=outside, 1=inside string) via gpu_parse primitive.
    d_quote_parity = quote_parity(d_bytes)

    # S2: Nesting depth via gpu_parse primitive (delta kernel + prefix sum).
    d_depth = bracket_depth(d_bytes, d_quote_parity)

    # S3: Find "coordinates": positions (with quote-state filter)
    coord_kernels = _coord_key_kernels()
    d_hits = cp.zeros(n, dtype=cp.uint8)
    _launch_kernel(runtime, coord_kernels["find_coord_key"], n, (
        (ptr(d_bytes), ptr(d_quote_parity), ptr(d_hits), n_i64),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I64),
    ))
    d_coord_positions = cp.flatnonzero(d_hits).astype(cp.int64)
    del d_hits
    n_features = int(len(d_coord_positions))

    if n_features == 0:
        # Empty file — return empty Point geometry
        owned = _build_device_single_family_owned(
            family=GeometryFamily.POINT,
            validity_device=cp.ones(0, dtype=cp.bool_),
            x_device=cp.empty(0, dtype=cp.float64),
            y_device=cp.empty(0, dtype=cp.float64),
            geometry_offsets_device=cp.zeros(1, dtype=cp.int32),
            empty_mask_device=cp.zeros(0, dtype=cp.bool_),
            detail="GPU byte-classification GeoJSON parse (empty)",
        )
        return GeoJSONGpuResult(
            owned=owned,
            n_features=0,
            host_bytes=host_bytes,
            feature_starts=np.empty(0, dtype=np.int64),
            feature_ends=np.empty(0, dtype=np.int64),
        )

    # S3.5: Type detection — find "type": at geometry depth and classify
    tk_kernels = _type_key_kernels()
    d_type_hits = cp.zeros(n, dtype=cp.uint8)
    _launch_kernel(runtime, tk_kernels["find_type_key"], n, (
        (ptr(d_bytes), ptr(d_quote_parity), ptr(d_depth), ptr(d_type_hits), n_i64),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I64),
    ))
    d_type_positions = cp.flatnonzero(d_type_hits).astype(cp.int64)
    del d_type_hits
    n_type_matches = int(len(d_type_positions))
    if n_type_matches != n_features:
        raise ValueError(
            f"GeoJSON type detection mismatch: found {n_type_matches} geometry "
            f'"type" keys but {n_features} "coordinates" keys'
        )

    ct_kernels = _classify_type_kernels()
    d_family_tags = cp.empty(n_features, dtype=cp.int8)
    _launch_kernel(runtime, ct_kernels["classify_type_value"], n_features, (
        (ptr(d_bytes), ptr(d_type_positions), ptr(d_family_tags),
         np.int32(n_features), n_i64),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_I32, KERNEL_PARAM_I64),
    ))
    del d_type_positions

    # Check for unsupported types (only GeometryCollection now)
    unsupported_mask = d_family_tags < 0
    if cp.any(unsupported_mask):
        n_unsupported = int(cp.sum(unsupported_mask))
        raise NotImplementedError(
            f"GPU GeoJSON parser: {n_unsupported} features have unsupported "
            f"geometry types (GeometryCollection)"
        )

    # Determine if homogeneous or mixed
    unique_tags = cp.unique(d_family_tags)
    is_homogeneous = len(unique_tags) == 1
    single_tag = int(unique_tags[0]) if is_homogeneous else None
    pg_tag = np.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    mpoly_tag = np.int8(FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
    has_polygons = bool(cp.any(unique_tags == pg_tag))
    has_multipolygons = bool(cp.any(unique_tags == mpoly_tag))

    # S3b: Find coordinate span ends
    span_kernels = _coord_span_end_kernels()
    d_coord_ends = cp.empty(n_features, dtype=cp.int64)
    _launch_kernel(runtime, span_kernels["coord_span_end"], n_features, (
        (ptr(d_depth), ptr(d_coord_positions), ptr(d_coord_ends),
         np.int32(n_features), n_i64),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_I32, KERNEL_PARAM_I64),
    ))

    # S3c: Count rings and coordinate pairs per feature.
    # The kernel output has different semantics per type:
    #   LineString/MultiPoint:       ring_counts = coord pairs
    #   Polygon/MultiLineString:     ring_counts = rings/parts, pair_counts = coord pairs
    #   MultiPolygon:                needs 3-level kernel (part/ring/pair counts)
    # For homogeneous Point we skip it entirely.
    d_ring_counts = None
    d_pair_counts = None
    d_all_geometry_offsets = None  # Polygon/MultiLineString ring/part-level offsets
    d_ring_offsets = None
    # MultiPolygon-specific arrays
    d_mpoly_part_counts = None
    d_mpoly_ring_counts = None
    d_mpoly_pair_counts = None
    d_mpoly_geom_offsets = None
    d_mpoly_part_offsets = None
    d_mpoly_ring_offsets = None

    if single_tag == FAMILY_TAGS[GeometryFamily.POINT]:
        # Point: every feature has exactly 1 coordinate pair
        d_effective_pairs = cp.ones(n_features, dtype=cp.int32)
    else:
        # Run counting kernel for all non-Point types.
        # The 2-level kernel produces correct results for LineString,
        # Polygon, MultiPoint, and MultiLineString.  Its output for
        # MultiPolygon features is ignored (overridden by mpoly kernel).
        ring_kernels = _ring_count_kernels()
        d_ring_counts = cp.empty(n_features, dtype=cp.int32)
        d_pair_counts = cp.empty(n_features, dtype=cp.int32)
        _launch_kernel(runtime, ring_kernels["count_rings_and_coords"], n_features, (
            (ptr(d_bytes), ptr(d_depth), ptr(d_coord_positions), ptr(d_coord_ends),
             ptr(d_ring_counts), ptr(d_pair_counts), np.int32(n_features)),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ))

        # Run 3-level counting kernel for MultiPolygon features
        if has_multipolygons:
            mpoly_kernels = _mpoly_count_kernels()
            d_mpoly_part_counts = cp.zeros(n_features, dtype=cp.int32)
            d_mpoly_ring_counts = cp.zeros(n_features, dtype=cp.int32)
            d_mpoly_pair_counts = cp.zeros(n_features, dtype=cp.int32)
            _launch_kernel(runtime, mpoly_kernels["count_mpoly_levels"], n_features, (
                (ptr(d_bytes), ptr(d_depth), ptr(d_coord_positions), ptr(d_coord_ends),
                 ptr(d_family_tags),
                 ptr(d_mpoly_part_counts), ptr(d_mpoly_ring_counts),
                 ptr(d_mpoly_pair_counts), np.int32(n_features), mpoly_tag),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I8),
            ))

        # S3d: Compute effective pair counts per feature based on type.
        # Point: 1
        # LineString/MultiPoint: ring_counts (2-level kernel)
        # Polygon/MultiLineString: pair_counts (2-level kernel)
        # MultiPolygon: mpoly_pair_counts (3-level kernel)
        pt_tag = np.int8(FAMILY_TAGS[GeometryFamily.POINT])
        ls_tag = np.int8(FAMILY_TAGS[GeometryFamily.LINESTRING])
        mpt_tag = np.int8(FAMILY_TAGS[GeometryFamily.MULTIPOINT])
        d_effective_pairs = cp.where(
            d_family_tags == pt_tag,
            np.int32(1),
            cp.where(
                (d_family_tags == ls_tag) | (d_family_tags == mpt_tag),
                d_ring_counts,
                d_pair_counts,
            ),
        )
        # Override for MultiPolygon if present
        if has_multipolygons:
            d_effective_pairs = cp.where(
                d_family_tags == mpoly_tag,
                d_mpoly_pair_counts,
                d_effective_pairs,
            )

    # Compute per-feature coordinate offsets in flat x/y
    d_feature_coord_offsets = cp.zeros(n_features + 1, dtype=cp.int32)
    cp.cumsum(d_effective_pairs, out=d_feature_coord_offsets[1:])
    total_pairs = int(d_feature_coord_offsets[-1].get())

    # S3e: Polygon/MultiLineString ring offsets (when polygons or mls present)
    mls_tag_val = np.int8(FAMILY_TAGS[GeometryFamily.MULTILINESTRING])
    has_ring_types = has_polygons or bool(cp.any(unique_tags == mls_tag_val))
    if has_ring_types and d_ring_counts is not None:
        d_pair_offset_starts = cp.empty(n_features, dtype=cp.int32)
        cp.cumsum(d_pair_counts, out=d_pair_offset_starts)
        d_pair_offset_starts = cp.concatenate(
            [cp.zeros(1, dtype=cp.int32), d_pair_offset_starts[:-1]]
        )

        d_all_geometry_offsets = cp.empty(n_features + 1, dtype=cp.int32)
        d_all_geometry_offsets[0] = 0
        cp.cumsum(d_ring_counts, out=d_all_geometry_offsets[1:])
        total_rings = int(d_all_geometry_offsets[-1].get())

        d_ring_offsets = cp.empty(total_rings + 1, dtype=cp.int32)
        d_ring_offsets[-1] = (d_pair_offset_starts[-1] + d_pair_counts[-1]) if n_features > 0 else 0

        d_ring_scatter_starts = d_all_geometry_offsets[:n_features].copy()
        scatter_kernels = _scatter_coords_kernels()
        _launch_kernel(runtime, scatter_kernels["scatter_ring_offsets"], n_features, (
            (ptr(d_bytes), ptr(d_depth), ptr(d_coord_positions), ptr(d_coord_ends),
             ptr(d_ring_scatter_starts), ptr(d_pair_offset_starts),
             ptr(d_ring_offsets), np.int32(n_features)),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ))
        del d_pair_offset_starts, d_ring_scatter_starts

    # S3f: MultiPolygon 3-level offset scatter
    if has_multipolygons and d_mpoly_part_counts is not None:
        # Compute prefix sums for part/ring counts (mpoly-local space)
        d_mp_part_psum = cp.empty(n_features, dtype=cp.int32)
        cp.cumsum(d_mpoly_part_counts, out=d_mp_part_psum)
        d_mp_part_offset_starts = cp.concatenate(
            [cp.zeros(1, dtype=cp.int32), d_mp_part_psum[:-1]]
        )
        total_mpoly_parts = int(d_mp_part_psum[-1])

        d_mp_ring_psum = cp.empty(n_features, dtype=cp.int32)
        cp.cumsum(d_mpoly_ring_counts, out=d_mp_ring_psum)
        d_mp_ring_offset_starts = cp.concatenate(
            [cp.zeros(1, dtype=cp.int32), d_mp_ring_psum[:-1]]
        )
        total_mpoly_rings = int(d_mp_ring_psum[-1])

        # For pair offset starts, use the GLOBAL feature coord offsets so that
        # ring_offsets index into the global flat coordinate array.  This is
        # critical for mixed-type files where MultiPolygon coordinates don't
        # start at index 0 in the flat array.
        d_mp_pair_offset_starts = d_feature_coord_offsets[:n_features].copy()

        # Geometry offsets: per-feature part count prefix sums
        d_mpoly_geom_offsets = cp.empty(n_features + 1, dtype=cp.int32)
        d_mpoly_geom_offsets[0] = 0
        d_mpoly_geom_offsets[1:] = d_mp_part_psum

        # Allocate part and ring offset arrays
        d_mpoly_part_offsets = cp.empty(total_mpoly_parts + 1, dtype=cp.int32)
        d_mpoly_ring_offsets = cp.empty(total_mpoly_rings + 1, dtype=cp.int32)
        # Write sentinel end values
        if total_mpoly_parts > 0:
            d_mpoly_part_offsets[-1] = total_mpoly_rings
        if total_mpoly_rings > 0:
            d_mpoly_ring_offsets[-1] = int(d_feature_coord_offsets[n_features])

        # Scatter offsets
        mpoly_scatter_k = _mpoly_scatter_kernels()
        _launch_kernel(runtime, mpoly_scatter_k["scatter_mpoly_offsets"], n_features, (
            (ptr(d_bytes), ptr(d_depth), ptr(d_coord_positions), ptr(d_coord_ends),
             ptr(d_family_tags),
             ptr(d_mp_part_offset_starts), ptr(d_mp_ring_offset_starts),
             ptr(d_mp_pair_offset_starts),
             ptr(d_mpoly_part_offsets), ptr(d_mpoly_ring_offsets),
             np.int32(n_features), mpoly_tag),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_I32, KERNEL_PARAM_I8),
        ))
        del d_mp_part_psum, d_mp_ring_psum
        del d_mp_part_offset_starts, d_mp_ring_offset_starts, d_mp_pair_offset_starts

    # S8: Find feature boundaries for property extraction.
    # Moved before S4 so that d_depth (n*4 bytes = 8.6 GB for a 2 GB file)
    # can be freed before the memory-intensive number extraction stage.
    fb_kernels = _feature_boundary_kernels()
    d_feat_start = cp.zeros(n, dtype=cp.uint8)
    d_feat_end = cp.zeros(n, dtype=cp.uint8)
    _launch_kernel(runtime, fb_kernels["find_feature_boundaries"], n, (
        (ptr(d_bytes), ptr(d_depth), ptr(d_feat_start), ptr(d_feat_end), n_i64),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I64),
    ))
    d_feat_start_pos = cp.flatnonzero(d_feat_start).astype(cp.int64)
    d_feat_end_pos = cp.flatnonzero(d_feat_end).astype(cp.int64) + 1
    del d_feat_start, d_feat_end

    # Free d_depth early: it is the single largest allocation (n*4 bytes,
    # ~8.6 GB for a 2 GB input file).  All consumers (type detection, span
    # end, ring counting, mpoly scatter, feature boundaries) have finished.
    del d_depth

    # S4: Find all number boundaries (with quote-state filter) via gpu_parse.
    d_is_start, d_is_end = number_boundaries(d_bytes, d_quote_parity)
    del d_quote_parity

    # Filter numbers to only those inside coordinate spans via gpu_parse.
    # The original GeoJSON kernel added +14 internally for the
    # "coordinates": key length; here we pass pre-computed starts.
    d_coord_span_starts = (d_coord_positions + 14).astype(cp.int64)
    d_in_coords = mark_spans(d_coord_span_starts, d_coord_ends, n)
    del d_coord_span_starts

    # Apply the coordinate-span mask in-place to minimise peak GPU memory.
    # The old code did this inline; extract_number_positions(d_mask=...)
    # would create temporaries while the caller still holds references to
    # the originals, adding ~4 GB peak overhead on a 2 GB file.
    d_is_start *= d_in_coords
    d_is_end *= d_in_coords
    del d_in_coords

    # Extract number positions (compact start/end arrays, half-open).
    d_starts, d_ends = extract_number_positions(d_is_start, d_is_end)
    del d_is_start, d_is_end

    # S5: Parse ASCII floats via gpu_parse.
    d_coords = parse_ascii_floats(d_bytes, d_starts, d_ends)
    del d_starts, d_ends

    # S6: Split into x, y (zero-copy views)
    d_x = d_coords[0::2]
    d_y = d_coords[1::2]

    # Verify coordinate count matches expected pairs
    if total_pairs > 0 and len(d_x) != total_pairs:
        if len(d_x) > total_pairs:
            d_x = d_x[:total_pairs].copy()
            d_y = d_y[:total_pairs].copy()
        else:
            pad_x = cp.zeros(total_pairs - len(d_x), dtype=cp.float64)
            pad_y = cp.zeros(total_pairs - len(d_y), dtype=cp.float64)
            d_x = cp.concatenate([d_x, pad_x])
            d_y = cp.concatenate([d_y, pad_y])

    d_x = cp.ascontiguousarray(d_x)
    d_y = cp.ascontiguousarray(d_y)

    # S6b: Fused CRS transform (device-only, no host round-trip).
    # GeoJSON is EPSG:4326 per RFC 7946.  If target_crs is set and
    # differs from 4326, transform in-place before assembly.
    if target_crs is not None:
        from vibespatial.io.gpu_parse.transform import transform_coordinates_inplace

        transform_coordinates_inplace(d_x, d_y, src_crs="EPSG:4326", dst_crs=target_crs)

    del d_bytes, d_coord_positions, d_coord_ends
    del d_coords

    h_feat_starts = cp.asnumpy(d_feat_start_pos)
    h_feat_ends = cp.asnumpy(d_feat_end_pos)
    del d_feat_start_pos, d_feat_end_pos

    # S7: Family-aware assembly
    if is_homogeneous:
        owned = _assemble_homogeneous(
            single_tag, n_features, d_x, d_y,
            d_effective_pairs, d_feature_coord_offsets,
            d_ring_counts, d_all_geometry_offsets, d_ring_offsets,
            d_mpoly_geom_offsets=d_mpoly_geom_offsets,
            d_mpoly_part_offsets=d_mpoly_part_offsets,
            d_mpoly_ring_offsets=d_mpoly_ring_offsets,
        )
    else:
        owned = _assemble_mixed(
            n_features, d_x, d_y, d_family_tags,
            d_effective_pairs, d_feature_coord_offsets,
            d_ring_counts, d_pair_counts,
            d_all_geometry_offsets, d_ring_offsets,
            d_mpoly_part_counts=d_mpoly_part_counts,
            d_mpoly_ring_counts=d_mpoly_ring_counts,
            d_mpoly_geom_offsets=d_mpoly_geom_offsets,
            d_mpoly_part_offsets=d_mpoly_part_offsets,
            d_mpoly_ring_offsets=d_mpoly_ring_offsets,
        )

    return GeoJSONGpuResult(
        owned=owned,
        n_features=n_features,
        host_bytes=host_bytes,
        feature_starts=h_feat_starts,
        feature_ends=h_feat_ends,
    )


def _assemble_homogeneous(
    tag, n_features, d_x, d_y,
    d_effective_pairs, d_feature_coord_offsets,
    d_ring_counts, d_all_geometry_offsets, d_ring_offsets,
    *, d_mpoly_geom_offsets=None, d_mpoly_part_offsets=None,
    d_mpoly_ring_offsets=None,
):
    """Build single-family OwnedGeometryArray for homogeneous files."""
    d_empty_mask = (d_effective_pairs == 0)
    d_validity = ~d_empty_mask

    if tag == FAMILY_TAGS[GeometryFamily.POINT]:
        d_geom_offsets = cp.arange(n_features + 1, dtype=cp.int32)
        return _build_device_single_family_owned(
            family=GeometryFamily.POINT,
            validity_device=d_validity,
            x_device=d_x,
            y_device=d_y,
            geometry_offsets_device=d_geom_offsets,
            empty_mask_device=d_empty_mask,
            detail="GPU byte-classification GeoJSON parse (Point)",
        )

    if tag == FAMILY_TAGS[GeometryFamily.LINESTRING]:
        return _build_device_single_family_owned(
            family=GeometryFamily.LINESTRING,
            validity_device=d_validity,
            x_device=d_x,
            y_device=d_y,
            geometry_offsets_device=d_feature_coord_offsets,
            empty_mask_device=d_empty_mask,
            detail="GPU byte-classification GeoJSON parse (LineString)",
        )

    if tag == FAMILY_TAGS[GeometryFamily.POLYGON]:
        d_pg_empty = (d_all_geometry_offsets[1:] == d_all_geometry_offsets[:-1])
        d_pg_validity = ~d_pg_empty
        return _build_device_single_family_owned(
            family=GeometryFamily.POLYGON,
            validity_device=d_pg_validity,
            x_device=d_x,
            y_device=d_y,
            geometry_offsets_device=d_all_geometry_offsets,
            empty_mask_device=d_pg_empty,
            ring_offsets_device=d_ring_offsets,
            detail="GPU byte-classification GeoJSON parse (Polygon)",
        )

    if tag == FAMILY_TAGS[GeometryFamily.MULTIPOINT]:
        # MultiPoint: geometry_offsets = per-feature coord count (same as LineString layout)
        return _build_device_single_family_owned(
            family=GeometryFamily.MULTIPOINT,
            validity_device=d_validity,
            x_device=d_x,
            y_device=d_y,
            geometry_offsets_device=d_feature_coord_offsets,
            empty_mask_device=d_empty_mask,
            detail="GPU byte-classification GeoJSON parse (MultiPoint)",
        )

    if tag == FAMILY_TAGS[GeometryFamily.MULTILINESTRING]:
        # MultiLineString: same nesting as Polygon.
        # d_all_geometry_offsets = per-feature part count prefix sums (into ring_offsets)
        # d_ring_offsets = per-part coord count prefix sums (into x/y)
        # For MultiLineString: geometry_offsets -> parts, part_offsets -> coords
        d_mls_empty = (d_all_geometry_offsets[1:] == d_all_geometry_offsets[:-1])
        d_mls_validity = ~d_mls_empty
        return _build_device_single_family_owned(
            family=GeometryFamily.MULTILINESTRING,
            validity_device=d_mls_validity,
            x_device=d_x,
            y_device=d_y,
            geometry_offsets_device=d_all_geometry_offsets,
            empty_mask_device=d_mls_empty,
            part_offsets_device=d_ring_offsets,
            detail="GPU byte-classification GeoJSON parse (MultiLineString)",
        )

    if tag == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]:
        # MultiPolygon: 3-level offsets
        d_mp_empty = (d_mpoly_geom_offsets[1:] == d_mpoly_geom_offsets[:-1])
        d_mp_validity = ~d_mp_empty
        return _build_device_single_family_owned(
            family=GeometryFamily.MULTIPOLYGON,
            validity_device=d_mp_validity,
            x_device=d_x,
            y_device=d_y,
            geometry_offsets_device=d_mpoly_geom_offsets,
            empty_mask_device=d_mp_empty,
            part_offsets_device=d_mpoly_part_offsets,
            ring_offsets_device=d_mpoly_ring_offsets,
            detail="GPU byte-classification GeoJSON parse (MultiPolygon)",
        )

    raise ValueError(f"Unsupported family tag {tag} in homogeneous assembly")


def _assemble_mixed(
    n_features, d_x, d_y, d_family_tags,
    d_effective_pairs, d_feature_coord_offsets,
    d_ring_counts, d_pair_counts,
    d_all_geometry_offsets, d_ring_offsets,
    *, d_mpoly_part_counts=None, d_mpoly_ring_counts=None,
    d_mpoly_geom_offsets=None, d_mpoly_part_offsets=None,
    d_mpoly_ring_offsets=None,
):
    """Build multi-family OwnedGeometryArray for mixed-type files."""
    # Partition features by family
    family_devices = {}
    partitions = {}  # tag_val -> rows (cached for reuse in tag assignment)
    tag_map = [
        (FAMILY_TAGS[GeometryFamily.POINT], GeometryFamily.POINT),
        (FAMILY_TAGS[GeometryFamily.LINESTRING], GeometryFamily.LINESTRING),
        (FAMILY_TAGS[GeometryFamily.POLYGON], GeometryFamily.POLYGON),
        (FAMILY_TAGS[GeometryFamily.MULTIPOINT], GeometryFamily.MULTIPOINT),
        (FAMILY_TAGS[GeometryFamily.MULTILINESTRING], GeometryFamily.MULTILINESTRING),
        (FAMILY_TAGS[GeometryFamily.MULTIPOLYGON], GeometryFamily.MULTIPOLYGON),
    ]

    # Pre-compute coords_2d once for gather operations
    coords_2d = cp.column_stack([d_x, d_y]) if d_x.size > 0 else cp.empty((0, 2), dtype=cp.float64)

    for tag_val, family in tag_map:
        rows = cp.flatnonzero(d_family_tags == tag_val).astype(cp.int32)
        if rows.size == 0:
            continue
        partitions[tag_val] = rows

        n_f = rows.size

        if family == GeometryFamily.POINT:
            pt_starts = d_feature_coord_offsets[rows]
            pt_x = d_x[pt_starts]
            pt_y = d_y[pt_starts]
            family_devices[family] = DeviceFamilyGeometryBuffer(
                family=family,
                x=cp.ascontiguousarray(pt_x),
                y=cp.ascontiguousarray(pt_y),
                geometry_offsets=cp.arange(n_f + 1, dtype=cp.int32),
                empty_mask=cp.zeros(n_f, dtype=cp.bool_),
            )

        elif family == GeometryFamily.LINESTRING:
            gathered, ls_geom_offsets = _device_gather_offset_slices(
                coords_2d, d_feature_coord_offsets, rows,
            )
            family_devices[family] = DeviceFamilyGeometryBuffer(
                family=family,
                x=cp.ascontiguousarray(gathered[:, 0]) if gathered.size else cp.empty(0, dtype=cp.float64),
                y=cp.ascontiguousarray(gathered[:, 1]) if gathered.size else cp.empty(0, dtype=cp.float64),
                geometry_offsets=ls_geom_offsets,
                empty_mask=(ls_geom_offsets[1:] == ls_geom_offsets[:-1]),
            )

        elif family == GeometryFamily.POLYGON:
            ring_indices, pg_geom_offsets = _device_gather_offset_slices(
                cp.arange(d_ring_offsets.size, dtype=cp.int32),
                d_all_geometry_offsets,
                rows,
            )
            pg_coords, pg_ring_offsets = _device_gather_offset_slices(
                coords_2d, d_ring_offsets, ring_indices,
            )
            family_devices[family] = DeviceFamilyGeometryBuffer(
                family=family,
                x=cp.ascontiguousarray(pg_coords[:, 0]) if pg_coords.size else cp.empty(0, dtype=cp.float64),
                y=cp.ascontiguousarray(pg_coords[:, 1]) if pg_coords.size else cp.empty(0, dtype=cp.float64),
                geometry_offsets=pg_geom_offsets,
                empty_mask=(pg_geom_offsets[1:] == pg_geom_offsets[:-1]),
                ring_offsets=pg_ring_offsets,
            )

        elif family == GeometryFamily.MULTIPOINT:
            # MultiPoint: same layout as LineString (geometry_offsets -> coords)
            gathered, mp_geom_offsets = _device_gather_offset_slices(
                coords_2d, d_feature_coord_offsets, rows,
            )
            family_devices[family] = DeviceFamilyGeometryBuffer(
                family=family,
                x=cp.ascontiguousarray(gathered[:, 0]) if gathered.size else cp.empty(0, dtype=cp.float64),
                y=cp.ascontiguousarray(gathered[:, 1]) if gathered.size else cp.empty(0, dtype=cp.float64),
                geometry_offsets=mp_geom_offsets,
                empty_mask=(mp_geom_offsets[1:] == mp_geom_offsets[:-1]),
            )

        elif family == GeometryFamily.MULTILINESTRING:
            # MultiLineString: same nesting as Polygon.
            # d_all_geometry_offsets indexes ring/part counts, d_ring_offsets indexes coords.
            # For MLS: geometry_offsets -> parts, part_offsets -> coords
            part_indices, mls_geom_offsets = _device_gather_offset_slices(
                cp.arange(d_ring_offsets.size, dtype=cp.int32),
                d_all_geometry_offsets,
                rows,
            )
            mls_coords, mls_part_offsets = _device_gather_offset_slices(
                coords_2d, d_ring_offsets, part_indices,
            )
            family_devices[family] = DeviceFamilyGeometryBuffer(
                family=family,
                x=cp.ascontiguousarray(mls_coords[:, 0]) if mls_coords.size else cp.empty(0, dtype=cp.float64),
                y=cp.ascontiguousarray(mls_coords[:, 1]) if mls_coords.size else cp.empty(0, dtype=cp.float64),
                geometry_offsets=mls_geom_offsets,
                empty_mask=(mls_geom_offsets[1:] == mls_geom_offsets[:-1]),
                part_offsets=mls_part_offsets,
            )

        elif family == GeometryFamily.MULTIPOLYGON:
            # MultiPolygon: 3-level offsets (geometry -> parts -> rings -> coords)
            # Gather part indices for these rows
            part_indices, mpg_geom_offsets = _device_gather_offset_slices(
                cp.arange(d_mpoly_part_offsets.size, dtype=cp.int32),
                d_mpoly_geom_offsets,
                rows,
            )
            # Gather ring indices for these parts
            ring_indices, mpg_part_offsets = _device_gather_offset_slices(
                cp.arange(d_mpoly_ring_offsets.size, dtype=cp.int32),
                d_mpoly_part_offsets,
                part_indices,
            )
            # Gather coords for these rings
            mpg_coords, mpg_ring_offsets = _device_gather_offset_slices(
                coords_2d, d_mpoly_ring_offsets, ring_indices,
            )
            family_devices[family] = DeviceFamilyGeometryBuffer(
                family=family,
                x=cp.ascontiguousarray(mpg_coords[:, 0]) if mpg_coords.size else cp.empty(0, dtype=cp.float64),
                y=cp.ascontiguousarray(mpg_coords[:, 1]) if mpg_coords.size else cp.empty(0, dtype=cp.float64),
                geometry_offsets=mpg_geom_offsets,
                empty_mask=(mpg_geom_offsets[1:] == mpg_geom_offsets[:-1]),
                part_offsets=mpg_part_offsets,
                ring_offsets=mpg_ring_offsets,
            )

    # Build tags and family_row_offsets (reuse cached partitions)
    d_validity = cp.ones(n_features, dtype=cp.bool_)
    d_family_row_offsets = cp.full(n_features, -1, dtype=cp.int32)
    for tag_val, rows in partitions.items():
        d_family_row_offsets[rows] = cp.arange(int(rows.size), dtype=cp.int32)

    return _build_device_mixed_owned(
        validity_device=d_validity,
        tags_device=d_family_tags,
        family_row_offsets_device=d_family_row_offsets,
        family_devices=family_devices,
        detail="GPU byte-classification GeoJSON parse (mixed)",
    )
