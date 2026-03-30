"""NVRTC kernel sources for GPU FlatGeobuf decoder."""

from __future__ import annotations

_FB_DEVICE_HELPERS = r"""
/* ---- FlatBuffer device helpers ---------------------------------------- */

__device__ int fb_read_i16(const unsigned char* __restrict__ buf, long long off) {
    return (int)(short)(buf[off] | (buf[off+1] << 8));
}

__device__ unsigned int fb_read_u16(const unsigned char* __restrict__ buf, long long off) {
    return (unsigned int)(buf[off] | (buf[off+1] << 8));
}

__device__ int fb_read_i32(const unsigned char* __restrict__ buf, long long off) {
    int v;
    memcpy(&v, buf + off, 4);
    return v;
}

__device__ unsigned int fb_read_u32(const unsigned char* __restrict__ buf, long long off) {
    unsigned int v;
    memcpy(&v, buf + off, 4);
    return v;
}

__device__ double fb_read_f64(const unsigned char* __restrict__ buf, long long off) {
    double v;
    memcpy(&v, buf + off, 8);
    return v;
}

/* Navigate to a FlatBuffer table field.
 * Returns the absolute offset of the field data, or -1 if not present.
 *
 * table_off: absolute offset of the table in the buffer
 * field_idx: 0-based field index
 */
__device__ long long fb_field(
    const unsigned char* __restrict__ buf,
    long long table_off,
    int field_idx
) {
    int vtable_soffset = fb_read_i32(buf, table_off);
    long long vtable_off = table_off - (long long)vtable_soffset;
    unsigned int vtable_size = fb_read_u16(buf, vtable_off);
    unsigned int field_voff_pos = 4 + field_idx * 2;
    if (field_voff_pos >= vtable_size) return -1;
    unsigned int field_voff = fb_read_u16(buf, vtable_off + field_voff_pos);
    if (field_voff == 0) return -1;
    return table_off + (long long)field_voff;
}

/* Read a FlatBuffer vector length and return the absolute offset of
 * element data (past the length prefix). vec_len is set to the length.
 *
 * field_off: the absolute offset of the field that contains the
 *            relative offset to the vector.
 */
__device__ long long fb_vec(
    const unsigned char* __restrict__ buf,
    long long field_off,
    int* vec_len
) {
    int rel = fb_read_i32(buf, field_off);
    long long vec_abs = field_off + (long long)rel;
    *vec_len = fb_read_i32(buf, vec_abs);
    return vec_abs + 4;
}

/* Navigate to the geometry table within a Feature FlatBuffer.
 *
 * feat_off: absolute byte offset of the feature in the file.
 *           The feature is: [uint32 size] [FlatBuffer data].
 *           The FlatBuffer root table starts at feat_off + 4 + read_u32(feat_off + 4).
 *
 * Returns the absolute offset of the Geometry table, or -1 if not present.
 *
 * Feature table fields:
 *   0: geometry (Geometry table)
 *   1: properties ([ubyte])
 *   2: columns ([Column])
 */
__device__ long long fgb_geometry_table(
    const unsigned char* __restrict__ buf,
    long long feat_off
) {
    /* Skip the 4-byte size prefix to get to the FlatBuffer root */
    long long fb_start = feat_off + 4;
    long long root_off = fb_start + (long long)fb_read_u32(buf, fb_start);

    /* Field 0 = geometry (offset to Geometry table) */
    long long geom_field = fb_field(buf, root_off, 0);
    if (geom_field < 0) return -1;

    /* The field contains a relative offset to the Geometry table */
    int geom_rel = fb_read_i32(buf, geom_field);
    return geom_field + (long long)geom_rel;
}

/* Navigate to the xy vector within a Geometry table.
 *
 * Geometry table fields:
 *   0: ends ([uint32])
 *   1: xy ([double])
 *   2: z ([double])
 *   3: m ([double])
 *   4: t ([double])
 *   5: tm ([uint64])
 *   6: type (uint8)
 *   7: parts ([Geometry])
 *
 * Returns the absolute offset of the xy vector data (past length prefix),
 * and sets n_doubles to the vector length (number of doubles = 2 * n_coords).
 * Returns -1 if geometry or xy is not present.
 */
__device__ long long fgb_xy_vec(
    const unsigned char* __restrict__ buf,
    long long geom_off,
    int* n_doubles
) {
    long long xy_field = fb_field(buf, geom_off, 1);
    if (xy_field < 0) { *n_doubles = 0; return -1; }
    return fb_vec(buf, xy_field, n_doubles);
}

/* Navigate to the ends vector within a Geometry table.
 * Returns the absolute offset of the ends vector data (past length prefix),
 * and sets n_ends to the vector length.
 * Returns -1 if ends is not present.
 */
__device__ long long fgb_ends_vec(
    const unsigned char* __restrict__ buf,
    long long geom_off,
    int* n_ends
) {
    long long ends_field = fb_field(buf, geom_off, 0);
    if (ends_field < 0) { *n_ends = 0; return -1; }
    return fb_vec(buf, ends_field, n_ends);
}

/* Navigate to the parts vector within a Geometry table (for Multi* types).
 *
 * Returns the absolute offset of the parts vector data (past length prefix),
 * and sets n_parts to the vector length.
 * Returns -1 if parts is not present.
 */
__device__ long long fgb_parts_vec(
    const unsigned char* __restrict__ buf,
    long long geom_off,
    int* n_parts
) {
    long long parts_field = fb_field(buf, geom_off, 7);
    if (parts_field < 0) { *n_parts = 0; return -1; }
    return fb_vec(buf, parts_field, n_parts);
}
"""

_FGB_DECODE_POINTS_SOURCE = (
    _FB_DEVICE_HELPERS
    + r"""
extern "C" __global__ void __launch_bounds__(256, 4)
fgb_decode_points(
    const unsigned char* __restrict__ data,
    const long long*     __restrict__ feature_offsets,
    double*              __restrict__ out_x,
    double*              __restrict__ out_y,
    const int n_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_features) return;

    long long feat_off = feature_offsets[idx];
    long long geom = fgb_geometry_table(data, feat_off);
    if (geom < 0) {
        out_x[idx] = 0.0;
        out_y[idx] = 0.0;
        return;
    }

    int n_doubles = 0;
    long long xy_data = fgb_xy_vec(data, geom, &n_doubles);
    if (xy_data < 0 || n_doubles < 2) {
        out_x[idx] = 0.0;
        out_y[idx] = 0.0;
        return;
    }

    out_x[idx] = fb_read_f64(data, xy_data);
    out_y[idx] = fb_read_f64(data, xy_data + 8);
}
"""
)

_FGB_COUNT_COORDS_SOURCE = (
    _FB_DEVICE_HELPERS
    + r"""
extern "C" __global__ void __launch_bounds__(256, 4)
fgb_count_coords(
    const unsigned char* __restrict__ data,
    const long long*     __restrict__ feature_offsets,
    int*                 __restrict__ out_coord_counts,
    int*                 __restrict__ out_ring_counts,
    int*                 __restrict__ out_part_counts,
    const int n_features,
    const int geom_type
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_features) return;

    long long feat_off = feature_offsets[idx];
    long long geom = fgb_geometry_table(data, feat_off);
    if (geom < 0) {
        out_coord_counts[idx] = 0;
        out_ring_counts[idx] = 0;
        out_part_counts[idx] = 0;
        return;
    }

    /* For simple geometry types (Point/LineString/Polygon):
     * xy has all the coordinates, ends has ring boundaries.
     * For Multi* types: parts vector has sub-geometries.
     */
    if (geom_type <= 5) {
        /* Flat-xy path: Point(1), LineString(2), Polygon(3),
         * MultiPoint(4), MultiLineString(5).
         * These types store all coordinates directly in xy with
         * ends[] for ring/part boundaries.  Only MultiPolygon(6)
         * uses the parts vector for sub-geometries. */
        int n_doubles = 0;
        fgb_xy_vec(data, geom, &n_doubles);
        int n_coords = n_doubles >> 1;  /* / 2 */
        out_coord_counts[idx] = n_coords;

        int n_ends = 0;
        fgb_ends_vec(data, geom, &n_ends);
        /* For Polygon: n_ends = number of rings. For LineString: n_ends = 0 or 1. */
        out_ring_counts[idx] = (n_ends > 0) ? n_ends : (n_coords > 0 ? 1 : 0);
        out_part_counts[idx] = (n_coords > 0) ? 1 : 0;
    } else {
        /* MultiPolygon(6): iterate over parts sub-geometries */
        int n_parts = 0;
        long long parts_data = fgb_parts_vec(data, geom, &n_parts);
        if (n_parts == 0) {
            out_coord_counts[idx] = 0;
            out_ring_counts[idx] = 0;
            out_part_counts[idx] = 0;
            return;
        }

        int total_coords = 0;
        int total_rings = 0;
        for (int p = 0; p < n_parts; p++) {
            /* Each part is a relative offset to a Geometry table */
            long long part_rel_off = parts_data + (long long)(p * 4);
            int part_rel = fb_read_u32(data, part_rel_off);
            long long part_geom = part_rel_off + (long long)part_rel;

            int n_doubles = 0;
            fgb_xy_vec(data, part_geom, &n_doubles);
            total_coords += n_doubles >> 1;

            int n_ends = 0;
            fgb_ends_vec(data, part_geom, &n_ends);
            total_rings += (n_ends > 0) ? n_ends : ((n_doubles > 0) ? 1 : 0);
        }

        out_coord_counts[idx] = total_coords;
        out_ring_counts[idx] = total_rings;
        out_part_counts[idx] = n_parts;
    }
}
"""
)

_FGB_GATHER_COORDS_SOURCE = (
    _FB_DEVICE_HELPERS
    + r"""
extern "C" __global__ void __launch_bounds__(256, 4)
fgb_gather_coords(
    const unsigned char* __restrict__ data,
    const long long*     __restrict__ feature_offsets,
    const int*           __restrict__ coord_offsets,
    const int*           __restrict__ ring_offsets,
    const int*           __restrict__ part_offsets,
    double*              __restrict__ out_x,
    double*              __restrict__ out_y,
    int*                 __restrict__ out_ring_ends,
    const int n_features,
    const int geom_type
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_features) return;

    long long feat_off = feature_offsets[idx];
    long long geom = fgb_geometry_table(data, feat_off);
    if (geom < 0) return;

    int write_coord = coord_offsets[idx];
    int write_ring = ring_offsets[idx];

    if (geom_type <= 5) {
        /* Flat-xy path: Point(1), LineString(2), Polygon(3),
         * MultiPoint(4), MultiLineString(5).
         * These types store all coordinates directly in xy with
         * ends[] for ring/part boundaries.  Only MultiPolygon(6)
         * uses the parts vector for sub-geometries. */
        int n_doubles = 0;
        long long xy_data = fgb_xy_vec(data, geom, &n_doubles);
        int n_coords = n_doubles >> 1;

        /* Copy coordinates to SoA output */
        for (int c = 0; c < n_coords; c++) {
            out_x[write_coord + c] = fb_read_f64(data, xy_data + (long long)(c * 16));
            out_y[write_coord + c] = fb_read_f64(data, xy_data + (long long)(c * 16 + 8));
        }

        /* Write ring end indices (relative to this geometry's coord start) */
        int n_ends = 0;
        long long ends_data = fgb_ends_vec(data, geom, &n_ends);
        if (n_ends > 0) {
            for (int r = 0; r < n_ends; r++) {
                int ring_end = (int)fb_read_u32(data, ends_data + (long long)(r * 4));
                out_ring_ends[write_ring + r] = write_coord + ring_end;
            }
        } else if (n_coords > 0) {
            /* LineString or single-ring: one implicit ring */
            out_ring_ends[write_ring] = write_coord + n_coords;
        }
    } else {
        /* MultiPolygon(6): iterate over parts sub-geometries */
        int n_parts = 0;
        long long parts_data = fgb_parts_vec(data, geom, &n_parts);

        int cur_coord = write_coord;
        int cur_ring = write_ring;

        for (int p = 0; p < n_parts; p++) {
            long long part_rel_off = parts_data + (long long)(p * 4);
            int part_rel = fb_read_u32(data, part_rel_off);
            long long part_geom = part_rel_off + (long long)part_rel;

            int n_doubles = 0;
            long long xy_data = fgb_xy_vec(data, part_geom, &n_doubles);
            int n_coords = n_doubles >> 1;

            for (int c = 0; c < n_coords; c++) {
                out_x[cur_coord + c] = fb_read_f64(data, xy_data + (long long)(c * 16));
                out_y[cur_coord + c] = fb_read_f64(data, xy_data + (long long)(c * 16 + 8));
            }

            int n_ends = 0;
            long long ends_data = fgb_ends_vec(data, part_geom, &n_ends);
            if (n_ends > 0) {
                for (int r = 0; r < n_ends; r++) {
                    int ring_end = (int)fb_read_u32(data, ends_data + (long long)(r * 4));
                    out_ring_ends[cur_ring + r] = cur_coord + ring_end;
                }
                cur_ring += n_ends;
            } else if (n_coords > 0) {
                out_ring_ends[cur_ring] = cur_coord + n_coords;
                cur_ring += 1;
            }

            cur_coord += n_coords;
        }
    }
}
"""
)

_POINT_KERNEL_NAMES = ("fgb_decode_points",)

_COUNT_KERNEL_NAMES = ("fgb_count_coords",)

_GATHER_KERNEL_NAMES = ("fgb_gather_coords",)

