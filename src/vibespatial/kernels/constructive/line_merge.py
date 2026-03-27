"""NVRTC kernels for line_merge: merge connected LineStrings into longer chains.

ADR-0033: Tier 1 NVRTC for geometry-specific inner loops (endpoint graph
construction and chain-following).  Tier 3a CCCL for prefix sum (offset
computation).

ADR-0002: CONSTRUCTIVE class -- fp64 uniform precision.  Coordinates are
exact subsets of input (no arithmetic on coordinate values), so the
precision plan stays fp64 on all devices.

Two-pass count-scatter architecture:
    Pass 1 (count):   One thread per geometry row.  Builds an in-register
        endpoint adjacency graph, follows chains, and counts the total
        number of output coordinates.
    Pass 2 (scatter):  Same graph-walk logic, but writes coordinates into
        pre-allocated output buffers at the offsets computed by prefix sum.

The line-merge graph per geometry is small (bounded by MAX_SEGMENTS), so
all adjacency data fits in thread-local arrays.  No shared memory needed.
"""

from __future__ import annotations

# Maximum segments (LineString parts) per geometry that the kernel supports.
# Geometries exceeding this are left unmerged (parts copied as-is).
MAX_SEGMENTS = 256

# Maximum coordinates per geometry that the kernel supports in the output.
# This bounds the chain-walk output buffer.
MAX_COORDS = 8192


_LINE_MERGE_KERNEL_SOURCE = r"""
/* ------------------------------------------------------------------ */
/* line_merge NVRTC kernel                                            */
/*                                                                    */
/* One thread per geometry row.  Each thread builds a local endpoint  */
/* adjacency graph from the LineString parts of the input geometry,   */
/* follows chains, and outputs merged LineString coordinates.         */
/*                                                                    */
/* For MultiLineString: merge connected parts.                        */
/* For LineString: single segment, returned as-is (trivial merge).    */
/* Others: pass through as empty.                                     */
/* ------------------------------------------------------------------ */

#define MAX_SEGMENTS """ + str(MAX_SEGMENTS) + r"""
#define MAX_COORDS """ + str(MAX_COORDS) + r"""
#define COORD_EQ_TOL 1e-12

/* ------------------------------------------------------------------ */
/* Device helpers                                                      */
/* ------------------------------------------------------------------ */

__device__ bool coords_equal(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return (dx * dx + dy * dy) < (COORD_EQ_TOL * COORD_EQ_TOL);
}

/* Find or insert a node in the node table.  Returns the node index. */
__device__ int find_or_insert_node(
    double* node_x, double* node_y, int* node_count,
    double x, double y, int max_nodes
) {
    for (int i = 0; i < *node_count; i++) {
        if (coords_equal(node_x[i], node_y[i], x, y)) {
            return i;
        }
    }
    if (*node_count >= max_nodes) return -1;  /* overflow */
    int idx = *node_count;
    node_x[idx] = x;
    node_y[idx] = y;
    (*node_count)++;
    return idx;
}

/* ------------------------------------------------------------------ */
/* Count kernel: count output coordinates per geometry                 */
/* ------------------------------------------------------------------ */

extern "C" __global__ void __launch_bounds__(128, 4)
line_merge_count(
    /* MultiLineString family buffers */
    const double* __restrict__ mls_x,
    const double* __restrict__ mls_y,
    const int* __restrict__ mls_geom_off,   /* row -> parts */
    const int* __restrict__ mls_part_off,   /* part -> coords */
    const int mls_row_count,

    /* LineString family buffers */
    const double* __restrict__ ls_x,
    const double* __restrict__ ls_y,
    const int* __restrict__ ls_geom_off,    /* row -> coords */
    const int ls_row_count,

    /* Row mapping */
    const int* __restrict__ global_rows,    /* global row indices for this launch */
    const int* __restrict__ family_codes,   /* 0=MLS, 1=LS */
    const int* __restrict__ fam_local_rows, /* family-local row index */
    const int directed,                     /* 1=directed, 0=undirected */

    /* Output */
    int* __restrict__ out_coord_counts,     /* per-thread output coord count */
    int* __restrict__ out_part_counts,      /* per-thread output part count */
    const int n_rows
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rows) return;

    const int fam_code = family_codes[tid];
    const int fam_row = fam_local_rows[tid];

    /* Determine segments (parts) for this geometry */
    int n_segments = 0;
    int seg_starts[MAX_SEGMENTS];  /* coordinate start index per segment */
    int seg_lengths[MAX_SEGMENTS]; /* number of coordinates per segment */
    const double* src_x;
    const double* src_y;

    if (fam_code == 0) {
        /* MultiLineString */
        src_x = mls_x;
        src_y = mls_y;
        int part_begin = mls_geom_off[fam_row];
        int part_end = mls_geom_off[fam_row + 1];
        n_segments = part_end - part_begin;
        if (n_segments > MAX_SEGMENTS) {
            /* Too many segments -- output as-is (count all coords) */
            int total = 0;
            for (int p = part_begin; p < part_end; p++) {
                total += mls_part_off[p + 1] - mls_part_off[p];
            }
            out_coord_counts[tid] = total;
            out_part_counts[tid] = n_segments;
            return;
        }
        for (int i = 0; i < n_segments; i++) {
            seg_starts[i] = mls_part_off[part_begin + i];
            seg_lengths[i] = mls_part_off[part_begin + i + 1] - seg_starts[i];
        }
    } else {
        /* LineString -- single segment, return as-is */
        src_x = ls_x;
        src_y = ls_y;
        int coord_begin = ls_geom_off[fam_row];
        int coord_end = ls_geom_off[fam_row + 1];
        int n_coords = coord_end - coord_begin;
        out_coord_counts[tid] = n_coords;
        out_part_counts[tid] = (n_coords > 0) ? 1 : 0;
        return;
    }

    if (n_segments == 0) {
        out_coord_counts[tid] = 0;
        out_part_counts[tid] = 0;
        return;
    }
    if (n_segments == 1) {
        out_coord_counts[tid] = seg_lengths[0];
        out_part_counts[tid] = (seg_lengths[0] > 0) ? 1 : 0;
        return;
    }

    /* ---------------------------------------------------------- */
    /* Build endpoint graph                                        */
    /* ---------------------------------------------------------- */
    /* Each segment has a start node and end node.                 */
    /* Node table: unique (x,y) endpoints.                         */
    /* Adjacency: for each node, list of incident segment edges.   */
    /* Degree: count of incident segments per node.                */
    /* ---------------------------------------------------------- */

    double node_x[MAX_SEGMENTS * 2];
    double node_y[MAX_SEGMENTS * 2];
    int node_count = 0;

    int seg_start_node[MAX_SEGMENTS];
    int seg_end_node[MAX_SEGMENTS];

    for (int s = 0; s < n_segments; s++) {
        if (seg_lengths[s] < 2) {
            /* Degenerate segment (0 or 1 coords) -- skip */
            seg_start_node[s] = -1;
            seg_end_node[s] = -1;
            continue;
        }
        double sx = src_x[seg_starts[s]];
        double sy = src_y[seg_starts[s]];
        double ex = src_x[seg_starts[s] + seg_lengths[s] - 1];
        double ey = src_y[seg_starts[s] + seg_lengths[s] - 1];

        seg_start_node[s] = find_or_insert_node(node_x, node_y, &node_count,
                                                  sx, sy, MAX_SEGMENTS * 2);
        seg_end_node[s] = find_or_insert_node(node_x, node_y, &node_count,
                                                ex, ey, MAX_SEGMENTS * 2);
    }

    /* Compute node degree */
    int node_degree[MAX_SEGMENTS * 2];
    for (int i = 0; i < node_count; i++) node_degree[i] = 0;
    for (int s = 0; s < n_segments; s++) {
        if (seg_start_node[s] < 0) continue;
        if (directed) {
            /* Directed: only count outgoing (start) and incoming (end) */
            node_degree[seg_end_node[s]]++;    /* incoming at end */
            node_degree[seg_start_node[s]]++;  /* outgoing at start */
        } else {
            node_degree[seg_start_node[s]]++;
            node_degree[seg_end_node[s]]++;
        }
    }

    /* ---------------------------------------------------------- */
    /* Chain following                                              */
    /* ---------------------------------------------------------- */
    /* Strategy:                                                   */
    /*   1. Find chain starts: degree-1 nodes (for open paths)     */
    /*      or any unvisited node (for rings/closed chains).       */
    /*   2. Walk from start along connected segments.              */
    /*   3. Count coordinates in each chain.                       */
    /* ---------------------------------------------------------- */

    unsigned char seg_visited[MAX_SEGMENTS];
    for (int s = 0; s < n_segments; s++) seg_visited[s] = 0;

    int total_coords = 0;
    int total_parts = 0;

    /* Helper: find an unvisited segment incident to a node.       */
    /* For directed mode: from a node, find segment whose start    */
    /*   node matches (outgoing).                                  */
    /* For undirected mode: find segment whose start or end node   */
    /*   matches.                                                  */
    /* Returns segment index, or -1 if none found.                 */
    /* Also returns 'reversed' flag: 1 if segment must be walked   */
    /*   in reverse (undirected mode, matched on end node).        */

    /* Phase 1: Walk from degree-1 nodes (open chain starts) */
    for (int n_idx = 0; n_idx < node_count; n_idx++) {
        if (node_degree[n_idx] != 1) continue;

        /* Check if any unvisited segment is incident at this node */
        int first_seg = -1;
        int first_reversed = 0;
        for (int s = 0; s < n_segments; s++) {
            if (seg_visited[s]) continue;
            if (seg_start_node[s] < 0) continue;
            if (directed) {
                if (seg_start_node[s] == n_idx) {
                    first_seg = s; first_reversed = 0; break;
                }
            } else {
                if (seg_start_node[s] == n_idx) {
                    first_seg = s; first_reversed = 0; break;
                }
                if (seg_end_node[s] == n_idx) {
                    first_seg = s; first_reversed = 1; break;
                }
            }
        }
        if (first_seg < 0) continue;

        /* Walk the chain */
        int chain_coords = 0;
        int cur_seg = first_seg;
        int cur_reversed = first_reversed;

        while (cur_seg >= 0) {
            seg_visited[cur_seg] = 1;
            int len = seg_lengths[cur_seg];

            if (chain_coords == 0) {
                chain_coords += len;  /* first segment: all coords */
            } else {
                chain_coords += len - 1;  /* skip shared endpoint */
            }

            /* Find the "exit" node of this segment */
            int exit_node;
            if (cur_reversed) {
                exit_node = seg_start_node[cur_seg];
            } else {
                exit_node = seg_end_node[cur_seg];
            }

            /* Find next segment from exit_node */
            int next_seg = -1;
            int next_reversed = 0;
            for (int s = 0; s < n_segments; s++) {
                if (seg_visited[s]) continue;
                if (seg_start_node[s] < 0) continue;
                if (directed) {
                    if (seg_start_node[s] == exit_node) {
                        next_seg = s; next_reversed = 0; break;
                    }
                } else {
                    if (seg_start_node[s] == exit_node) {
                        next_seg = s; next_reversed = 0; break;
                    }
                    if (seg_end_node[s] == exit_node) {
                        next_seg = s; next_reversed = 1; break;
                    }
                }
            }
            cur_seg = next_seg;
            cur_reversed = next_reversed;
        }

        total_coords += chain_coords;
        total_parts++;
    }

    /* Phase 2: Handle remaining unvisited segments (rings / closed loops) */
    for (int s = 0; s < n_segments; s++) {
        if (seg_visited[s]) continue;
        if (seg_start_node[s] < 0) continue;

        /* Start a chain from this segment */
        int chain_coords = 0;
        int cur_seg = s;
        int cur_reversed = 0;

        while (cur_seg >= 0) {
            seg_visited[cur_seg] = 1;
            int len = seg_lengths[cur_seg];

            if (chain_coords == 0) {
                chain_coords += len;
            } else {
                chain_coords += len - 1;
            }

            int exit_node;
            if (cur_reversed) {
                exit_node = seg_start_node[cur_seg];
            } else {
                exit_node = seg_end_node[cur_seg];
            }

            int next_seg = -1;
            int next_reversed = 0;
            for (int ss = 0; ss < n_segments; ss++) {
                if (seg_visited[ss]) continue;
                if (seg_start_node[ss] < 0) continue;
                if (directed) {
                    if (seg_start_node[ss] == exit_node) {
                        next_seg = ss; next_reversed = 0; break;
                    }
                } else {
                    if (seg_start_node[ss] == exit_node) {
                        next_seg = ss; next_reversed = 0; break;
                    }
                    if (seg_end_node[ss] == exit_node) {
                        next_seg = ss; next_reversed = 1; break;
                    }
                }
            }
            cur_seg = next_seg;
            cur_reversed = next_reversed;
        }

        total_coords += chain_coords;
        total_parts++;
    }

    /* Also handle degenerate segments (< 2 coords) that were skipped */
    for (int s = 0; s < n_segments; s++) {
        if (seg_visited[s]) continue;
        if (seg_start_node[s] >= 0) continue;
        /* Degenerate segment: include as-is if it has coords */
        if (seg_lengths[s] > 0) {
            total_coords += seg_lengths[s];
            total_parts++;
        }
        seg_visited[s] = 1;
    }

    out_coord_counts[tid] = total_coords;
    out_part_counts[tid] = total_parts;
}


/* ------------------------------------------------------------------ */
/* Scatter kernel: write merged coordinates into output buffers       */
/* ------------------------------------------------------------------ */

extern "C" __global__ void __launch_bounds__(128, 4)
line_merge_scatter(
    /* MultiLineString family buffers */
    const double* __restrict__ mls_x,
    const double* __restrict__ mls_y,
    const int* __restrict__ mls_geom_off,
    const int* __restrict__ mls_part_off,
    const int mls_row_count,

    /* LineString family buffers */
    const double* __restrict__ ls_x,
    const double* __restrict__ ls_y,
    const int* __restrict__ ls_geom_off,
    const int ls_row_count,

    /* Row mapping */
    const int* __restrict__ global_rows,
    const int* __restrict__ family_codes,
    const int* __restrict__ fam_local_rows,
    const int directed,

    /* Output offsets (from prefix sum) */
    const int* __restrict__ coord_offsets,   /* per-thread coord write offset */
    const int* __restrict__ part_offsets,    /* per-thread part write offset */

    /* Output buffers */
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int* __restrict__ out_part_off,          /* part offset array for output MLS */

    const int n_rows
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rows) return;

    const int fam_code = family_codes[tid];
    const int fam_row = fam_local_rows[tid];

    int coord_pos = coord_offsets[tid];
    int part_pos = part_offsets[tid];

    /* Determine segments for this geometry */
    int n_segments = 0;
    int seg_starts[MAX_SEGMENTS];
    int seg_lengths[MAX_SEGMENTS];
    const double* src_x;
    const double* src_y;

    if (fam_code == 0) {
        /* MultiLineString */
        src_x = mls_x;
        src_y = mls_y;
        int part_begin = mls_geom_off[fam_row];
        int part_end = mls_geom_off[fam_row + 1];
        n_segments = part_end - part_begin;
        if (n_segments > MAX_SEGMENTS) {
            /* Overflow -- copy parts as-is */
            for (int p = part_begin; p < part_end; p++) {
                out_part_off[part_pos] = coord_pos;
                part_pos++;
                int cs = mls_part_off[p];
                int ce = mls_part_off[p + 1];
                for (int c = cs; c < ce; c++) {
                    out_x[coord_pos] = src_x[c];
                    out_y[coord_pos] = src_y[c];
                    coord_pos++;
                }
            }
            return;
        }
        for (int i = 0; i < n_segments; i++) {
            seg_starts[i] = mls_part_off[part_begin + i];
            seg_lengths[i] = mls_part_off[part_begin + i + 1] - seg_starts[i];
        }
    } else {
        /* LineString -- copy as-is */
        src_x = ls_x;
        src_y = ls_y;
        int coord_begin = ls_geom_off[fam_row];
        int coord_end = ls_geom_off[fam_row + 1];
        int n_coords = coord_end - coord_begin;
        if (n_coords > 0) {
            out_part_off[part_pos] = coord_pos;
            /* part_pos not incremented further here; final offset written by host */
            for (int c = coord_begin; c < coord_end; c++) {
                out_x[coord_pos] = src_x[c];
                out_y[coord_pos] = src_y[c];
                coord_pos++;
            }
        }
        return;
    }

    if (n_segments == 0) return;
    if (n_segments == 1) {
        out_part_off[part_pos] = coord_pos;
        for (int c = seg_starts[0]; c < seg_starts[0] + seg_lengths[0]; c++) {
            out_x[coord_pos] = src_x[c];
            out_y[coord_pos] = src_y[c];
            coord_pos++;
        }
        return;
    }

    /* ---------------------------------------------------------- */
    /* Build endpoint graph (identical to count kernel)            */
    /* ---------------------------------------------------------- */

    double node_x[MAX_SEGMENTS * 2];
    double node_y[MAX_SEGMENTS * 2];
    int node_count = 0;
    int seg_start_node[MAX_SEGMENTS];
    int seg_end_node[MAX_SEGMENTS];

    for (int s = 0; s < n_segments; s++) {
        if (seg_lengths[s] < 2) {
            seg_start_node[s] = -1;
            seg_end_node[s] = -1;
            continue;
        }
        double sx = src_x[seg_starts[s]];
        double sy = src_y[seg_starts[s]];
        double ex = src_x[seg_starts[s] + seg_lengths[s] - 1];
        double ey = src_y[seg_starts[s] + seg_lengths[s] - 1];

        seg_start_node[s] = find_or_insert_node(node_x, node_y, &node_count,
                                                  sx, sy, MAX_SEGMENTS * 2);
        seg_end_node[s] = find_or_insert_node(node_x, node_y, &node_count,
                                                ex, ey, MAX_SEGMENTS * 2);
    }

    int node_degree[MAX_SEGMENTS * 2];
    for (int i = 0; i < node_count; i++) node_degree[i] = 0;
    for (int s = 0; s < n_segments; s++) {
        if (seg_start_node[s] < 0) continue;
        if (directed) {
            node_degree[seg_end_node[s]]++;
            node_degree[seg_start_node[s]]++;
        } else {
            node_degree[seg_start_node[s]]++;
            node_degree[seg_end_node[s]]++;
        }
    }

    unsigned char seg_visited[MAX_SEGMENTS];
    for (int s = 0; s < n_segments; s++) seg_visited[s] = 0;

    /* Helper: write a segment's coordinates in forward or reverse order */
    /* skip_first: if true, skip the first coordinate (shared endpoint) */
    /* Returns: nothing, but updates coord_pos */

    /* Phase 1: Walk from degree-1 nodes */
    for (int n_idx = 0; n_idx < node_count; n_idx++) {
        if (node_degree[n_idx] != 1) continue;

        int first_seg = -1;
        int first_reversed = 0;
        for (int s = 0; s < n_segments; s++) {
            if (seg_visited[s]) continue;
            if (seg_start_node[s] < 0) continue;
            if (directed) {
                if (seg_start_node[s] == n_idx) {
                    first_seg = s; first_reversed = 0; break;
                }
            } else {
                if (seg_start_node[s] == n_idx) {
                    first_seg = s; first_reversed = 0; break;
                }
                if (seg_end_node[s] == n_idx) {
                    first_seg = s; first_reversed = 1; break;
                }
            }
        }
        if (first_seg < 0) continue;

        /* Record part start */
        out_part_off[part_pos] = coord_pos;
        part_pos++;

        int chain_started = 0;
        int cur_seg = first_seg;
        int cur_reversed = first_reversed;

        while (cur_seg >= 0) {
            seg_visited[cur_seg] = 1;
            int len = seg_lengths[cur_seg];
            int base = seg_starts[cur_seg];
            int skip = (chain_started) ? 1 : 0;

            if (cur_reversed) {
                for (int i = len - 1 - skip; i >= 0; i--) {
                    out_x[coord_pos] = src_x[base + i];
                    out_y[coord_pos] = src_y[base + i];
                    coord_pos++;
                }
            } else {
                for (int i = skip; i < len; i++) {
                    out_x[coord_pos] = src_x[base + i];
                    out_y[coord_pos] = src_y[base + i];
                    coord_pos++;
                }
            }
            chain_started = 1;

            int exit_node;
            if (cur_reversed) {
                exit_node = seg_start_node[cur_seg];
            } else {
                exit_node = seg_end_node[cur_seg];
            }

            int next_seg = -1;
            int next_reversed = 0;
            for (int s = 0; s < n_segments; s++) {
                if (seg_visited[s]) continue;
                if (seg_start_node[s] < 0) continue;
                if (directed) {
                    if (seg_start_node[s] == exit_node) {
                        next_seg = s; next_reversed = 0; break;
                    }
                } else {
                    if (seg_start_node[s] == exit_node) {
                        next_seg = s; next_reversed = 0; break;
                    }
                    if (seg_end_node[s] == exit_node) {
                        next_seg = s; next_reversed = 1; break;
                    }
                }
            }
            cur_seg = next_seg;
            cur_reversed = next_reversed;
        }
    }

    /* Phase 2: Rings (remaining unvisited segments) */
    for (int s = 0; s < n_segments; s++) {
        if (seg_visited[s]) continue;
        if (seg_start_node[s] < 0) continue;

        out_part_off[part_pos] = coord_pos;
        part_pos++;

        int chain_started = 0;
        int cur_seg = s;
        int cur_reversed = 0;

        while (cur_seg >= 0) {
            seg_visited[cur_seg] = 1;
            int len = seg_lengths[cur_seg];
            int base = seg_starts[cur_seg];
            int skip = (chain_started) ? 1 : 0;

            if (cur_reversed) {
                for (int i = len - 1 - skip; i >= 0; i--) {
                    out_x[coord_pos] = src_x[base + i];
                    out_y[coord_pos] = src_y[base + i];
                    coord_pos++;
                }
            } else {
                for (int i = skip; i < len; i++) {
                    out_x[coord_pos] = src_x[base + i];
                    out_y[coord_pos] = src_y[base + i];
                    coord_pos++;
                }
            }
            chain_started = 1;

            int exit_node;
            if (cur_reversed) {
                exit_node = seg_start_node[cur_seg];
            } else {
                exit_node = seg_end_node[cur_seg];
            }

            int next_seg = -1;
            int next_reversed = 0;
            for (int ss = 0; ss < n_segments; ss++) {
                if (seg_visited[ss]) continue;
                if (seg_start_node[ss] < 0) continue;
                if (directed) {
                    if (seg_start_node[ss] == exit_node) {
                        next_seg = ss; next_reversed = 0; break;
                    }
                } else {
                    if (seg_start_node[ss] == exit_node) {
                        next_seg = ss; next_reversed = 0; break;
                    }
                    if (seg_end_node[ss] == exit_node) {
                        next_seg = ss; next_reversed = 1; break;
                    }
                }
            }
            cur_seg = next_seg;
            cur_reversed = next_reversed;
        }
    }

    /* Phase 3: Degenerate segments */
    for (int s = 0; s < n_segments; s++) {
        if (seg_visited[s]) continue;
        if (seg_lengths[s] > 0) {
            out_part_off[part_pos] = coord_pos;
            part_pos++;
            for (int c = seg_starts[s]; c < seg_starts[s] + seg_lengths[s]; c++) {
                out_x[coord_pos] = src_x[c];
                out_y[coord_pos] = src_y[c];
                coord_pos++;
            }
        }
        seg_visited[s] = 1;
    }
}
"""


LINE_MERGE_KERNEL_NAMES: tuple[str, ...] = (
    "line_merge_count",
    "line_merge_scatter",
)


def _get_kernel_names() -> tuple[str, ...]:
    """Return the tuple of kernel entry point names."""
    return LINE_MERGE_KERNEL_NAMES
