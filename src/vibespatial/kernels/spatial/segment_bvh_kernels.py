"""Cooperative segment refinement with conservative per-row envelope hierarchies."""

SOURCE = r"""
extern "C" __global__ __launch_bounds__(256) void row_bvh_build(
    const int* __restrict__ owner, const int* __restrict__ nodes, const int* __restrict__ capacity,
    const int* __restrict__ offsets, const double* __restrict__ x0, const double* __restrict__ y0,
    const double* __restrict__ x1, const double* __restrict__ y1, INDEX_T* bounds, int total, int stage) {
    int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= total) return;
    int row = owner[at], local = at - nodes[row], cap = capacity[row];
    if (stage == 0) {
        INDEX_T lo_x = INFINITY, lo_y = INFINITY, hi_x = -INFINITY, hi_y = -INFINITY;
        int leaf = local - (cap - 1);
        if (leaf >= 0 && leaf < offsets[row+1] - offsets[row]) {
            int s = offsets[row] + leaf;
            lo_x = LOWER(fmin(x0[s], x1[s]));
            lo_y = LOWER(fmin(y0[s], y1[s]));
            hi_x = UPPER(fmax(x0[s], x1[s]));
            hi_y = UPPER(fmax(y0[s], y1[s]));
        }
        bounds[at] = lo_x; bounds[total+at] = lo_y;
        bounds[2*total+at] = hi_x; bounds[3*total+at] = hi_y;
    } else {
        int width = cap >> stage;
        if (!width || local < width-1 || local >= 2*width-1) return;
        int a = nodes[row] + 2*local+1, b = a+1;
        bounds[at] = fmin(bounds[a], bounds[b]);
        bounds[total+at] = fmin(bounds[total+a], bounds[total+b]);
        bounds[2*total+at] = fmax(bounds[2*total+a], bounds[2*total+b]);
        bounds[3*total+at] = fmax(bounds[3*total+a], bounds[3*total+b]);
    }
}

__device__ double segment_box_lower(const INDEX_T* __restrict__ b, int at, int total,
                                    double ax, double ay, double bx, double by) {
#if FLOAT_BOUNDS
    float lx = __double2float_rd(fmin(ax,bx)), hx = __double2float_ru(fmax(ax,bx));
    float ly = __double2float_rd(fmin(ay,by)), hy = __double2float_ru(fmax(ay,by));
    float dx = fmaxf(0.f, fmaxf(__fsub_rd(b[at],hx), __fsub_rd(lx,b[2*total+at])));
    float dy = fmaxf(0.f, fmaxf(__fsub_rd(b[total+at],hy), __fsub_rd(ly,b[3*total+at])));
    return (double)__fadd_rd(__fmul_rd(dx,dx),__fmul_rd(dy,dy));
#else
    double lx = fmin(ax,bx), hx = fmax(ax,bx);
    double ly = fmin(ay,by), hy = fmax(ay,by);
    double dx = fmax(0., fmax(__dsub_rd(b[at],hx), __dsub_rd(lx,b[2*total+at])));
    double dy = fmax(0., fmax(__dsub_rd(b[total+at],hy), __dsub_rd(ly,b[3*total+at])));
    return __dadd_rd(__dmul_rd(dx,dx),__dmul_rd(dy,dy));
#endif
}

extern "C" __global__ __launch_bounds__(256) void cooperative_segment_refine(
    const int* __restrict__ left_idx, const int* __restrict__ right_idx, const bool* __restrict__ active,
    const signed char* __restrict__ ltags, const int* __restrict__ lfro, const int* __restrict__ lgo,
    const int* __restrict__ lpo, const int* __restrict__ lro, const double* __restrict__ lx, const double* __restrict__ ly,
    const signed char* __restrict__ rtags, const int* __restrict__ rfro, const int* __restrict__ rgo,
    const int* __restrict__ rpo, const int* __restrict__ rro, const double* __restrict__ rx, const double* __restrict__ ry,
    const double* __restrict__ left_bounds, const double* __restrict__ right_bounds,
    const int* __restrict__ ls, const double* __restrict__ lax, const double* __restrict__ lay, const double* __restrict__ lbx, const double* __restrict__ lby,
    const int* __restrict__ rs, const double* __restrict__ rax, const double* __restrict__ ray, const double* __restrict__ rbx, const double* __restrict__ rby,
    const int* __restrict__ rn, const int* __restrict__ rc, const INDEX_T* __restrict__ bounds,
    double* out, int pairs, int total, int ltag, int lkind, int rtag, int rkind) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int pair = thread >> 5, lane = thread & 31;
    if (pair >= pairs || !active[pair]) return;
    int li = left_idx[pair], ri = right_idx[pair];
    if (ltags[li] != ltag || rtags[ri] != rtag) return;
    int a = lfro[li], b = rfro[ri];
    int lstart = ls[li], lend = ls[li+1], rstart = rs[ri], rend = rs[ri+1];
    if (a < 0 || b < 0 || lstart == lend || rstart == rend) return;
    bool boxes_overlap = left_bounds[4*li] <= right_bounds[4*ri+2] &&
        right_bounds[4*ri] <= left_bounds[4*li+2] &&
        left_bounds[4*li+1] <= right_bounds[4*ri+3] &&
        right_bounds[4*ri+1] <= left_bounds[4*li+3];
    bool contained = false;
    if (boxes_overlap && rkind >= 2) {
        int ranges = boundary_range_count(lkind,a,lgo,lpo);
        for (int i = lane; i < ranges; i += 32) {
            int cs,ce; boundary_coord_range(lkind,a,i,lgo,lpo,lro,&cs,&ce);
            if (ce > cs && point_in_polygonal_family(lx[cs],ly[cs],rkind,b,rgo,rpo,rro,rx,ry)) contained = true;
        }
    }
    if (boxes_overlap && lkind >= 2) {
        int ranges = boundary_range_count(rkind,b,rgo,rpo);
        for (int i = lane; i < ranges; i += 32) {
            int cs,ce; boundary_coord_range(rkind,b,i,rgo,rpo,rro,&cs,&ce);
            if (ce > cs && point_in_polygonal_family(rx[cs],ry[cs],lkind,a,lgo,lpo,lro,lx,ly)) contained = true;
        }
    }
    if (__any_sync(0xffffffff, contained)) {
        if (lane == 0) out[pair] = 0.;
        return;
    }
    double best = INFINITY;
    for (int tile = lstart; tile < lend; tile += 32) {
        int s = tile + lane;
        if (s < lend) {
            double ax = lax[s], ay = lay[s], bx = lbx[s], by = lby[s];
                int stack[32], sp=1; stack[0]=0;
                int root = rn[ri], first = rc[ri]-1;
                double scale = fabs(ax)+fabs(ay)+fabs(bx)+fabs(by)+1.;
                while (sp && best > 0.) {
                    int node = stack[--sp];
                    double limit = sqrt(best) + 64. * 2.2204460492503131e-16 * (scale+sqrt(best));
                    if (segment_box_lower(bounds,root+node,total,ax,ay,bx,by) > __dmul_ru(limit,limit)) continue;
                    if (node >= first) {
                        int t = rstart+node-first;
                        if (t < rend) {
                            best=fmin(best,segment_segment_sq_dist(ax,ay,bx,by,rax[t],ray[t],rbx[t],rby[t]));
                        }
                    } else {
                        int c=2*node+1;
                        double d0=segment_box_lower(bounds,root+c,total,ax,ay,bx,by);
                        double d1=segment_box_lower(bounds,root+c+1,total,ax,ay,bx,by);
                        if (d0 < d1) { stack[sp++]=c+1; stack[sp++]=c; }
                        else { stack[sp++]=c; stack[sp++]=c+1; }
                    }
                }
        }
        for (int delta=16; delta; delta>>=1) best=fmin(best,__shfl_down_sync(0xffffffff,best,delta));
        best=__shfl_sync(0xffffffff,best,0);
        if (best == 0.) break;
    }
    if (lane == 0) out[pair] = best;
}
"""
