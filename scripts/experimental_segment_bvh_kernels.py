"""Local cooperative segment refinement and per-row binary bounds trees."""

SOURCE = r"""
extern "C" __global__ void row_bvh_build(
    const int* owner, const int* nodes, const int* capacity,
    const int* offsets, const double* x0, const double* y0,
    const double* x1, const double* y1, float* bounds, int total, int stage) {
    int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= total) return;
    int row = owner[at], local = at - nodes[row], cap = capacity[row];
    if (stage == 0) {
        float lo_x = INFINITY, lo_y = INFINITY, hi_x = -INFINITY, hi_y = -INFINITY;
        int leaf = local - (cap - 1);
        if (leaf >= 0 && leaf < offsets[row+1] - offsets[row]) {
            int s = offsets[row] + leaf;
            lo_x = __double2float_rd(fmin(x0[s], x1[s]));
            lo_y = __double2float_rd(fmin(y0[s], y1[s]));
            hi_x = __double2float_ru(fmax(x0[s], x1[s]));
            hi_y = __double2float_ru(fmax(y0[s], y1[s]));
        }
        bounds[at] = lo_x; bounds[total+at] = lo_y;
        bounds[2*total+at] = hi_x; bounds[3*total+at] = hi_y;
    } else {
        int width = cap >> stage;
        if (!width || local < width-1 || local >= 2*width-1) return;
        int a = nodes[row] + 2*local+1, b = a+1;
        bounds[at] = fminf(bounds[a], bounds[b]);
        bounds[total+at] = fminf(bounds[total+a], bounds[total+b]);
        bounds[2*total+at] = fmaxf(bounds[2*total+a], bounds[2*total+b]);
        bounds[3*total+at] = fmaxf(bounds[3*total+a], bounds[3*total+b]);
    }
}

__device__ double segment_box_lower(const float* b, int at, int total,
                                    double ax, double ay, double bx, double by) {
    float lx = __double2float_rd(fmin(ax,bx)), hx = __double2float_ru(fmax(ax,bx));
    float ly = __double2float_rd(fmin(ay,by)), hy = __double2float_ru(fmax(ay,by));
    float dx = fmaxf(0.f, fmaxf(__fsub_rd(b[at],hx), __fsub_rd(lx,b[2*total+at])));
    float dy = fmaxf(0.f, fmaxf(__fsub_rd(b[total+at],hy), __fsub_rd(ly,b[3*total+at])));
    return (double)__fadd_rd(__fmul_rd(dx,dx),__fmul_rd(dy,dy));
}

extern "C" __global__ void cooperative_segment_refine(
    const int* left_idx, const int* right_idx, const bool* active,
    const signed char* ltags, const int* lfro, const int* lgo,
    const int* lpo, const int* lro, const double* lx, const double* ly,
    const signed char* rtags, const int* rfro, const int* rgo,
    const int* rpo, const int* rro, const double* rx, const double* ry,
    const double* left_bounds, const double* right_bounds,
    const int* ls, const double* lax, const double* lay, const double* lbx, const double* lby,
    const int* rs, const double* rax, const double* ray, const double* rbx, const double* rby,
    const int* rn, const int* rc, const float* bounds,
    double* out, unsigned long long* work, const long long* tile_offsets, int pairs, int total,
    int ltag, int lkind, int rtag, int rkind, int hierarchy, int phase, int tile_size, int work_count) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int item = thread >> 5, lane = thread & 31, pair = item;
    if (item >= work_count) return;
    if (phase == 1) {
        int lo=0,hi=pairs;
        if (lane == 0) {
            while (lo < hi) {
                int mid=lo+(hi-lo)/2;
                if (tile_offsets[mid+1] <= item) lo=mid+1; else hi=mid;
            }
        }
        pair=__shfl_sync(0xffffffff,lo,0);
    }
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
    if (phase != 1 && boxes_overlap && rkind >= 2) {
        int ranges = boundary_range_count(lkind,a,lgo,lpo);
        for (int i = lane; i < ranges; i += 32) {
            int cs,ce; boundary_coord_range(lkind,a,i,lgo,lpo,lro,&cs,&ce);
            if (ce > cs && point_in_polygonal_family(lx[cs],ly[cs],rkind,b,rgo,rpo,rro,rx,ry)) contained = true;
        }
    }
    if (phase != 1 && boxes_overlap && lkind >= 2) {
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
    unsigned long long initial_bits=0x7ff0000000000000ULL;
    if (phase == 1 && lane == 0) initial_bits=atomicCAS((unsigned long long*)(out+pair),0ULL,0ULL);
    initial_bits=__shfl_sync(0xffffffff,initial_bits,0);
    if (initial_bits == 0ULL) return;
    if (phase == 1) {
        lstart += 32+(item-tile_offsets[pair])*tile_size;
        lend = min(lend,lstart+tile_size);
    } else if (phase == 0) {
        lend = min(lend,lstart+32);
    }
    double best = __longlong_as_double(initial_bits);
    unsigned long long nodes_seen = 0, segments_tested = 0;
    for (int tile = lstart; tile < lend; tile += 32) {
        int s = tile + lane;
        if (s < lend) {
            double ax = lax[s], ay = lay[s], bx = lbx[s], by = lby[s];
            if (!hierarchy) {
                for (int t = rstart; t < rend; ++t) {
                    best = fmin(best,segment_segment_sq_dist(ax,ay,bx,by,rax[t],ray[t],rbx[t],rby[t]));
                    ++segments_tested;
                    if (best == 0.) break;
                }
            } else {
                int stack[32], sp=1; stack[0]=0;
                int root = rn[ri], first = rc[ri]-1;
                double scale = fabs(ax)+fabs(ay)+fabs(bx)+fabs(by)+1.;
                while (sp && best > 0.) {
                    int node = stack[--sp]; ++nodes_seen;
                    double limit = sqrt(best) + 64. * 2.2204460492503131e-16 * (scale+sqrt(best));
                    if (segment_box_lower(bounds,root+node,total,ax,ay,bx,by) > limit*limit) continue;
                    if (node >= first) {
                        int t = rstart+node-first;
                        if (t < rend) {
                            best=fmin(best,segment_segment_sq_dist(ax,ay,bx,by,rax[t],ray[t],rbx[t],rby[t]));
                            ++segments_tested;
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
        }
        for (int delta=16; delta; delta>>=1) best=fmin(best,__shfl_down_sync(0xffffffff,best,delta));
        best=__shfl_sync(0xffffffff,best,0);
        if (best == 0.) break;
    }
    for (int delta=16; delta; delta>>=1) {
        nodes_seen+=__shfl_down_sync(0xffffffff,nodes_seen,delta);
        segments_tested+=__shfl_down_sync(0xffffffff,segments_tested,delta);
    }
    if (lane == 0) {
        if (phase == 1) atomicMin((unsigned long long*)(out+pair),__double_as_longlong(best));
        else out[pair]=best;
        atomicAdd(work,nodes_seen); atomicAdd(work+1,segments_tested);
    }
}
"""
