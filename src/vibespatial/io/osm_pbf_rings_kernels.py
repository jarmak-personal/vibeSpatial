"""Endpoint-indexed ring chaining and OGR-compatible binary64 ring nesting."""
from __future__ import annotations

_PBF_RING_NAMES = ("pbf_endpoint_successors", "pbf_branch_rings", "pbf_emit_rings", "pbf_ring_bounds", "pbf_ring_parent", "pbf_reorder_rings")
_PBF_RING_SOURCE = r'''
// The sorted endpoint directory contains only open, active way members.
// Group offsets make ordinary successor lookup constant-time. Each branch
// cursor consumes its ordered endpoint list once, including visited entries.
extern "C" __global__ void __launch_bounds__(128)
pbf_endpoint_successors(const i64* __restrict__ groups,const i64* __restrict__ order,
                        const i64* __restrict__ inverse,const i64* __restrict__ members,
                        const double* __restrict__ x,const double* __restrict__ y,
                        const i64* __restrict__ offsets,i64* __restrict__ next,
                        int* __restrict__ branches,int n) {
    int state=blockIdx.x*blockDim.x+threadIdx.x; if(state>=2*n) return;
    int member=state/2; const i64* m=members+member*M;
    next[state]=-1;
    if(m[1]!=1 || m[2]==3 || m[5]<2) return;
    i64 begin=offsets[member],end=offsets[member+1]-1;
    if(x[begin]==x[end] && y[begin]==y[end]) { next[state]=state; return; }
    i64 group=inverse[state^1],lo=groups[group],count=groups[group+1]-lo;
    if(count==2) next[state]=order[lo+(order[lo]/2==member)];
    else if(count>2) atomicExch(branches+m[3],1);
}
extern "C" __global__ void __launch_bounds__(128)
pbf_branch_rings(const i64* __restrict__ groups,const i64* __restrict__ order,
                 const i64* __restrict__ inverse,const i64* __restrict__ members,
                 const i64* __restrict__ relation_offsets,const int* __restrict__ branches,
                 const double* __restrict__ x,const double* __restrict__ y,
                 const i64* __restrict__ offsets,i64* __restrict__ next,
                 int* __restrict__ visited,i64* __restrict__ cursors,int nr) {
    int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=nr || !branches[r]) return;
    i64 first=relation_offsets[r],stop=relation_offsets[r+1];
    for(i64 i=first;i<stop;++i) { next[2*i]=next[2*i+1]=-1; visited[i]=0; }
    for(i64 seed=first;seed<stop;++seed) {
        const i64* sm=members+seed*M;
        if(visited[seed] || sm[1]!=1 || sm[2]==3 || sm[5]<2) continue;
        i64 sb=offsets[seed],se=offsets[seed+1]-1;
        if(x[sb]==x[se] && y[sb]==y[se]) { next[2*seed]=2*seed; next[2*seed+1]=2*seed+1; visited[seed]=1; continue; }
        i64 state=2*seed,seed_group=inverse[2*seed];
        while(state>=0) {
            i64 member=state/2; visited[member]=1;
            i64 group=inverse[state^1];
            if(group==seed_group) {
                next[state]=2*seed; next[2*seed+1]=state^1; break;
            }
            i64 at=cursors[group],end=groups[group+1];
            while(at<end && visited[order[at]/2]) ++at;
            cursors[group]=at<end?at+1:at;
            if(at==end) break;
            i64 best=order[at];
            next[state]=best; next[best^1]=state^1; state=best;
        }
    }
}
extern "C" __global__ void __launch_bounds__(128)
pbf_emit_rings(const double* __restrict__ x,const double* __restrict__ y,
               const i64* __restrict__ offsets,const i64* __restrict__ states,
               const i64* __restrict__ ring_ids,const i64* __restrict__ prefixes,
               const i64* __restrict__ ring_offsets,double* __restrict__ ox,
               double* __restrict__ oy,int n) {
    int i=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31; if(i>=n) return;
    i64 state=states[i]; if(state<0) return;
    i64 member=state/2,begin=offsets[member],end=offsets[member+1],count=end-begin-1;
    i64 ring=ring_ids[i],dest=ring_offsets[ring]+prefixes[i]-count;
    for(i64 j=lane;j<count;j+=32) {
        i64 source=(state&1)?end-1-j:begin+j;
        ox[dest+j]=x[source]; oy[dest+j]=y[source];
    }
    if(lane==0 && prefixes[i]==count) { ox[ring_offsets[ring+1]-1]=x[begin]; oy[ring_offsets[ring+1]-1]=y[begin]; }
}
__device__ void add128(u64& lo,i64& hi,u64 b,i64 bh) {
    u64 old=lo; lo+=b; hi=(i64)((u64)hi+(u64)bh+(lo<old));
}
extern "C" __global__ void __launch_bounds__(128)
pbf_ring_bounds(const double* __restrict__ x,const double* __restrict__ y,
                const i64* __restrict__ offsets,i64* __restrict__ bounds,int n) {
    int r=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31; if(r>=n) return;
    i64 minx=0x7fffffffffffffffLL,miny=minx,maxx=-minx,maxy=-minx,hi=0; u64 lo=0;
    for(i64 i=offsets[r]+lane;i<offsets[r+1];i+=32) {
        i64 px=llrint(x[i]*1e7),py=llrint(y[i]*1e7);
        minx=min(minx,px); miny=min(miny,py); maxx=max(maxx,px); maxy=max(maxy,py);
        if(i+1<offsets[r+1]) {
            i64 qx=llrint(x[i+1]*1e7),qy=llrint(y[i+1]*1e7);
            i64 cross=px*qy-qx*py; add128(lo,hi,(u64)cross,cross<0?-1:0);
        }
    }
    for(int step=16;step;step>>=1) {
        minx=min(minx,__shfl_down_sync(0xffffffff,minx,step)); miny=min(miny,__shfl_down_sync(0xffffffff,miny,step));
        maxx=max(maxx,__shfl_down_sync(0xffffffff,maxx,step)); maxy=max(maxy,__shfl_down_sync(0xffffffff,maxy,step));
        u64 l=__shfl_down_sync(0xffffffff,lo,step); i64 h=__shfl_down_sync(0xffffffff,hi,step); add128(lo,hi,l,h);
    }
    if(lane==0) {
        if(hi<0) { lo=~lo+1; hi=(i64)(~(u64)hi+(lo==0)); }
        i64* b=bounds+r*6; b[0]=minx; b[1]=miny; b[2]=maxx; b[3]=maxy; b[4]=(i64)lo; b[5]=hi;
    }
}
// The topology codec follows OGR's binary64 segment tests. Explicit rounding
// prevents NVRTC contraction from changing a touching-ring classification.
__device__ int ring_query(const double* x,const double* y,i64 begin,i64 end,
                          double px,double py,int lane) {
    bool parity=false,boundary=false;
    for(i64 k=begin+lane;k<end-1;k+=32) {
        double ax=__dsub_rn(x[k],px),ay=__dsub_rn(y[k],py);
        double bx=__dsub_rn(x[k+1],px),by=__dsub_rn(y[k+1],py);
        double cross=__dsub_rn(__dmul_rn(bx,ay),__dmul_rn(ax,by));
        if(((by>0) && (ay<=0)) || ((ay>0) && (by<=0)))
            parity ^= __ddiv_rn(cross,__dsub_rn(ay,by))>0;
        if(cross==0 && !(ax==bx && ay==by)) {
            double dx=__dsub_rn(x[k+1],x[k]),dy=__dsub_rn(y[k+1],y[k]);
            double dot=__dadd_rn(__dmul_rn(-ax,dx),__dmul_rn(-ay,dy));
            double length=__dadd_rn(__dmul_rn(dx,dx),__dmul_rn(dy,dy));
            boundary |= dot>=0 && dot<=length;
        }
    }
    return __any_sync(0xffffffff,boundary)?-1:(__popc(__ballot_sync(0xffffffff,parity))&1);
}
__device__ bool area_less(const i64* a,const i64* b) {
    return a[5]<b[5] || (a[5]==b[5] && (u64)a[4]<(u64)b[4]);
}
extern "C" __global__ void __launch_bounds__(128)
pbf_ring_parent(const double* __restrict__ x,const double* __restrict__ y,
                const i64* __restrict__ offsets,const i64* __restrict__ rows,
                const i64* __restrict__ row_offsets,const i64* __restrict__ bounds,
                const i64* __restrict__ invalid_rows,i64* __restrict__ parent,int n) {
    int r=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31; if(r>=n) return;
    const i64* b=bounds+r*6; i64 best=-1,row=rows[r];
    if(invalid_rows[row]) { if(lane==0) parent[r]=-1; return; }
    for(i64 j=row_offsets[row];j<row_offsets[row+1];++j) {
        const i64* q=bounds+j*6;
        bool equal_area=b[4]==q[4] && b[5]==q[5];
        const i64* best_bounds=best>=0?bounds+best*6:nullptr;
        bool better=best<0 || area_less(q,best_bounds) || (q[4]==best_bounds[4] && q[5]==best_bounds[5] && j>best);
        if(j==r || !(area_less(b,q) || (equal_area && j<r)) || !better ||
           b[0]<q[0] || b[1]<q[1] || b[2]>q[2] || b[3]>q[3]) continue;
        int inside=-1;
        for(i64 candidate=offsets[r];candidate<offsets[r+1] && inside<0;++candidate)
            inside=ring_query(x,y,offsets[j],offsets[j+1],x[candidate],y[candidate],lane);
        // Boundary-only vertices need the same midpoint test as OGR.
        for(i64 candidate=offsets[r];candidate<offsets[r+1]-1 && inside<0;++candidate) {
            double px=__dmul_rn(__dadd_rn(x[candidate],x[candidate+1]),0.5);
            double py=__dmul_rn(__dadd_rn(y[candidate],y[candidate+1]),0.5);
            inside=ring_query(x,y,offsets[j],offsets[j+1],px,py,lane);
        }
        if(inside==1) best=j;
    }
    if(lane==0) parent[r]=best;
}
extern "C" __global__ void __launch_bounds__(128)
pbf_reorder_rings(const double* __restrict__ x,const double* __restrict__ y,
                  const i64* __restrict__ offsets,const i64* __restrict__ order,
                  const i64* __restrict__ new_offsets,double* __restrict__ ox,
                  double* __restrict__ oy,int n) {
    int r=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31; if(r>=n) return;
    i64 source=order[r],begin=offsets[source],end=offsets[source+1],dest=new_offsets[r];
    for(i64 i=begin+lane;i<end;i+=32) { ox[dest+i-begin]=x[i]; oy[dest+i-begin]=y[i]; }
}
'''
