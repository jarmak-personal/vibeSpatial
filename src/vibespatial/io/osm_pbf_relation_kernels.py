"""PBF relation/member byte codecs and indexed way expansion."""
from __future__ import annotations

_PBF_RELATION_NAMES = ("pbf_parse_relations", "pbf_decode_members", "pbf_match_member_ways", "pbf_expand_member_refs", "pbf_member_nodes")
_PBF_RELATION_SOURCE = r'''
constexpr int M=10;
extern "C" __global__ void __launch_bounds__(128)
pbf_parse_relations(const unsigned char* __restrict__ data,const i64* __restrict__ meta,
                    const i64* __restrict__ strings,i64* __restrict__ relations,
                    int* __restrict__ keep,int* __restrict__ error,int n,int layer) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    i64* r=relations+i*R; const i64* m=meta+r[2]*B;
    Cursor c{data,r[0],r[1],error}; bool hasid=false;
    while(c.p<c.end) {
        u64 t=c.tag(); int f=t>>3;
        if(t==8) { r[3]=(i64)c.var(); hasid=true; }
        else if((t&7)==2 && (f==2 || f==3 || f==8 || f==9 || f==10)) {
            i64 l=c.length(); int s=f==2?7:f==3?9:f==8?12:f==9?4:14;
            if(r[s+1]) atomicExch(error,2);
            r[s]=c.p; r[s+1]=c.p+l; c.p+=l;
        } else c.skip(t&7);
    }
    if(!hasid) atomicExch(error,1);
    r[6]=packed_count(data,r[4],r[5],error);
    if(packed_count(data,r[12],r[13],error)!=r[6] || packed_count(data,r[14],r[15],error)!=r[6]) atomicExch(error,1);
    Cursor k{data,r[7],r[8],error},v{data,r[9],r[10],error};
    int category=4; bool interesting=false; r[18]=-1;
    while(k.p<k.end && v.p<v.end) {
        i64 kp,kn,vp,vn; u64 key=k.var(),value=v.var();
        strref(m,strings,key,kp,kn,error); strref(m,strings,value,vp,vn,error);
        if(eq(data,kp,kn,"type")) {
            r[18]=value;
            if(eq(data,vp,vn,"multipolygon") || eq(data,vp,vn,"boundary")) category=2;
            else if(category!=2 && (eq(data,vp,vn,"multilinestring") || eq(data,vp,vn,"route"))) category=3;
        } else if(!eq(data,kp,kn,"created_by")) interesting=true;
    }
    if(k.p!=k.end || v.p!=v.end) atomicExch(error,1);
    r[16]=category; r[17]=!interesting; keep[i]=category==layer && r[6]>0; r[11]=keep[i];
}
extern "C" __global__ void __launch_bounds__(128)
pbf_decode_members(const unsigned char* __restrict__ data,const i64* __restrict__ meta,
                   const i64* __restrict__ strings,i64* __restrict__ relations,
                   const int* __restrict__ keep,const i64* __restrict__ offsets,
                   i64* __restrict__ members,int* __restrict__ error,int n,int layer) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n || !keep[i]) return;
    i64* r=relations+i*R; const i64* m=meta+r[2]*B;
    Cursor id{data,r[4],r[5],error},role{data,r[12],r[13],error},type{data,r[14],r[15],error};
    i64 absolute=0,j=offsets[i]; r[19]=j;
    while(id.p<id.end && role.p<role.end && type.p<type.end) {
        absolute=(i64)((u64)absolute+(u64)zz(id.var())); u64 t=type.var();
        i64 rp,rn; strref(m,strings,role.var(),rp,rn,error);
        if(t>2) atomicExch(error,1);
        int role_kind=eq(data,rp,rn,"outer")?1:eq(data,rp,rn,"inner")?2:eq(data,rp,rn,"subarea")?3:0;
        i64* out=members+j*M;
        out[0]=absolute; out[1]=t; out[2]=role_kind; out[3]=i; out[4]=j-offsets[i];
        out[5]=t==0 && layer==4?1:0; out[9]=-1;
        ++j;
    }
    if(j!=offsets[i+1] || id.p!=id.end || role.p!=role.end || type.p!=type.end) atomicExch(error,1);
}
// Work is ways x matching requested members. No global way/node dictionary.
extern "C" __global__ void __launch_bounds__(128)
pbf_match_member_ways(const i64* __restrict__ ways,const i64* __restrict__ sorted_ids,
                      const i64* __restrict__ order,i64* __restrict__ members,
                      int* __restrict__ error,int n,int nm) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    const i64* w=ways+i*W;
    int lo=0,hi=nm;
    while(lo<hi) { int mid=lo+((hi-lo)>>1); if(sorted_ids[mid]<w[3]) lo=mid+1; else hi=mid; }
    for(int j=lo;j<nm && sorted_ids[j]==w[3];++j) {
        i64* m=members+order[j]*M; if(m[1]!=1 || m[2]==3) continue;
        // Count+1 also claims empty ways. Exactly one source record may
        // provide a member's count and payload, even within the same warp.
        if(atomicCAS((u64*)(m+5),0ULL,(u64)w[6]+1ULL)) { atomicExch(error,4); continue; }
        m[6]=w[12]; m[7]=w[13]; m[8]=w[14]; m[9]=w[15];
    }
}
extern "C" __global__ void __launch_bounds__(128)
pbf_expand_member_refs(const unsigned char* __restrict__ data,const i64* __restrict__ ways,
                       const i64* __restrict__ sorted_ids,const i64* __restrict__ order,
                       const i64* __restrict__ members,const i64* __restrict__ offsets,
                       i64* __restrict__ x,int n,int nm) {
    int i=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31; if(i>=n) return;
    const i64* w=ways+i*W; if(w[6]<2) return;
    int lo=0,hi=nm;
    while(lo<hi) { int mid=lo+((hi-lo)>>1); if(sorted_ids[mid]<w[3]) lo=mid+1; else hi=mid; }
    for(int j=lo;j<nm && sorted_ids[j]==w[3];++j) {
        i64 row=order[j]; const i64* m=members+row*M;
        if(m[1]!=1 || m[2]==3 || !m[5]) continue;
        i64 sum=0,out=0,start=w[4],end=w[5],dest=offsets[row];
        for(i64 base=start;base<end;base+=32) {
            i64 p=base+lane; bool term=p<end && !(data[p]&128); i64 val=0;
            if(term) {
                i64 begin=p; while(begin>start && (data[begin-1]&128)) --begin;
                u64 raw=0; int shift=0;
                for(i64 q=begin;q<=p;++q,shift+=7) raw|=(u64)(data[q]&127)<<shift;
                val=zz(raw);
            }
            unsigned mask=__ballot_sync(0xffffffff,term); int rank=__popc(mask&((1u<<lane)-1));
            for(int step=1;step<32;step<<=1) { i64 v=__shfl_up_sync(0xffffffff,val,step); if(lane>=step) val=(i64)((u64)val+(u64)v); }
            if(term) x[dest+out+rank]=(i64)((u64)sum+(u64)val);
            sum=(i64)((u64)sum+(u64)__shfl_sync(0xffffffff,val,31)); out+=__popc(mask);
        }
    }
}
extern "C" __global__ void __launch_bounds__(128)
pbf_member_nodes(const i64* __restrict__ members,const i64* __restrict__ offsets,
                 i64* __restrict__ x,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    if(members[i*M+1]==0 && members[i*M+5]) x[offsets[i]]=members[i*M];
}
'''
_PBF_RELATION_NAMES += ("pbf_select_indices",)
_PBF_RELATION_SOURCE += r'''
extern "C" __global__ void __launch_bounds__(128)
pbf_select_indices(const int* __restrict__ mask,const i64* __restrict__ positions,
                   i64* __restrict__ out,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n && mask[i]) out[positions[i]]=i;
}
'''
_PBF_RELATION_NAMES += ("pbf_way_ranges",)
_PBF_RELATION_SOURCE += r'''
extern "C" __global__ void __launch_bounds__(128)
pbf_way_ranges(const i64* __restrict__ meta,const i64* __restrict__ ways,
               i64* __restrict__ ranges,int n) {
    int b=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31; if(b>=n) return;
    const i64* m=meta+b*B; i64 lo=0x7fffffffffffffffLL,hi=-lo;
    for(i64 i=m[10]+lane;i<m[10]+m[6];i+=32) { lo=min(lo,ways[i*W+3]); hi=max(hi,ways[i*W+3]); }
    for(int step=16;step;step>>=1) { lo=min(lo,__shfl_down_sync(0xffffffff,lo,step)); hi=max(hi,__shfl_down_sync(0xffffffff,hi,step)); }
    if(lane==0) { ranges[b*2]=lo; ranges[b*2+1]=hi; }
}
'''
_PBF_RELATION_NAMES += ("pbf_select_area_ways",)
_PBF_RELATION_SOURCE += r'''
extern "C" __global__ void __launch_bounds__(128)
pbf_select_area_ways(const i64* __restrict__ ways,int* __restrict__ keep,
                     const i64* __restrict__ excluded,int n,int ne) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    const i64* w=ways+i*W; int lo=0,hi=ne;
    while(lo<hi) { int mid=lo+((hi-lo)>>1); if(excluded[mid]<w[3]) lo=mid+1; else hi=mid; }
    keep[i]=w[6]>=2 && w[14] && w[15] && (lo==ne || excluded[lo]!=w[3]);
}
'''
_PBF_RELATION_NAMES += ("pbf_compact_member_coords",)
_PBF_RELATION_SOURCE += r'''
extern "C" __global__ void __launch_bounds__(128)
pbf_compact_member_coords(const double* __restrict__ x,const double* __restrict__ y,
                          const i64* __restrict__ offsets,i64* __restrict__ members,
                          const i64* __restrict__ new_offsets,double* __restrict__ ox,
                          double* __restrict__ oy,int n,int emit) {
    int i=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31; if(i>=n) return;
    i64* m=members+i*M; i64 begin=offsets[i],end=offsets[i+1];
    bool area=m[1]==1 && m[8]; if(area && end>begin) --end;
    i64 count=0,first=-1;
    if(emit && !m[5]) return;
    for(i64 base=begin;base<end;base+=32) {
        i64 pos=base+lane; bool valid=pos<end && !isnan(x[pos]);
        unsigned mask=__ballot_sync(0xffffffff,valid); int rank=__popc(mask&((1u<<lane)-1));
        if(first<0 && mask) first=base+__ffs(mask)-1;
        if(emit && valid) { i64 out=new_offsets[i]+count+rank; ox[out]=x[pos]; oy[out]=y[pos]; }
        count+=__popc(mask);
    }
    if(lane==0) {
        if(!emit) {
            if(area && count) ++count;
            m[5]=(m[1]==0 || count>=2)?count:0;
        } else if(area && first>=0) {
            ox[new_offsets[i]+count]=x[first]; oy[new_offsets[i]+count]=y[first];
        }
    }
}
'''
