"""Bounded PBF byte codecs. Coordinates are decoded to fp64, never downcast."""

from __future__ import annotations

from vibespatial.io.osm_pbf_attributes_kernels import _PBF_ATTRIBUTE_SOURCE
from vibespatial.io.osm_pbf_relation_kernels import _PBF_RELATION_NAMES, _PBF_RELATION_SOURCE
from vibespatial.io.osm_pbf_rings_kernels import _PBF_RING_NAMES, _PBF_RING_SOURCE

_PBF_SOURCE = r'''
using i64 = long long;
using u64 = unsigned long long;
struct Cursor {
    const unsigned char* data; i64 p, end; int* error;
    __device__ u64 var() {
        u64 v=0;
        for (int s=0; s<70; s+=7) {
            if(p>=end) { atomicExch(error,1); p=end; return 0; }
            unsigned int b=data[p++];
            if(s==63 && b>1) { atomicExch(error,1); p=end; return 0; }
            v |= (u64)(b&127)<<s;
            if(!(b&128)) return v;
        }
        atomicExch(error,1); p=end; return 0;
    }
    __device__ u64 tag() {
        u64 value=var();
        if(!(value>>3) || (value>>3)>0x1fffffffULL) { atomicExch(error,1); p=end; return 0; }
        return value;
    }
    __device__ i64 length() {
        u64 n=var();
        if(n>(u64)(end-p)) { atomicExch(error,1); p=end; return 0; }
        return (i64)n;
    }
    __device__ void skip(int wire) {
        if(wire==0) { var(); return; }
        i64 n=wire==2 ? length() : wire==1 ? 8 : wire==5 ? 4 : -1;
        if(n<0 || n>end-p) { atomicExch(error,1); p=end; }
        else p+=n;
    }
};
__device__ i64 zz(u64 n) { return (i64)((n>>1)^(-(n&1))); }
// Block: st begin/end, granularity, lat/lon offset, strings, ways, dense,
// nodes, string base, way base, dense base, node base.
constexpr int B=15, D=14, W=16, R=20;
__device__ i64 packed_count(const unsigned char* d,i64 p,i64 e,int* error) {
    i64 n=0; bool nonempty=p<e;
    while(p<e && (p&7)) n+=!(d[p++]&128);
    while(e-p>=8) { n+=8-__popcll(*(const u64*)(d+p)&0x8080808080808080ULL); p+=8; }
    while(p<e) n+=!(d[p++]&128);
    if(nonempty && (d[e-1]&128)) atomicExch(error,1);
    return n;
}
extern "C" __global__ void __launch_bounds__(128)
pbf_count_blocks(const unsigned char* __restrict__ data,
                 const i64* __restrict__ offsets, i64* __restrict__ meta,
                 int* __restrict__ error, int n) {
    int b=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    if(b>=n || (threadIdx.x&31)) return;
    i64* m=meta+b*B; m[2]=100;
    Cursor c{data,offsets[b],offsets[b+1],error};
    while(c.p<c.end) {
        u64 tag=c.tag(); int f=tag>>3,w=tag&7;
        if(!f || tag>>32) { atomicExch(error,1); return; }
        if(w==2 && (f==1 || f==2)) {
            i64 len=c.length(),end=c.p+len;
            if(f==1) {
                if(m[1]) atomicExch(error,2);
                m[0]=c.p; m[1]=end; Cursor st{data,c.p,end,error};
                while(st.p<st.end) {
                    u64 t=st.tag();
                    if(t==10) { i64 l=st.length(); st.p+=l; ++m[5]; }
                    else st.skip(t&7);
                }
            } else {
                Cursor g{data,c.p,end,error};
                while(g.p<g.end) {
                    u64 t=g.tag(); int kind=t>>3;
                    if((t&7)==2 && (kind==1 || kind==2 || kind==3 || kind==4)) {
                        i64 l=g.length(),ge=g.p+l;
                        if(kind==1) atomicExch(error,2); // ordinary Nodes: explicit decline
                        if(kind==3) ++m[6];
                        if(kind==4) ++m[13];
                        if(kind==2) {
                            ++m[7]; Cursor dn{data,g.p,ge,error};
                            while(dn.p<dn.end) {
                                u64 dt=dn.tag();
                                if(dt==10) { i64 dl=dn.length(); m[8]+=packed_count(data,dn.p,dn.p+dl,error); dn.p+=dl; }
                                else dn.skip(dt&7);
                            }
                        }
                        g.p=ge;
                    } else g.skip(t&7);
                }
            }
            c.p=end;
        } else if(w==0 && (f==17 || f==19 || f==20)) {
            // PrimitiveBlock offsets are int64, NOT ZigZag sint64.
            i64 v=(i64)c.var(); m[f==17?2:f==19?3:4]=v;
        } else c.skip(w);
    }
    if(m[2]<=0 || m[2]>2147483647 || !m[5]) atomicExch(error,1);
}
extern "C" __global__ void __launch_bounds__(128)
pbf_index_blocks(const unsigned char* __restrict__ data,
                 const i64* __restrict__ offsets,const i64* __restrict__ meta,
                 i64* __restrict__ strings,i64* __restrict__ ways,
                 i64* __restrict__ dense,i64* __restrict__ relations,int* __restrict__ error,int n) {
    int b=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    if(b>=n || (threadIdx.x&31)) return;
    const i64* m=meta+b*B;
    Cursor st{data,m[0],m[1],error}; i64 si=m[9];
    while(st.p<st.end) {
        u64 t=st.tag();
        if(t==10) { i64 l=st.length(); strings[si*2]=st.p; strings[si*2+1]=l; if(si==m[9] && l) atomicExch(error,1); ++si; st.p+=l; }
        else st.skip(t&7);
    }
    Cursor c{data,offsets[b],offsets[b+1],error}; i64 wi=m[10],di=m[11],ni=m[12],ri=m[14];
    while(c.p<c.end) {
        u64 tag=c.tag();
        if(tag==18) {
            i64 len=c.length(); Cursor g{data,c.p,c.p+len,error}; c.p+=len;
            while(g.p<g.end) {
                u64 t=g.tag();
                if(t==18 || t==26 || t==34) {
                    i64 l=g.length(),end=g.p+l;
                    if(t==34) {
                        if(relations) { i64* r=relations+(ri++)*R; r[0]=g.p; r[1]=end; r[2]=b; }
                    } else if(t==26) {
                        i64* r=ways+(wi++)*W; r[0]=g.p; r[1]=end; r[2]=b;
                    } else {
                        i64* r=dense+(di++)*D; r[0]=g.p; r[1]=end; r[2]=b; r[3]=ni;
                        Cursor d{data,g.p,end,error};
                        while(d.p<d.end) {
                            u64 dt=d.tag(); int f=dt>>3;
                            if((dt&7)==2 && (f==1 || f==8 || f==9 || f==10)) {
                                i64 dl=d.length(); int slot=f==1?5:f==8?7:f==9?9:11;
                                if(r[slot+1]) atomicExch(error,2); // split packed field
                                r[slot]=d.p; r[slot+1]=d.p+dl; d.p+=dl;
                            } else d.skip(dt&7);
                        }
                        r[4]=packed_count(data,r[5],r[6],error); ni+=r[4];
                        if(!r[4]) atomicExch(error,2);
                        if(packed_count(data,r[7],r[8],error)!=r[4] || packed_count(data,r[9],r[10],error)!=r[4]) atomicExch(error,1);
                    }
                    g.p=end;
                } else g.skip(t&7);
            }
        } else c.skip(tag&7);
    }
}
__device__ bool eq(const unsigned char* d,i64 p,i64 n,const char* s) {
    for(i64 k=0;k<n;++k) if(!s[k] || d[p+k]!=(unsigned char)s[k]) return false;
    return s[n]==0;
}
__device__ bool starts(const unsigned char* d,i64 p,i64 n,const char* s) {
    for(i64 k=0;s[k];++k) if(k>=n || d[p+k]!=(unsigned char)s[k]) return false;
    return true;
}
__device__ bool ignored(const unsigned char* d,i64 p,i64 n) {
    return eq(d,p,n,"created_by") || eq(d,p,n,"converted_by") || eq(d,p,n,"source") ||
        eq(d,p,n,"time") || eq(d,p,n,"ele") || eq(d,p,n,"note") || eq(d,p,n,"todo") ||
        eq(d,p,n,"fixme") || eq(d,p,n,"FIXME") || starts(d,p,n,"openGeoDB:");
}
__device__ bool early_ignored(const unsigned char* d,i64 p,i64 n) {
    return eq(d,p,n,"area") || eq(d,p,n,"created_by") || eq(d,p,n,"converted_by") ||
        eq(d,p,n,"note") || eq(d,p,n,"todo") || eq(d,p,n,"fixme") || eq(d,p,n,"FIXME");
}
__device__ bool insignificant_point(const unsigned char* d,i64 p,i64 n) {
    return eq(d,p,n,"created_by") || eq(d,p,n,"converted_by") || eq(d,p,n,"source") ||
        eq(d,p,n,"time") || eq(d,p,n,"ele") || eq(d,p,n,"attribution");
}
__device__ bool polygon_key(const unsigned char* d,i64 p,i64 n) {
    return eq(d,p,n,"aeroway") || eq(d,p,n,"amenity") || eq(d,p,n,"boundary") ||
        eq(d,p,n,"building") || eq(d,p,n,"craft") || eq(d,p,n,"geological") ||
        eq(d,p,n,"historic") || eq(d,p,n,"landuse") || eq(d,p,n,"leisure") ||
        eq(d,p,n,"military") || eq(d,p,n,"natural") || eq(d,p,n,"office") ||
        eq(d,p,n,"place") || eq(d,p,n,"shop") || eq(d,p,n,"sport") || eq(d,p,n,"tourism");
}
__device__ bool strref(const i64* m,const i64* strings,u64 sid,i64& p,i64& n,int* error) {
    if(sid>=(u64)m[5]) { atomicExch(error,1); p=n=0; return false; }
    p=strings[(m[9]+sid)*2]; n=strings[(m[9]+sid)*2+1]; return true;
}
extern "C" __global__ void __launch_bounds__(128)
pbf_parse_ways(const unsigned char* __restrict__ data,const i64* __restrict__ meta,
               const i64* __restrict__ strings,i64* __restrict__ ways,
               int* __restrict__ keep,int* __restrict__ error,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    i64* r=ways+i*W; const i64* m=meta+r[2]*B; Cursor c{data,r[0],r[1],error}; bool hasid=false;
    while(c.p<c.end) {
        u64 t=c.tag(); int f=t>>3;
        if(t==8) { r[3]=(i64)c.var(); hasid=true; }
        else if((t&7)==2 && (f==2 || f==3 || f==8)) {
            i64 l=c.length(); int s=f==2?7:f==3?9:4;
            if(r[s+1]) atomicExch(error,2);
            r[s]=c.p; r[s+1]=c.p+l; c.p+=l;
        } else c.skip(t&7);
    }
    if(!hasid) atomicExch(error,1);
    Cursor refs{data,r[4],r[5],error}; i64 first=0,last=0,count=0;
    while(refs.p<refs.end) { last=(i64)((u64)last+(u64)zz(refs.var())); if(count++==0) first=last; }
    r[6]=count;
    Cursor k{data,r[7],r[8],error},v{data,r[9],r[10],error}; bool significant=false,area=false,poly_significant=false; int explicit_area=0;
    while(k.p<k.end && v.p<v.end) {
        i64 kp,kn,vp,vn; strref(m,strings,k.var(),kp,kn,error); strref(m,strings,v.var(),vp,vn,error);
        significant=true; poly_significant |= !early_ignored(data,kp,kn);
        if(polygon_key(data,kp,kn) || ((eq(data,kp,kn,"highway") || eq(data,kp,kn,"public_transport")) && eq(data,vp,vn,"platform"))) area=true;
        if(!explicit_area && eq(data,kp,kn,"area")) {
            if(eq(data,vp,vn,"yes")) explicit_area=1;
            if(eq(data,vp,vn,"no")) explicit_area=-1;
        }
    }
    if(k.p!=k.end || v.p!=v.end) atomicExch(error,1);
    if(explicit_area) area=explicit_area==1;
    keep[i]=count>=2 && significant && !(first==last && area);
    r[11]=keep[i]; r[12]=first; r[13]=last; r[14]=first==last && area;
    r[15]=poly_significant;
}
// Every zero varint resets the keys/values parser to expecting a key.
// A zero is a node delimiter exactly when an even number of nonzero values
// precede it since the previous zero (an odd number makes it an empty value).
__device__ bool delimiter(const unsigned char* data,i64 p,i64 begin) {
    if(data[p]!=0) return false;
    i64 count=0;
    for(i64 j=p-1;j>=begin && data[j]!=0;--j) count+=!(data[j]&128);
    return !(count&1);
}
extern "C" __global__ void __launch_bounds__(128)
pbf_dense_tags(const unsigned char* __restrict__ data,const i64* __restrict__ meta,
               const i64* __restrict__ strings,const i64* __restrict__ dense,
               i64* __restrict__ spans,int* __restrict__ keep,int* __restrict__ error,int n) {
    int i=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31; if(i>=n) return;
    const i64* r=dense+i*D; const i64* m=meta+r[2]*B;
    if(r[11]==r[12]) {
        for(i64 j=lane;j<r[4];j+=32) {
            i64 row=r[3]+j; spans[row*3]=spans[row*3+1]=0; spans[row*3+2]=r[2]; keep[row]=0;
        }
        return;
    }
    i64 nodes=0;
    for(i64 base=r[11];base<r[12];base+=32) {
        i64 pos=base+lane; bool end=pos<r[12] && delimiter(data,pos,r[11]);
        unsigned mask=__ballot_sync(0xffffffff,end);
        int rank=__popc(mask & ((1u<<lane)-1));
        if(end) {
            i64 start=pos;
            while(start>r[11] && !delimiter(data,start-1,r[11])) --start;
            Cursor c{data,start,pos,error}; bool significant=false;
            while(c.p<c.end) {
                i64 kp,kn,vp,vn; u64 key=c.var();
                if(!key) atomicExch(error,1);
                strref(m,strings,key,kp,kn,error); strref(m,strings,c.var(),vp,vn,error);
                significant |= !insignificant_point(data,kp,kn);
            }
            if(nodes+rank>=r[4]) atomicExch(error,1);
            else {
                i64 row=r[3]+nodes+rank;
                spans[row*3]=start; spans[row*3+1]=pos; spans[row*3+2]=r[2]; keep[row]=significant;
            }
        }
        nodes+=__popc(mask);
    }
    if(lane==0 && (nodes!=r[4] || !delimiter(data,r[12]-1,r[11]))) atomicExch(error,1);
}
// Warp-segmented byte scan: terminal bytes decode independently, then a warp
// prefix restores delta values. No CPU varint position arrays or per-field launches.
extern "C" __global__ void __launch_bounds__(128)
pbf_decode_dense(const unsigned char* __restrict__ data,const i64* __restrict__ meta,
                 const i64* __restrict__ dense,i64* __restrict__ ids,
                 double* __restrict__ x,double* __restrict__ y,
                 const int* __restrict__ keep,const i64* __restrict__ positions,
                 i64* __restrict__ ranges,int* __restrict__ error,int n,int selected) {
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31;
    int record=warp/3,field=warp%3; if(record>=n || (field>0 && !x)) return;
    const i64* r=dense+record*D; const i64* m=meta+r[2]*B;
    i64 start=r[5+2*field],end=r[6+2*field],sum=0,out=0;
    for(i64 base=start;base<end;base+=32) {
        i64 pos=base+lane; bool term=pos<end && !(data[pos]&128); i64 val=0;
        if(term) {
            i64 p=pos; while(p>start && (data[p-1]&128) && pos-p<10) --p;
            u64 raw=0; int shift=0;
            if(pos-p>9 || (pos-p==9 && data[pos]>1) || (p>start && (data[p-1]&128))) atomicExch(error,1);
            else for(i64 q=p;q<=pos;++q,shift+=7) raw|=(u64)(data[q]&127)<<shift;
            val=zz(raw);

        }
        unsigned mask=__ballot_sync(0xffffffff,term); int rank=__popc(mask & ((1u<<lane)-1));
        if(term && field==0 && ranges && out+rank>0 && val<=0) atomicExch(error,2);
        i64 pref=val;
        for(int step=1;step<32;step<<=1) { i64 v=__shfl_up_sync(0xffffffff,pref,step); if(lane>=step) pref=(i64)((u64)pref+(u64)v); }
        i64 absolute=(i64)((u64)sum+(u64)pref);
        if(term) {
            i64 row=r[3]+out+rank;
            if(field==0 && ranges && out+rank==0) ranges[record*4]=absolute;
            if(!selected || keep[row]) {
                i64 dest=selected?positions[row]:row;
                if(field==0) { if(ids) ids[dest]=absolute; }
                else {
                    double coordinate=__dadd_rn(__dmul_rn((double)absolute,(double)m[2]),(double)m[field==1?3:4])*1e-9;
                    if(coordinate < (field==1?-90.0:-180.0) || coordinate > (field==1?90.0:180.0)) atomicExch(error,1);
                    if(field==1) y[dest]=coordinate; else x[dest]=coordinate;
                }
            }
        }
        sum=(i64)((u64)sum+(u64)__shfl_sync(0xffffffff,pref,31)); out+=__popc(mask);
    }
    if(lane==0 && out!=r[4]) atomicExch(error,1);
    // Record extrema from ID stream without making coordinates host-visible.
    if(field==0 && ranges && lane==0) { ranges[record*4+1]=sum; ranges[record*4+2]=r[3]; ranges[record*4+3]=r[4]; }
}
'''

_PBF_NAMES = (
    "pbf_count_blocks", "pbf_index_blocks", "pbf_parse_ways", "pbf_dense_tags",
    "pbf_decode_dense",
)

_PBF_SOURCE += r'''
extern "C" __global__ void __launch_bounds__(128)
pbf_validate_inflate(const unsigned char* __restrict__ data,
                     const i64* __restrict__ offsets,const i64* __restrict__ checksums,
                     int* __restrict__ error,int n) {
    int b=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31;
    if(b>=n || checksums[b]<0) return;
    i64 length=offsets[b+1]-offsets[b]; u64 a=0,s=0;
    for(i64 i=lane;i<length;i+=32) {
        u64 v=data[offsets[b]+i]; a+=v; s+=(u64)(length-i)*v;
    }
    for(int step=16;step;step>>=1) { a+=__shfl_down_sync(0xffffffff,a,step); s+=__shfl_down_sync(0xffffffff,s,step); }
    if(lane==0) {
        u64 checksum=(((s+length)%65521)<<16) | ((a+1)%65521);
        if(checksum!=(u64)checksums[b]) atomicExch(error,1);
    }
}
'''
_PBF_NAMES += ("pbf_validate_inflate",)

_PBF_SOURCE += r'''
extern "C" __global__ void __launch_bounds__(128)
pbf_emit_refs(const unsigned char* __restrict__ data,const i64* __restrict__ ways,
              const int* __restrict__ keep,const i64* __restrict__ rows,
              const i64* __restrict__ coord_offsets,i64* __restrict__ x,
              int* __restrict__ geometry_offsets,int n,i64 row_base,i64 coord_base) {
    int i=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31;
    if(i>=n || !keep[i]) return;
    const i64* r=ways+i*W; i64 start=r[4],end=r[5],sum=0,out=0;
    i64 dest=coord_base+coord_offsets[i];
    if(lane==0) geometry_offsets[row_base+rows[i]]=dest;
    for(i64 base=start;base<end;base+=32) {
        i64 p=base+lane; bool term=p<end && !(data[p]&128); i64 val=0;
        if(term) {
            i64 begin=p; while(begin>start && (data[begin-1]&128) && p-begin<10) --begin;
            u64 raw=0; int shift=0;
            for(i64 j=begin;j<=p;++j,shift+=7) raw|=(u64)(data[j]&127)<<shift;
            val=zz(raw);
        }
        unsigned mask=__ballot_sync(0xffffffff,term); int rank=__popc(mask&((1u<<lane)-1));
        for(int step=1;step<32;step<<=1) { i64 v=__shfl_up_sync(0xffffffff,val,step); if(lane>=step) val=(i64)((u64)val+(u64)v); }
        if(term) x[dest+out+rank]=(i64)((u64)sum+(u64)val);
        sum=(i64)((u64)sum+(u64)__shfl_sync(0xffffffff,val,31)); out+=__popc(mask);
    }
}
extern "C" __global__ void __launch_bounds__(128)
pbf_link_refs(i64* __restrict__ x,i64* __restrict__ y,
              const i64* __restrict__ first,const i64* __restrict__ last,
              i64* __restrict__ heads,int* __restrict__ error,i64 n,int nranges,int shards) {
    for(i64 i=(i64)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(i64)blockDim.x*gridDim.x) {
        i64 target=x[i]; int lo=0,hi=nranges;
        while(lo<hi) { int mid=lo+((hi-lo)>>1); if(last[mid]<target) lo=mid+1; else hi=mid; }
        if(lo==nranges || first[lo]>target) { atomicExch(error,3); x[i]=y[i]=0x7ff8000000000000LL; }
        else y[i]=(i64)atomicExch((u64*)(heads+(i64)lo*shards+(i&(shards-1))),(u64)i);
    }
}
extern "C" __global__ void __launch_bounds__(128)
pbf_resolve_refs(const i64* __restrict__ ids,const double* __restrict__ nx,
                 const double* __restrict__ ny,const i64* __restrict__ dense,
                 const i64* __restrict__ heads,i64* __restrict__ x,i64* __restrict__ y,
                 int* __restrict__ error,int n,int shards,const i64* __restrict__ range_ids) {
    int work=blockIdx.x*blockDim.x+threadIdx.x;
    if(work>=n*shards) return;
    int record=work/shards,shard=work&(shards-1);
    const i64* r=dense+record*D; i64 p=heads[range_ids[record]*shards+shard];
    while(p>=0) {
        i64 next=y[p],target=x[p],lo=r[3],hi=r[3]+r[4];
        while(lo<hi) { i64 mid=lo+((hi-lo)>>1); if(ids[mid]<target) lo=mid+1; else hi=mid; }
        if(lo<r[3]+r[4] && ids[lo]==target) {
            x[p]=__double_as_longlong(floor(__dadd_rn(__dmul_rn(nx[lo],1e7),0.5))/1e7);
            y[p]=__double_as_longlong(floor(__dadd_rn(__dmul_rn(ny[lo],1e7),0.5))/1e7);
        } else { atomicExch(error,3); x[p]=y[p]=0x7ff8000000000000LL; }
        p=next;
    }
}
'''
_PBF_NAMES += ("pbf_emit_refs", "pbf_link_refs", "pbf_resolve_refs")

_PBF_SOURCE += _PBF_ATTRIBUTE_SOURCE
_PBF_NAMES += ("pbf_attributes",)

_PBF_SOURCE += _PBF_RELATION_SOURCE
_PBF_NAMES += _PBF_RELATION_NAMES

_PBF_SOURCE += _PBF_RING_SOURCE
_PBF_NAMES += _PBF_RING_NAMES
