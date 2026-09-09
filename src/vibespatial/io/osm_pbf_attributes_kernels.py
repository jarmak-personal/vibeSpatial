"""Device OSM tag projection for the standard GDAL points/lines schema."""
from __future__ import annotations

_PBF_ATTRIBUTE_SOURCE = r'''
__device__ int promoted(const unsigned char* d,i64 p,i64 n,int kind) {
    if(eq(d,p,n,"name")) return 0;
    if(kind>=2) {
        if(eq(d,p,n,"type")) return 1;
        if(kind==3) return -1;
        const char* keys[]={"aeroway","amenity","admin_level","barrier","boundary","building","craft","geological","historic","land_area","landuse","leisure","man_made","military","natural","office","place","shop","sport","tourism"};
        for(int j=0;j<20;++j) if(eq(d,p,n,keys[j])) return j+2;
        return -1;
    }
    if(kind==0) {
        if(eq(d,p,n,"barrier")) return 1;
        if(eq(d,p,n,"highway")) return 2;
        if(eq(d,p,n,"ref")) return 3;
        if(eq(d,p,n,"address")) return 4;
        if(eq(d,p,n,"is_in")) return 5;
        if(eq(d,p,n,"place")) return 6;
        if(eq(d,p,n,"man_made")) return 7;
    } else {
        if(eq(d,p,n,"highway")) return 1;
        if(eq(d,p,n,"waterway")) return 2;
        if(eq(d,p,n,"aerialway")) return 3;
        if(eq(d,p,n,"barrier")) return 4;
        if(eq(d,p,n,"man_made")) return 5;
        if(eq(d,p,n,"railway")) return 6;
    }
    return -1;
}
__device__ bool truth(const unsigned char* d,i64 p,i64 n) {
    return eq(d,p,n,"yes") || eq(d,p,n,"true") || eq(d,p,n,"1");
}
__device__ int highway_rank(const unsigned char* d,i64 p,i64 n) {
    if(eq(d,p,n,"minor") || eq(d,p,n,"road") || eq(d,p,n,"unclassified") || eq(d,p,n,"residential")) return 3;
    if(eq(d,p,n,"tertiary") || eq(d,p,n,"tertiary_link")) return 4;
    if(eq(d,p,n,"secondary") || eq(d,p,n,"secondary_link")) return 6;
    if(eq(d,p,n,"primary") || eq(d,p,n,"primary_link")) return 7;
    if(eq(d,p,n,"trunk") || eq(d,p,n,"trunk_link")) return 8;
    if(eq(d,p,n,"motorway") || eq(d,p,n,"motorway_link")) return 9;
    return 0;
}
__device__ int atoi_tag(const unsigned char* d,i64 p,i64 n) {
    i64 i=0; while(i<n && (d[p+i]==' ' || (d[p+i]>=9 && d[p+i]<=13))) ++i;
    bool neg=i<n && d[p+i]=='-'; if(i<n && (d[p+i]=='-' || d[p+i]=='+')) ++i;
    unsigned int v=0;
    while(i<n && d[p+i]>='0' && d[p+i]<='9') v=v*10+d[p+i++]-'0';
    return neg ? (int)(0u-v) : (int)v;
}
__device__ i64 quoted(const unsigned char* d,i64 p,i64 n,unsigned char* out,i64 at) {
    i64 begin=at; if(out) out[at]='"'; ++at;
    for(i64 i=0;i<n;++i) {
        unsigned char ch=d[p+i];
        if(ch=='"' || ch=='\\') { if(out) out[at]='\\'; ++at; }
        if(out) out[at]=ch; ++at;
    }
    if(out) out[at]='"'; ++at;
    return at-begin;
}
__device__ int decimal(i64 id,unsigned char* dest) {
    char buf[21]; int n=0; u64 value=id<0 ? 0ULL-(u64)id : (u64)id;
    do { buf[n++]='0'+value%10; value/=10; } while(value);
    int length=n+(id<0);
    if(dest) { if(id<0) *dest++='-'; while(n) *dest++=buf[--n]; }
    return length;
}
extern "C" __global__ void __launch_bounds__(128)
pbf_attributes(const unsigned char* __restrict__ data,const i64* __restrict__ meta,
               const i64* __restrict__ strings,const i64* __restrict__ ways,
               const i64* __restrict__ spans,const int* __restrict__ keep,
               const i64* __restrict__ positions,const i64* __restrict__ ids,
               i64* __restrict__ lengths,const i64* __restrict__ offsets,
               const u64* __restrict__ outputs,int* __restrict__ z_order,
               int* __restrict__ error,int n,int rows,int kind,int stride,int way_id,int ncol,int emit,i64 row_base) {
    int input=blockIdx.x*blockDim.x+threadIdx.x; if(input>=n || !keep[input]) return;
    i64 row=positions[input],global=row_base+row;
    bool point=kind==0; int idcols=kind==2?2:1;
    const i64* w=point?nullptr:ways+input*stride;
    i64 id=point?ids[row]:w[3]; int idlen=decimal(id,nullptr);
    if(ncol==idcols) {
        for(int j=0;j<idcols;++j) {
            bool valid=idcols==1 || j==way_id;
            if(!emit) lengths[j*rows+row]=valid?idlen:-1;
            else {
                i64 start=outputs[j*4+3]+offsets[j*(rows+1)+row];
                ((int*)outputs[j*4+1])[global]=start;
                if(valid) {
                    atomicOr(((unsigned int*)outputs[j*4+2])+(global>>5),1u<<(global&31));
                    decimal(id,(unsigned char*)outputs[j*4]+start);
                }
            }
        }
        return;
    }
    const i64* s=point?spans+input*3:nullptr;
    const i64* m=meta+(point?s[2]:w[2])*B;
    Cursor k{data,point?s[0]:w[7],point?s[1]:w[8],error};
    Cursor v{data,point?s[0]:w[9],point?s[1]:w[10],error};
    int nprom=ncol-idcols-1;
    i64 vp[22],vn[22]; for(int j=0;j<22;++j) { vp[j]=0; vn[j]=-1; }
    i64 extras=0; bool have_extra=false;
    int highway=0,bridge=0,tunnel=0,railway=0,level=0; unsigned seen=0;
    unsigned char* extra_out=nullptr; i64 extra_base=0;
    if(emit) {
        extra_out=(unsigned char*)outputs[(ncol-1)*4];
        extra_base=outputs[(ncol-1)*4+3]+offsets[(ncol-1)*(rows+1)+row];
    }
    int saved_tags=0;
    while(k.p<k.end) {
        u64 key=k.var(),value;
        if(point) value=k.var(); else value=v.var();
        i64 kp,kn,p,l; strref(m,strings,key,kp,kn,error); strref(m,strings,value,p,l,error);
        if(way_id) {
            if(early_ignored(data,kp,kn)) continue;
            if(saved_tags++==255) break;
        }
        int column=promoted(data,kp,kn,kind);
        if(column>=0) { vp[column]=p; vn[column]=l; }
        else if(!ignored(data,kp,kn) && !(kind>=2 && eq(data,kp,kn,"area"))) {
            if(have_extra) { if(emit) extra_out[extra_base+extras]=','; ++extras; }
            extras+=quoted(data,kp,kn,extra_out,extra_base+extras);
            if(emit) { extra_out[extra_base+extras]='='; extra_out[extra_base+extras+1]='>'; }
            extras+=2;
            extras+=quoted(data,p,l,extra_out,extra_base+extras); have_extra=true;
        }
        if(kind==1) {
            if(eq(data,kp,kn,"highway")) highway=highway_rank(data,p,l);
            if(eq(data,kp,kn,"railway")) railway=5;
            if(eq(data,kp,kn,"bridge") && !(seen&1)) { bridge=truth(data,p,l)?10:0; seen|=1; }
            if(eq(data,kp,kn,"tunnel") && !(seen&2)) { tunnel=truth(data,p,l)?-10:0; seen|=2; }
            if(eq(data,kp,kn,"layer") && !(seen&4)) { level=atoi_tag(data,p,l); seen|=4; }
        }
    }
    if(!emit) {
        for(int j=0;j<idcols;++j) lengths[j*rows+row]=(idcols==1 || j==way_id)?idlen:-1;
        for(int j=0;j<nprom;++j) lengths[(j+idcols)*rows+row]=vn[j];
        lengths[(ncol-1)*rows+row]=have_extra?extras:-1;
    } else {
        if(kind==1) z_order[global]=(int)((unsigned)highway+(unsigned)bridge+(unsigned)tunnel+(unsigned)railway+10u*(unsigned)level);
        for(int j=0;j<ncol;++j) {
            i64 len=j<idcols?((idcols==1 || j==way_id)?idlen:-1):j==ncol-1?(have_extra?extras:-1):vn[j-idcols];
            i64 start=outputs[j*4+3]+offsets[j*(rows+1)+row];
            ((int*)outputs[j*4+1])[global]=start;
            if(len>=0) {
                atomicOr(((unsigned int*)outputs[j*4+2])+(global>>5),1u<<(global&31));
                unsigned char* chars=(unsigned char*)outputs[j*4];
                if(j<idcols) decimal(id,chars+start);
                else if(j<ncol-1) for(i64 c=0;c<len;++c) chars[start+c]=data[vp[j-idcols]+c];
            }
        }
    }
}
'''
