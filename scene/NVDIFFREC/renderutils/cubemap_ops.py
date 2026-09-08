"""Differentiable CUDA cubemap filters derived from the checked-out JGaussian implementation.

Forward kernels keep one writer per output. Diffuse reverse keeps one writer per
input. Specular reverse uses FP64 atomic accumulation so independent process
scheduling remains below the project's resume-continuity threshold.
"""

import math
import numpy as np
import jittor as jt


def _pixel_area(x, y, resolution):
    if resolution <= 1:
        return 1.0
    half = resolution // 2
    x = abs(x - half)
    y = abs(y - half)
    return ((math.atan((x + 1.0) / half) - math.atan(x / half)) *
            (math.atan((y + 1.0) / half) - math.atan(y / half)))


_AREA_CACHE = {}
_BOUNDS_CACHE = {}


def _areas(resolution):
    if resolution not in _AREA_CACHE:
        values = np.empty((resolution, resolution), dtype=np.float32)
        for y in range(resolution):
            for x in range(resolution):
                values[y, x] = _pixel_area(x, y, resolution)
        _AREA_CACHE[resolution] = jt.array(values.reshape(-1), dtype=jt.float32)
    return _AREA_CACHE[resolution]


def _ndf_ggx(alpha_sqr, cos_theta):
    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    denominator = (cos_theta * alpha_sqr - cos_theta) * cos_theta + 1.0
    return alpha_sqr / (denominator * denominator * np.pi)


CUDA_HEADER = r"""
#include <cuda_runtime.h>
#ifdef cudaMemcpy
#undef cudaMemcpy
#endif

__device__ float gang_ndf_ggx(float alpha_sqr, float cos_theta) {
    float ct=fminf(fmaxf(cos_theta,0.0f),1.0f);
    float d=(ct*alpha_sqr-ct)*ct+1.0f;
    return alpha_sqr/(d*d*3.141592653589793f);
}

__device__ void gang_cube_dir(int side,float x,float y,int n,float* d) {
    float fx=2.0f*((x+0.5f)/(float)n)-1.0f;
    float fy=2.0f*((y+0.5f)/(float)n)-1.0f;
    float rx,ry,rz;
    switch(side) {
        case 0: rx=1.0f;ry=-fy;rz=-fx;break;
        case 1: rx=-1.0f;ry=-fy;rz=fx;break;
        case 2: rx=fx;ry=1.0f;rz=fy;break;
        case 3: rx=fx;ry=-1.0f;rz=-fy;break;
        case 4: rx=fx;ry=-fy;rz=1.0f;break;
        default: rx=-fx;ry=-fy;rz=-1.0f;break;
    }
    float inv=rsqrtf(rx*rx+ry*ry+rz*rz);
    d[0]=rx*inv;d[1]=ry*inv;d[2]=rz*inv;
}

__global__ void gang_diffuse_fwd(int n,const float* cube,const float* areas,float* out) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x,total=6*n*n;
    if(tid>=total)return;
    int face=tid/(n*n),r=tid%(n*n),oy=r/n,ox=r%n;
    float normal[3];gang_cube_dir(face,(float)ox,(float)oy,n,normal);
    float cr=0.0f,cg=0.0f,cb=0.0f;
    for(int s=0;s<6;++s)for(int y=0;y<n;++y)for(int x=0;x<n;++x){
        float light[3];gang_cube_dir(s,(float)x,(float)y,n,light);
        float cosine=fminf(fmaxf(light[0]*normal[0]+light[1]*normal[1]+light[2]*normal[2],0.0f),0.999f);
        float w=cosine*areas[y*n+x]/3.141592653589793f;
        int src=(s*n*n+y*n+x)*3;
        cr+=cube[src]*w;cg+=cube[src+1]*w;cb+=cube[src+2]*w;
    }
    int dst=tid*3;out[dst]=cr;out[dst+1]=cg;out[dst+2]=cb;
}

__global__ void gang_diffuse_bwd(int n,const float* gout,const float* areas,float* gin) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x,total=6*n*n;
    if(tid>=total)return;
    int face=tid/(n*n),r=tid%(n*n),iy=r/n,ix=r%n;
    float light[3];gang_cube_dir(face,(float)ix,(float)iy,n,light);
    float area=areas[iy*n+ix],gx=0.0f,gy=0.0f,gz=0.0f;
    for(int s=0;s<6;++s)for(int y=0;y<n;++y)for(int x=0;x<n;++x){
        float normal[3];gang_cube_dir(s,(float)x,(float)y,n,normal);
        float cosine=fminf(fmaxf(light[0]*normal[0]+light[1]*normal[1]+light[2]*normal[2],0.0f),0.999f);
        float w=cosine*area/3.141592653589793f;
        int src=(s*n*n+y*n+x)*3;
        gx+=gout[src]*w;gy+=gout[src+1]*w;gz+=gout[src+2]*w;
    }
    int dst=tid*3;gin[dst]=gx;gin[dst+1]=gy;gin[dst+2]=gz;
}

__global__ void gang_spec_bounds(int n,float cutoff,float* bounds) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x,total=6*n*n;
    if(tid>=total)return;
    int face=tid/(n*n),r=tid%(n*n),py=r/n,px=r%n;
    float view[3];gang_cube_dir(face,(float)px,(float)py,n,view);
    const int tile=16;
    for(int s=0;s<6;++s){
        int minx=n-1,maxx=0,miny=n-1,maxy=0;
        for(int tx=0;tx<(n+tile-1)/tile;++tx)for(int ty=0;ty<(n+tile-1)/tile;++ty){
            int sx=tx*tile,sy=ty*tile,ex=min((tx+1)*tile,n),ey=min((ty+1)*tile,n);
            float a[3],b[3],c[3],d[3];
            gang_cube_dir(s,(float)sx,(float)sy,n,a);gang_cube_dir(s,(float)ex,(float)sy,n,b);
            gang_cube_dir(s,(float)sx,(float)ey,n,c);gang_cube_dir(s,(float)ex,(float)ey,n,d);
            float lox=fminf(fminf(a[0],b[0]),fminf(c[0],d[0])),hix=fmaxf(fmaxf(a[0],b[0]),fmaxf(c[0],d[0]));
            float loy=fminf(fminf(a[1],b[1]),fminf(c[1],d[1])),hiy=fmaxf(fmaxf(a[1],b[1]),fmaxf(c[1],d[1]));
            float loz=fminf(fminf(a[2],b[2]),fminf(c[2],d[2])),hiz=fmaxf(fmaxf(a[2],b[2]),fmaxf(c[2],d[2]));
            float maxdot=fmaxf(lox*view[0],hix*view[0])+fmaxf(loy*view[1],hiy*view[1])+fmaxf(loz*view[2],hiz*view[2]);
            if(maxdot>=cutoff)for(int y=sy;y<ey;++y)for(int x=sx;x<ex;++x){
                float light[3];gang_cube_dir(s,(float)x,(float)y,n,light);
                if(light[0]*view[0]+light[1]*view[1]+light[2]*view[2]>=cutoff){
                    minx=min(minx,x);maxx=max(maxx,x);miny=min(miny,y);maxy=max(maxy,y);
                }
            }
        }
        int dst=tid*24+s*4;
        bounds[dst]=(float)minx;bounds[dst+1]=(float)maxx;
        bounds[dst+2]=(float)miny;bounds[dst+3]=(float)maxy;
    }
}

__device__ float gang_spec_weight(int n,int x,int y,int side,const float* view,float roughness,const float* areas){
    float light[3];gang_cube_dir(side,(float)x,(float)y,n,light);
    float dotlv=light[0]*view[0]+light[1]*view[1]+light[2]*view[2];
    float hx=light[0]+view[0],hy=light[1]+view[1],hz=light[2]+view[2];
    float inv=rsqrtf(hx*hx+hy*hy+hz*hz);hx*=inv;hy*=inv;hz*=inv;
    float vdh=fmaxf(view[0]*hx+view[1]*hy+view[2]*hz,0.0f);
    float alpha=roughness*roughness;
    return dotlv*gang_ndf_ggx(alpha*alpha,vdh)*areas[y*n+x]/4.0f;
}

__global__ void gang_spec_fwd(int n,float roughness,float cutoff,const float* cube,const float* areas,const float* bounds,float* out){
    int tid=blockIdx.x*blockDim.x+threadIdx.x,total=6*n*n;if(tid>=total)return;
    int face=tid/(n*n),r=tid%(n*n),py=r/n,px=r%n;
    float view[3];gang_cube_dir(face,(float)px,(float)py,n,view);
    float cr=0.0f,cg=0.0f,cb=0.0f,wsum=0.0f;
    for(int s=0;s<6;++s){
        int b=tid*24+s*4,x0=(int)bounds[b],x1=(int)bounds[b+1],y0=(int)bounds[b+2],y1=(int)bounds[b+3];
        if(x0<=x1)for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x){
            float light[3];gang_cube_dir(s,(float)x,(float)y,n,light);
            if(light[0]*view[0]+light[1]*view[1]+light[2]*view[2]>=cutoff){
                float w=gang_spec_weight(n,x,y,s,view,roughness,areas),src=(s*n*n+y*n+x)*3;
                cr+=cube[(int)src]*w;cg+=cube[(int)src+1]*w;cb+=cube[(int)src+2]*w;wsum+=w;
            }
        }
    }
    int dst=tid*4;out[dst]=cr;out[dst+1]=cg;out[dst+2]=cb;out[dst+3]=wsum;
}

__global__ void gang_spec_bwd(int n,float roughness,float cutoff,const float* gout,const float* areas,const float* bounds,double* gin){
    int tid=blockIdx.x*blockDim.x+threadIdx.x,total=6*n*n;if(tid>=total)return;
    int face=tid/(n*n),r=tid%(n*n),py=r/n,px=r%n;
    float view[3];gang_cube_dir(face,(float)px,(float)py,n,view);
    float gr=gout[tid*4],gg=gout[tid*4+1],gb=gout[tid*4+2];
    for(int s=0;s<6;++s){
        int b=tid*24+s*4,x0=(int)bounds[b],x1=(int)bounds[b+1],y0=(int)bounds[b+2],y1=(int)bounds[b+3];
        if(x0<=x1)for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x){
            float light[3];gang_cube_dir(s,(float)x,(float)y,n,light);
            if(light[0]*view[0]+light[1]*view[1]+light[2]*view[2]>=cutoff){
                double w=(double)gang_spec_weight(n,x,y,s,view,roughness,areas);
                int dst=(s*n*n+y*n+x)*3;
                atomicAdd(gin+dst,(double)gr*w);atomicAdd(gin+dst+1,(double)gg*w);atomicAdd(gin+dst+2,(double)gb*w);
            }
        }
    }
}
"""


def _launch_size(resolution):
    total = 6 * resolution * resolution
    return total, 256, (total + 255) // 256


def _bounds(resolution, roughness, cutoff):
    key = (int(resolution), float(roughness), float(cutoff))
    if key not in _BOUNDS_CACHE:
        samples = np.cos(np.linspace(0.0, np.pi / 2.0, 1000000))
        distribution = np.cumsum(_ndf_ggx(np.float64(roughness) ** 4, samples))
        index = int(np.argmax(distribution >= distribution[-1] * cutoff))
        cos_cutoff = float(samples[index])
        total, block, grid = _launch_size(resolution)
        bounds = jt.code(
            [total * 24], "float32", [_areas(resolution)],
            data={"n": resolution, "cutoff": cos_cutoff},
            cuda_header=CUDA_HEADER,
            cuda_src=r"""
            int n=data["n"];float c=data["cutoff"];
            int total=6*n*n,block=256,grid=(total+block-1)/block;
            gang_spec_bounds<<<grid,block>>>(n,c,out_p);
            """).reshape(total, 24)
        _BOUNDS_CACHE[key] = cos_cutoff, bounds
    return _BOUNDS_CACHE[key]


class _Diffuse(jt.Function):
    def execute(self, cubemap):
        self.saved_tensors = (cubemap,)
        self.area_values = _areas(cubemap.shape[1])
        resolution = cubemap.shape[1]
        total, _, _ = _launch_size(resolution)
        output = jt.code(
            [total * 3], "float32", [cubemap.reshape(-1, 3), self.area_values],
            data={"n": resolution}, cuda_header=CUDA_HEADER,
            cuda_src=r"""
            int n=data["n"],total=6*n*n,block=256,grid=(total+block-1)/block;
            gang_diffuse_fwd<<<grid,block>>>(n,in0_p,in1_p,out_p);
            """)
        return output.reshape(6, resolution, resolution, 3)

    def grad(self, dout):
        cubemap, = self.saved_tensors
        resolution = cubemap.shape[1]
        total, _, _ = _launch_size(resolution)
        output = jt.code(
            [total * 3], "float32", [dout.reshape(-1, 3), self.area_values],
            data={"n": resolution}, cuda_header=CUDA_HEADER,
            cuda_src=r"""
            int n=data["n"],total=6*n*n,block=256,grid=(total+block-1)/block;
            gang_diffuse_bwd<<<grid,block>>>(n,in0_p,in1_p,out_p);
            """).reshape(cubemap.shape)
        self.saved_tensors = ()
        self.area_values = None
        return output


class _Specular(jt.Function):
    def execute(self, cubemap, bounds, area_values, roughness, cutoff):
        self.saved_tensors = (cubemap, bounds, area_values)
        self.roughness = float(roughness)
        self.cutoff = float(cutoff)
        resolution = cubemap.shape[1]
        total, _, _ = _launch_size(resolution)
        raw = jt.code(
            [total * 4], "float32",
            [cubemap.reshape(-1, 3), area_values, bounds.reshape(-1, 24)],
            data={"n": resolution, "roughness": self.roughness, "cutoff": self.cutoff},
            cuda_header=CUDA_HEADER,
            cuda_src=r"""
            int n=data["n"];float r=data["roughness"],c=data["cutoff"];
            int total=6*n*n,block=256,grid=(total+block-1)/block;
            gang_spec_fwd<<<grid,block>>>(n,r,c,in0_p,in1_p,in2_p,out_p);
            """)
        return raw.reshape(6, resolution, resolution, 4)

    def grad(self, dout):
        cubemap, bounds, area_values = self.saved_tensors
        resolution = cubemap.shape[1]
        total, _, _ = _launch_size(resolution)
        output64 = jt.code(
            [total * 3], "float64",
            [dout.reshape(-1, 4), area_values, bounds.reshape(-1, 24)],
            data={"n": resolution, "roughness": self.roughness, "cutoff": self.cutoff},
            cuda_header=CUDA_HEADER,
            cuda_src=r"""
            int n=data["n"];float r=data["roughness"],c=data["cutoff"];
            int total=6*n*n,block=256,grid=(total+block-1)/block;
            cudaMemset(out_p,0,(size_t)total*3*sizeof(double));
            gang_spec_bwd<<<grid,block>>>(n,r,c,in0_p,in1_p,in2_p,out_p);
            """)
        output = output64.float32().reshape(cubemap.shape)
        self.saved_tensors = ()
        return output, None, None, None, None


def diffuse_cubemap(cubemap):
    assert cubemap.shape[0] == 6 and cubemap.shape[1] == cubemap.shape[2]
    return _Diffuse.apply(cubemap)


def specular_cubemap(cubemap, roughness, cutoff=0.99):
    assert cubemap.shape[0] == 6 and cubemap.shape[1] == cubemap.shape[2]
    cos_cutoff, bounds = _bounds(cubemap.shape[1], roughness, cutoff)
    raw = _Specular.apply(cubemap, bounds, _areas(cubemap.shape[1]), float(roughness), cos_cutoff)
    return raw[..., :3] / (raw[..., 3:4] + 1e-10)


__all__ = ["diffuse_cubemap", "specular_cubemap"]
