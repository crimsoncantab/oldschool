#include "def.h"
#include <stdio.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

//
// implement following functions
//
texture<uchar4, 2> tex_in; 
cudaArray* a_in;

__global__ void do_cuda_kernel(
    uchar4 *_out, int width, int height,
    int4 halfkernelsize, float id, float cd) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if( x < width && y < height) {
      float3 _sum = make_float3(0,0,0);
      float sumWeight = 0;

      uchar4 ctrPix = tex2D(tex_in, x + 0.5f, y + 0.5f);
      for(int j= y-halfkernelsize.w; j<= y+halfkernelsize.y; j++) {       
          for(int i= x-halfkernelsize.z; i<= x+halfkernelsize.x; i++) {
              float3 curPix;
              uchar4 tempPix;
              tempPix = tex2D(tex_in, i + 0.5f, j + 0.5f);
              curPix.x = tempPix.x;
              curPix.y = tempPix.y;
              curPix.z = tempPix.z;

              // define bilateral filter kernel weights
              float ix = i-x;
              float jy = j-y;
              float imageDist = (ix*ix + jy*jy)*id;
        
              float3 curPixT = make_float3(curPix.x - ctrPix.x,
                               curPix.y - ctrPix.y, curPix.z - ctrPix.z);
              float colorDist =
                    (curPixT.x*curPixT.x +
                     curPixT.y*curPixT.y +
                     curPixT.z*curPixT.z)*cd;

              float currWeight = 
                __expf((imageDist+colorDist)*(-1));
              sumWeight += currWeight;

              _sum.x += currWeight*curPix.x; 
              _sum.y += currWeight*curPix.y;
              _sum.z += currWeight*curPix.z;
          }  
      }
      ctrPix.x = (__fdividef(_sum.x,sumWeight));
      ctrPix.y = (__fdividef(_sum.y,sumWeight));
      ctrPix.z = (__fdividef(_sum.z,sumWeight));
      unsigned int ctrIdx = y*width + x;
      _out[ctrIdx] = ctrPix;
    }
}

void init_textures(uchar4 *_in, int width, int height, int size) {
    // Allocate CUDA array in device memory 
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8,
        cudaChannelFormatKindUnsigned);
    cutilSafeCall(cudaMallocArray(&a_in, &channelDesc, width, height));
 
    // Copy to texture memory
    cutilSafeCall(cudaMemcpyToArray(a_in, 0, 0, _in,
        size, cudaMemcpyHostToDevice));


    // Set texture reference state. Dictates how underlying data is sampled.
    tex_in.addressMode[0] = cudaAddressModeClamp; 
    tex_in.addressMode[1] = cudaAddressModeClamp; 
    tex_in.filterMode     = cudaFilterModePoint; 
    tex_in.normalized     = false; 

    // Bind texture reference to underlying data on the GPU.
    cutilSafeCall(cudaBindTextureToArray(tex_in, a_in, channelDesc));
}
void kernel_2d_convolution_gpu(uchar4 *_in, uchar4 *_out, int width, int height,
    int4 halfkernelsize, float id=1, float cd=1) {
    int size = width * height * sizeof(uchar4);

    init_textures(_in, width, height, size);
    uchar4 * d_out;
    cutilSafeCall(cudaMalloc(&d_out, size));
    
    dim3 blockDim(BLOCK_DIM_X,BLOCK_DIM_Y);
    dim3 gridDim((1+(width-1)/blockDim.x),
        (1+(height-1)/blockDim.y));
    
    //precalc some values
    id = 0.5f/(id*id);
    cd = 0.5f/(cd*cd);

    // * Invoke the kernel
    do_cuda_kernel<<<gridDim, blockDim>>>(
        d_out, width, height, halfkernelsize, id, cd);
    cutilSafeCall(cudaThreadSynchronize());
    
    // * Free device memory
    cutilSafeCall(cudaFreeArray(a_in));
    cutilSafeCall(cudaMemcpy(_out, d_out, size, cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaFree(d_out));    
}
