#include "def.h"
#include <stdio.h>
#include <cutil.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

texture<uchar4, 2> tex_in; 
cudaArray* a_in;

__global__ void do_cuda_kernel(
    uchar4 *_out, int width, int height,
    int halfkernelsize) {
  int kernelDim = 2*halfkernelsize+1;
  kernelDim = kernelDim * kernelDim;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if( x < width && y < height) {
	  // for each pixel
	  int3 _sum = make_int3(0, 0, 0);

      uchar4 ctrPix = tex2D(tex_in, x + 0.5f, y + 0.5f);

	  for(int j= y-halfkernelsize; j<= y+halfkernelsize; j++)
	    {       
	      for(int i= x-halfkernelsize; i<= x+halfkernelsize; i++)
		{
          uchar4 tempPix = tex2D(tex_in, i + 0.5f, j + 0.5f);

		  int3 curPix;
		  curPix.x = tempPix.x;
		  curPix.y = tempPix.y;
		  curPix.z = tempPix.z;

		  _sum.x += curPix.x; 
		  _sum.y += curPix.y;
		  _sum.z += curPix.z;
		}  
	    }

	  ctrPix.x = (__fdividef(_sum.x,kernelDim));
      ctrPix.y = (__fdividef(_sum.y,kernelDim));
      ctrPix.z = (__fdividef(_sum.z,kernelDim));
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
void kernel_2d_convolution_gpu(uchar4 *_in, uchar4 *_out, 
			       int width, int height, 
			       int halfkernelsize) {
    int size = width * height * sizeof(uchar4);

    init_textures(_in, width, height, size);
    uchar4 * d_out;
    cutilSafeCall(cudaMalloc(&d_out, size));
    
    dim3 blockDim(BLOCK_DIM_X,BLOCK_DIM_Y);
    dim3 gridDim((1+(width-1)/blockDim.x),
        (1+(height-1)/blockDim.y));

    // * Invoke the kernel
    do_cuda_kernel<<<gridDim, blockDim>>>(
        d_out, width, height, halfkernelsize);
    cutilSafeCall(cudaThreadSynchronize());
    
    // * Free device memory
    cutilSafeCall(cudaFreeArray(a_in));
    cutilSafeCall(cudaMemcpy(_out, d_out, size, cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaFree(d_out));  
}
