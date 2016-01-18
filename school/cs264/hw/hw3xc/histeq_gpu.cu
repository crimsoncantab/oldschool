#include "def.h"
#include <stdio.h>
#include <cutil.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>

#include "histogram_common.h"

#define BLOCK_DIM 512

uchar4 * d_in;
uchar4 * d_out;
uchar * d_lum;
uint * d_hist;
uint d_len;

void load_data(uchar4 * h_in, uint len) {

    d_len = len;
    cutilSafeCall(cudaMalloc(&d_in, len * sizeof(uchar4)));
    cutilSafeCall(cudaMalloc(&d_out, len * sizeof(uchar4)));
    cutilSafeCall(cudaMalloc(&d_lum, d_len * sizeof(uchar)));
    cutilSafeCall(cudaMalloc(&d_hist, HISTOGRAM256_BIN_COUNT * sizeof(uint)));
    cutilSafeCall(cudaMemcpy(d_in, h_in, len * sizeof(uchar4), cudaMemcpyHostToDevice));
}

void unload_data(uchar4 * h_out) {
    cutilSafeCall(cudaMemcpy(h_out, d_out, d_len * sizeof(uchar4), cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaFree(d_in));
    cutilSafeCall(cudaFree(d_out));
    cutilSafeCall(cudaFree(d_lum));
    cutilSafeCall(cudaFree(d_hist));
}

__global__ void calc_lum(uchar4 * data, uchar * lum, uint len) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < len) {
        uchar4 pix = data[x];
        //equation from wikipedia
        lum[x] = 0.2126 * pix.x + 0.7152 * pix.y + 0.0722 * pix.z;
    }
}

__global__ void map_hist_kernel(uchar4 * indata, uchar4 * outdata, uchar * lum, float * cdf, uint len) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < len) {
        //equation from wikipedia
        outdata[x].x =cdf[lum[x]] * (float) indata[x].x;
        outdata[x].y =cdf[lum[x]] * (float) indata[x].y;
        outdata[x].z =cdf[lum[x]] * (float) indata[x].z;
//        outdata[x].y *=cdf[lum[x]];
  //      outdata[x].z *=cdf[lum[x]];
    }
}

void get_histogram(uint * h_hist) {

    
    dim3 blockDim(BLOCK_DIM);
    dim3 gridDim(1+(d_len-1)/blockDim.x);
    calc_lum<<<gridDim, blockDim>>>(d_in, d_lum, d_len);
    cutilSafeCall(cudaThreadSynchronize());
    
    histogram256(d_hist,d_lum,d_len);
    cutilSafeCall(cudaMemcpy(h_hist, d_hist, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));
}

void map_lookup(float * cdf) {
    float * d_cdf;
    cutilSafeCall(cudaMalloc(&d_cdf, HISTOGRAM256_BIN_COUNT * sizeof(float)));
    
    cutilSafeCall(cudaMemcpy(d_cdf, cdf, HISTOGRAM256_BIN_COUNT * sizeof(float), cudaMemcpyHostToDevice));
    
    
    dim3 blockDim(BLOCK_DIM);
    dim3 gridDim(1+(d_len-1)/blockDim.x);
    map_hist_kernel<<<gridDim, blockDim>>>(d_in, d_out, d_lum, d_cdf, d_len);
    cutilSafeCall(cudaFree(d_cdf));    
}
