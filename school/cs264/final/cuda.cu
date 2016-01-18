#include <stdio.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>

#define BLOCK_DIM 512
#define GRID_DIM 16

__global__ void do_bsmooth_kernel(int sqrtn, int n, int * out, int numGrids) {
    for (int i = 0; i < numGrids; i++ ) {
        int x = (i * GRID_DIM + blockIdx.x) * blockDim.x + threadIdx.x + sqrtn;
        if (x < n) {
            
            int num = (x * x) % n;
            if (num != 0) {
                while (num % 2 == 0) {
                    num /= 2;
                }
                while (num % 3 == 0) {
                    num /= 3;
                }
                while (num % 5 == 0) {
                    num /= 5;
                }
                while (num % 7 == 0) {
                    num /= 7;
                }
                if (num == 1) {
                    out[blockIdx.x] = x;
                }
            }
        }
        __syncthreads();
        if (out[blockIdx.x] != 0) {
            break;
        }
    }
}

void kernel_bsmooth(int n) {
    int sqrtn = sqrt(n);
    int size = n - sqrtn;
    int h_out[GRID_DIM];
    int * d_out;
    memset(h_out, 0, GRID_DIM * sizeof(int));
    cutilSafeCall(cudaMalloc(&d_out, GRID_DIM * sizeof(int)));
    cutilSafeCall(cudaMemcpy(d_out, h_out, GRID_DIM * sizeof(int), cudaMemcpyHostToDevice));
    
    dim3 blockDim(BLOCK_DIM);
    int numGrids = (size + BLOCK_DIM -1) / BLOCK_DIM;
    numGrids = (numGrids + GRID_DIM - 1) / GRID_DIM;
    dim3 gridDim(GRID_DIM);

    // * Invoke the kernel
    do_bsmooth_kernel<<<gridDim, blockDim>>>(sqrtn, n, d_out, numGrids);
    cutilSafeCall(cudaThreadSynchronize());
    
    // * Free device memory
    cutilSafeCall(cudaMemcpy(h_out, d_out, GRID_DIM * sizeof(int), cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaFree(d_out));
    for (int i = 0; i < GRID_DIM; i++) {
        if (h_out[i]) {
            printf("%d ",h_out[i]);
        }
    }
    printf("\n");
}
