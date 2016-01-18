/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Adapted by Nady Obeid, Xiao-Long Wu, and I-Jui Sung, UIUC */

#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_

#define NUM_ELEMENTS 512

// **===----------------- MP3 - Modify this function ---------------------===**
//! @param g_idata  input data in global memory
//                  result is expected in index 0 of g_idata
//! @param n        input number of elements to scan from input data
// **===------------------------------------------------------------------===**

#define BLOCK_SIZE 512

__global__ void reduction(float *g_data, int n, int offset)
{
    // =====================================================
    // Code segment 2:  

    // Thread's index
    // Note: this is not == correct index into g_data
    // on progressive runs, since each block is put into
    // a single index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Determine index into g_data
    int g_i = i * offset;
    
    // Local index for shared array
    int l_i = threadIdx.x;

    // Define shared memory
    __shared__ float s_data[BLOCK_SIZE]; 

    // Load the shared memory from global location
    if (i < n) {
        s_data[l_i] = g_data[g_i];
    }
    __syncthreads();

    // Do sum reduction from shared memory
    int j = BLOCK_SIZE;
    do {
        j = j/2;
        //only if we're in the first half
        //and the latter element we're adding
        //is < n (implying the former element
        //is < n as well)
        if (l_i < j && i + j < n) s_data[l_i] += s_data[l_i+j];
        __syncthreads();
    
    } while (j != 1);

    // Store result from shared memory back to global memory
    // must go in the g_i of each (threadIdx.x == 0)
    if (l_i == 0)
        g_data[g_i] = s_data[l_i];

    return;
}
#endif // #ifndef _SCAN_NAIVE_KERNEL_H_
