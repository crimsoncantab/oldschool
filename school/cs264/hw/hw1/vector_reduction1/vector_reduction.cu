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

#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, project
#include <cutil_inline.h>

// includes, kernels
#include <vector_reduction_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

int ReadFile(float*, char* file_name, int num_elements);
float computeOnDevice(float* h_data, int array_mem_size);

extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
    runTest( argc, argv);
    return EXIT_SUCCESS;
}

void randomize(float * h_data, int num_elements) {
    // initialize the input data on the host to be integer values
    // between 0 and 1000
    for( unsigned int i = 0; i < num_elements; ++i) 
    {
        srand(time(NULL));
        h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run naive scan test
////////////////////////////////////////////////////////////////////////////////
void runTest( int argc, char** argv) 
{
    int num_elements = NUM_ELEMENTS;
    unsigned int array_mem_size;
    float* h_data;
    int errorM = 0;
    int num;


    // * No arguments: Randomly generate input data and compare against the 
    //   host's result, using default NUM_ELEMENTS
    // * One argument: Randomly generate arg1 input elements
    // * Two arguments: first is num elements, second is file containing vals
    switch(argc-1)
    {      
        case 2: //Two arguments
            num = atoi(argv[1]);
            if (num != 0) {
                num_elements = num;
                array_mem_size = sizeof( float) * num_elements;
                h_data = (float*) malloc( array_mem_size);           
                errorM = ReadFile(h_data, argv[2], num_elements);
                if(errorM != 1)
                {
                    printf("Error reading input file!\n");
                    exit(1);
                }
            }
            else {
                printf("first arg must be a non-zero number");
                exit(-1);
            }
        
        case 1:  // One Argument
            num = atoi(argv[1]);
            
            if (num != 0) {
                num_elements = num;
            }
            
            array_mem_size = sizeof( float) * num_elements;
            h_data = (float*) malloc( array_mem_size);
            randomize(h_data, num_elements);
            
            break;  
        default:  // No Arguments or one argument
        
            array_mem_size = sizeof( float) * num_elements;
            h_data = (float*) malloc( array_mem_size);
            randomize(h_data, num_elements);
            
            break;  
    }
    // compute reference solution
    float reference = 0.0f;  
    computeGold(&reference , h_data, num_elements);

    // **===-------- Modify the body of this function -----------===**
    float result = computeOnDevice(h_data, num_elements);
    // **===-----------------------------------------------------------===**


    // We can use an epsilon of 0 since values are integral and in a range 
    // that can be exactly represented
    float epsilon = 0.0f;
    unsigned int result_regtest = (abs(result - reference) <= epsilon);
    printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    printf( "device: %f  host: %f\n", result, reference);
    // cleanup memory
    free( h_data);
}


int ReadFile(float* M, char* file_name, int num_elements)
{
    unsigned int elements_read = num_elements;
    if (cutReadFilef(file_name, &M, &elements_read, true))
        return 1;
    else
        return 0;
}

// **===----------------- Modify this function ---------------------===**
// Take h_data from host, copy it to device, setup grid and thread 
// dimensions, execute kernel , and copy result back to h_data.
// Note: float* h_data is both the input and the output of this function.
float computeOnDevice(float* h_data, int n)
{
    float result = 0.0;

    // =====================================================
    // Code segment 1:

    // * Allocate CUDA device memory
    float * d_data;
    cudaMalloc(&d_data, n * sizeof(float));
	CUT_CHECK_ERROR("initial alloc");

    // * Copy input data from host memory to CUDA device memory
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);
	CUT_CHECK_ERROR("copy host to device");

    int offset = 1;
    int n_temp = n;
    
    while (n_temp != 1 /*last value is the answer */) {
        //divide-round-up for last block holding remainder
        int gridD = (n_temp + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // * Setup block and grid sizes
        dim3 blockDim(BLOCK_SIZE);
        dim3 gridDim(gridD);

        printf("%d, %d, %d\n", gridDim.x, n_temp, offset);
        // * Invoke the kernel
        reduction<<<gridDim, blockDim>>>(d_data, n_temp, offset);
        CUT_CHECK_ERROR("kernel run\n");

        cudaThreadSynchronize();
        cudaMemcpy(&result, d_data, sizeof(float), cudaMemcpyDeviceToHost);
        printf("result %f\n",result);
        //we have one value per block now
        n_temp = gridD;
        //outputs of last run in progressively more distant locations
        offset *= BLOCK_SIZE;
        
    }

    // * Copy results from CUDA device memory back to host memory
    cudaMemcpy(&result, d_data, sizeof(float), cudaMemcpyDeviceToHost);

    // * Free device memory
    cudaFree(d_data);


    return result;
}

