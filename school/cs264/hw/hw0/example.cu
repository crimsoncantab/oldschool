/* 
   
   Example CUDA code for Problem Set 0, CS 264, Harvard, Fall 2009.  

	 Takes an input string and mangles it using the current date and
	 time on the CPU and on the GPU.  Demonstrates device initialization
	 and error checking with libcutil, host<=>device memory transfers,
	 and CUDA kernel invocation.

   To compile: 
	     
        nvcc example.cu -o example -I$CUDASDK_HOME/common/inc	\
            -L$CUDASDK_HOME/lib/linux -lcutil

	 Usage: 
  
        example -string="<str>" {-device=<dev>} ,
	  
	 where <str> is the input string, and, optionally, <dev> is the device 
	 number.

	 Kevin Dale <dale@eecs.harvard.edu>
	 08.20.09
*/

#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include "cutil.h"

/* forward declarations */
__global__ void mangleGPU(char*,char*,int,int); // GPU kernel prototype
           void mangleCPU(char*,char*,int,int); // CPU prototype

/* macro to mangle an n-length char string, shared between CPU and GPU code */
#define MANGLE(instr,outstr,i,n,x) \
	((outstr)[(i)]=(((instr)[((i)+(x)+(instr)[(i)])%(n)])))

/* main driver */
int main(int argc, char** argv){

	// - initialize device
	CUT_DEVICE_INIT(argc,argv);

	// - read command-line args
	char *str;
	cutGetCmdLineArgumentstr(argc,(const char**)argv,"string",&str);
	int n=strlen(str);

	// - get the current time
	time_t now=time(0);
	char *nowstring=asctime(localtime(&now));

	// - allocate memory on the device
	char *d_str_in, *d_str_out;
	cudaMalloc((void**)&d_str_in, n*sizeof(char));
	cudaMalloc((void**)&d_str_out,n*sizeof(char));
	CUT_CHECK_ERROR("initial alloc");

	// - copy data to device
	cudaMemcpy(d_str_in,str,n*sizeof(char),cudaMemcpyHostToDevice);
	CUT_CHECK_ERROR("copy host to device");

	// - invoke the kernel
	int nblocks=1, nthreads=n;
	mangleGPU<<<nblocks,nthreads>>>(d_str_in,d_str_out,n,(int)now);
	CUT_CHECK_ERROR("kernel invocation");

	// - copy from device to main memory
	char *gpu_result=(char*)malloc((n+1)*sizeof(char));
	cudaMemcpy(gpu_result,d_str_out,n*sizeof(char),cudaMemcpyDeviceToHost); 
	CUT_CHECK_ERROR("copy device to host");

	// - invoke the equivalent CPU function
	char *cpu_result=(char*)malloc((n+1)*sizeof(char));	
	mangleCPU(str,cpu_result,n,(int)now);

	// - put null terminating character at end of each result
	gpu_result[n]=char(0);
	cpu_result[n]=char(0);

	// - report results
	printf("Current date/time: (%d) %s",now,nowstring);
	printf("Input string:      %s\n",str,n);
	printf("CPU result:        %s\n",cpu_result);
	printf("GPU result:        %s\n",gpu_result);

	// - cleanup and return
	cudaFree(d_str_in);
	cudaFree(d_str_out);
	free(gpu_result);
	free(cpu_result);

	return 0;
}

/* CUDA device kernel */
__global__ void mangleGPU(char* instr, char *outstr, int len, int x){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	MANGLE(instr,outstr,i,len,x);
}

/* CPU implementation */
void mangleCPU(char *instr, char *outstr, int len, int x){
	for(int i=0; i<len; i++)
		MANGLE(instr,outstr,i,len,x);
}
