// CS 264 HW 4

#include "mpi.h"
#include <cstdlib>

#include "def.h"
#include <stdio.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>

// -- Function declarations
void LoadBMPFile(uchar4 **dst, 
		 int *width, int *height, 
		 const char *name);

void SaveBMPFile(uchar4 *dst, 
		 int width, int height, 
		 const char *outputname, const char *inputname);

void kernel_2d_convolution_cpu(uchar4 *_in, uchar4 *_out, 
			       int width, int height, 
			       int halfKernelSize);

void kernel_2d_convolution_gpu(uchar4 *_in, uchar4 *_out, 
			       int width, int height, int halfKernelSize);

// we'll use MPI id=0 for the master
const int iMasterProc = 0;
const int paramMsg = 123;
const int inputMsg = 456;
const int outputMsg = 789;

// =============================================================================
// == MPI Master routine
// =============================================================================
void master(uint nProcs,
	    const char* inputFilename, 
	    const char* outputFilename, 
	    int halfKernelSize, 
	    int nTest, 
	    bool checkAccuracy)
{
  const char* msgPrefix = "[MASTER] >>>";

  // -- Load input BMP file
  int imgWidth, imgHeight;
  uchar4 *h_src;

  LoadBMPFile(&h_src, &imgWidth, &imgHeight, inputFilename);
  printf("%s BMP file loaded successfully!\n", msgPrefix);

  // -- Allocate host memory (not pinned)
  ulong imgDataSize = imgWidth*imgHeight*sizeof(uchar4);
  uchar4* h_cpuout = (uchar4*)malloc(imgDataSize);
  uchar4* h_mpigpuout = (uchar4*)malloc(imgDataSize); 

  // -- If needed, run CPU implementation (i.e. ground truth)
  if(checkAccuracy) 
    {
      printf("%s Running CPU (reference) implementation...\n", msgPrefix);
      kernel_2d_convolution_cpu(h_src, h_cpuout, 
				imgWidth, imgHeight, 
				halfKernelSize);
    }

  // -- Initialize the timer
  float totalTime = 0;
  float maxErr = 0;
  unsigned int timer = 0;
  printf("%s Timer starts...\n", msgPrefix);
  cutilCheckError(  cutCreateTimer(&timer)  );

  // ---------------------------------------------------------------------------
  // -- Split the problem 
  // ---------------------------------------------------------------------------

  // TODO: Split the problem
  int tileHeight = (imgHeight + nProcs - 1) / nProcs;
  int tileWidth = imgWidth;
  int masterStartY = tileHeight * (nProcs - 1);
  int masterStartYOverlap = max(0, masterStartY - halfKernelSize);
  int masterEndY = imgHeight;
  int masterEndYOverlap = imgHeight;
  ulong * offsetSend = (ulong *)malloc(sizeof(ulong) * nProcs);
  ulong * offsetSendSize = (ulong *)malloc(sizeof(ulong) * nProcs);
  ulong * offsetRecv = (ulong *)malloc(sizeof(ulong) * nProcs);
  offsetRecv[iMasterProc] = masterStartY * tileWidth;
  offsetSend[iMasterProc] = masterStartYOverlap * tileWidth;
  offsetSendSize[iMasterProc] = masterEndY * tileWidth - offsetSend[iMasterProc];
  
  // ---------------------------------------------------------------------------
  // -- Initialize the workers
  // ---------------------------------------------------------------------------
  int paramBuffer[6];
  paramBuffer[0] = tileWidth;
  paramBuffer[1] = tileHeight;
  paramBuffer[2] = halfKernelSize;
  paramBuffer[3] = nTest;
  
  // TODO: Initialize the workers (if needed)
  for (uint proc = 1; proc < nProcs; proc++) {
    int procStartY = (proc - 1) * tileHeight;
    int procStartYOverlap = max(0, procStartY - halfKernelSize);
    int procEndY = procStartY + tileHeight;
    int procEndYOverlap = min(imgHeight, procEndY + halfKernelSize);
    offsetRecv[proc] = procStartY * tileWidth;
    offsetSend[proc] = procStartYOverlap * tileWidth;
    offsetSendSize[proc] = procEndYOverlap * tileWidth - offsetSend[proc];
    paramBuffer[4] = procStartY - procStartYOverlap;
    paramBuffer[5] = procEndYOverlap - procEndY;
    
    MPI::COMM_WORLD.Send(paramBuffer, 6, MPI::INT, proc, paramMsg);
  
    
  }

  // ---------------------------------------------------------------------------
  // -- Process data
  // ---------------------------------------------------------------------------
  for(int iter=0; iter<nTest; iter++)
    {
      printf("%s Running GPU implementation...\n", msgPrefix);
      // -- Reset timer
      cutilCheckError( cutResetTimer(timer) );
      cutilCheckError( cutStartTimer(timer) );
      
      for (uint proc = 1; proc < nProcs; proc++) {
        MPI::COMM_WORLD.Send(&(h_src[offsetSend[proc]]),
            offsetSendSize[proc]*sizeof(uchar4), MPI::BYTE, proc, inputMsg);
      }
    
      kernel_2d_convolution_gpu(&(h_src[offsetSend[iMasterProc]]),
                &(h_mpigpuout[offsetSend[iMasterProc]]), 
				tileWidth, masterEndYOverlap-masterStartYOverlap, 
				halfKernelSize);
				
	  for (uint proc = 1; proc < nProcs; proc++) {
        MPI::COMM_WORLD.Recv(&(h_mpigpuout[offsetRecv[proc]]),
            tileHeight*tileWidth*sizeof(uchar4), MPI::BYTE, proc, outputMsg);
      }

      // -- Accumulate time
      cutilCheckError( cutStopTimer(timer) );
      printf("[SUMMARY] %.3f msec\n", cutGetTimerValue(timer));     
      totalTime += cutGetTimerValue(timer);

      // -- Check accuracy against CPU ground truth
      if(checkAccuracy)
	{
	  printf("%s Checking accuracy...\n", msgPrefix);
	  
	  float err = 0;
	  for(int idx=0; idx<imgWidth*imgHeight; idx++)
	    {	      	      
	      float curerr= sqrt( (float)( ( (float)h_mpigpuout[idx].x -  (float)h_cpuout[idx].x)*( (float)h_mpigpuout[idx].x -  (float)h_cpuout[idx].x) +
					   ( (float)h_mpigpuout[idx].y -  (float)h_cpuout[idx].y)*( (float)h_mpigpuout[idx].y -  (float)h_cpuout[idx].y) +
					   ( (float)h_mpigpuout[idx].z -  (float)h_cpuout[idx].z)*( (float)h_mpigpuout[idx].z -  (float)h_cpuout[idx].z) +
					   ( (float)h_mpigpuout[idx].w -  (float)h_cpuout[idx].w)*( (float)h_mpigpuout[idx].w -  (float)h_cpuout[idx].w) ) );

	      err += curerr;			    
	    }
	  maxErr = max(maxErr, err/(float)(imgWidth*imgHeight));
	}
    }	           

  // -- 
  printf("[SUMMARY] nProcs: %d, Filter size: %d x %d, Total average computing time: %.3f msec",
         nProcs,
         2*halfKernelSize+1, 2*halfKernelSize+1,
         totalTime/(float)nTest);

  if(checkAccuracy) 
    printf(", Error : %f\n",  maxErr);
  else 
    printf("\n");

  // -- Save output to BMP file	
  SaveBMPFile(h_mpigpuout, imgWidth, imgHeight, outputFilename, inputFilename);
	
  // -- Clean up
  printf("%s Cleaning up... \n", msgPrefix);

  free(h_src);
  free(h_cpuout);
  free(h_mpigpuout);

  // TODO: Clean up (if needed)
  free(offsetSend);
  free(offsetSendSize);
  free(offsetRecv);

}

// =============================================================================
// == MPI Worker routine
// =============================================================================
void worker(int iMyProc)
{
    (void)iMyProc;
  //char msgPrefix[20];
  //sprintf(msgPrefix, "[WORKER %d] >>>", iMyProc);
  
  //get values
  int paramBuffer[6];
  //printf("%s Waiting for parameters.\n", msgPrefix);
  MPI::COMM_WORLD.Recv(paramBuffer, 6, MPI::INT, iMasterProc, paramMsg);
  //printf("%s Recieved parameters.\n", msgPrefix);
  int tileWidth = paramBuffer[0];
  int tileHeight = paramBuffer[1];
  int halfKernelSize = paramBuffer[2];
  int nTest = paramBuffer[3];
  int topOverlap = paramBuffer[4];
  int bottomOverlap = paramBuffer[5];
  int inputHeight = topOverlap+tileHeight+bottomOverlap;
  
  //initialize buffers to recieve data
  ulong inputSize = sizeof(uchar4)*
        (inputHeight)*tileWidth;
  ulong outputSize = sizeof(uchar4)*tileWidth*tileHeight;
  ulong outputOffset = topOverlap*tileWidth;
  uchar4 * _in = (uchar4 *)malloc(inputSize);
  uchar4 * _out = (uchar4 *)malloc(inputSize);
  
  for(int iter=0; iter<nTest; iter++)
    {
      //printf("%s Beginning iteration, waiting for data.\n", msgPrefix);
      MPI::COMM_WORLD.Recv(_in,
            inputSize, MPI::BYTE, iMasterProc, inputMsg);
    
      kernel_2d_convolution_gpu(_in,
                _out, 
				tileWidth, inputHeight, 
				halfKernelSize);
				
      MPI::COMM_WORLD.Send(&(_out[outputOffset]),
            outputSize, MPI::BYTE, iMasterProc, outputMsg);
      //printf("%s Finished iteration.\n", msgPrefix);
    }
}

// =============================================================================
// == Main routine
// =============================================================================
int main(int argc, char *argv[])
{

  // --
  MPI::Init(argc, argv);

  const int iMyProc = MPI::COMM_WORLD.Get_rank();

  // -- Do we have enough command line arguments ?
  if(argc < 6)
    {
      if(iMyProc == iMasterProc)
	printf("Usage : %s input.bmp output.bmp halffiltersize numtest checkAccuracy\n", argv[0]);
      exit(1);
    }
    
  // -- How many processes do we have ?
  const int nProcs = MPI::COMM_WORLD.Get_size();
  if(iMyProc == iMasterProc)
    printf("%d MPI processes launched\n", nProcs);

  // -- Interpret command line arguments
  char* inputFilename = argv[1];
  char* outputFilename = argv[2];
  int halfKernelSize = atoi(argv[3]);
  int nTest = atoi(argv[4]);
  bool checkAccuracy = atoi(argv[5]) != 0;  
  cudaFree(NULL);  

  // -- Separate the master from the workers depending on their MPI id
  if(iMyProc == iMasterProc)
    {
      master(nProcs, 
	     inputFilename, outputFilename, 
	     halfKernelSize, nTest, checkAccuracy);
    }
  else
    {
      worker(iMyProc);
    }
  
  MPI::Finalize();
  return 0;
}

// EOF
