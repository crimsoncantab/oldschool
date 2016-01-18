// CS 264 HW 4

#include "mpi.h"
#include <cstdlib>

#include "def.h"
#include <stdio.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include "histogram_common.h"


// -- Function declarations
void LoadBMPFile(uchar4 **dst, 
		 int *width, int *height, 
		 const char *name);

void SaveBMPFile(uchar4 *dst, 
		 int width, int height, 
		 const char *outputname, const char *inputname);
		 

void load_data(uchar4 * h_in, uint len);

void unload_data(uchar4 * h_out);

void get_histogram(uint * hist);

void map_lookup(float * lut);




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
	    const char* outputFilename)
{
  const char* msgPrefix = "[MASTER] >>>";

  // -- Load input BMP file
  int imgWidth, imgHeight;
  uchar4 *h_src;

  LoadBMPFile(&h_src, &imgWidth, &imgHeight, inputFilename);
  printf("%s BMP file loaded successfully!\n", msgPrefix);

  // -- Allocate host memory (not pinned)
  ulong imgDataSize = imgWidth*imgHeight*sizeof(uchar4);
  uchar4* h_mpigpuout = (uchar4*)malloc(imgDataSize); 

  // -- Initialize the timer
  float totalTime = 0;
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
  int masterEndY = imgHeight;
  int masterTileHeight = masterEndY - masterStartY;
  uint masterLen = masterTileHeight * tileWidth;
  
  // ---------------------------------------------------------------------------
  // -- Initialize the workers
  // ---------------------------------------------------------------------------
  
  // TODO: Initialize the workers (if needed)
 /* for (uint proc = 1; proc < nProcs; proc++) {
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
  
    
  }*/
  
    

  // ---------------------------------------------------------------------------
  // -- Process data
  // ---------------------------------------------------------------------------
    printf("%s Running GPU implementation...\n", msgPrefix);
    // -- Reset timer
    cutilCheckError( cutResetTimer(timer) );
    cutilCheckError( cutStartTimer(timer) );
    
    
    //TODO
    load_data(h_src, masterLen);
    uint histogram[HISTOGRAM256_BIN_COUNT];
    float cdf[HISTOGRAM256_BIN_COUNT];
    
    get_histogram(histogram);
    
    float step = 0;
    for (int i = 0; i < HISTOGRAM256_BIN_COUNT; i++) {
        step += histogram[i];
    }
    step /= (HISTOGRAM256_BIN_COUNT-1);
    
    uint cum = 0;
    for (int i = 0; i < HISTOGRAM256_BIN_COUNT; i++) {
        cdf[i] = cum / step;
        cum += histogram[i];
    }
    
    map_lookup(cdf);
    
    unload_data(h_mpigpuout);
    
    // -- Accumulate time
    cutilCheckError( cutStopTimer(timer) );     
    totalTime += cutGetTimerValue(timer);
           

  // --
//  uint sum = 0;
  for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
        printf("%d:%.3f, ",i*16 + j,cdf[i*16 + j]);
        //sum += histogram[i*16 + j];
      }
      printf("\n");
  }
      //printf("sum: %d\n", sum);
  printf("[SUMMARY] nProcs: %d, Total average computing time: %.3f msec\n",
         nProcs,
         totalTime);

  // -- Save output to BMP file	
  SaveBMPFile(h_mpigpuout, imgWidth, imgHeight, outputFilename, inputFilename);
	
  // -- Clean up
  //printf("%s Cleaning up... \n", msgPrefix);

  free(h_src);
  free(h_mpigpuout);

  // TODO: Clean up (if needed)

}

// =============================================================================
// == MPI Worker routine
// =============================================================================
void worker(int iMyProc)
{
    (void)iMyProc;
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
  if(argc < 2)
    {
      if(iMyProc == iMasterProc)
	printf("Usage : %s input.bmp output.bmp\n", argv[0]);
      exit(1);
    }
    
  // -- How many processes do we have ?
  const int nProcs = MPI::COMM_WORLD.Get_size();
  if(iMyProc == iMasterProc)
    printf("%d MPI processes launched\n", nProcs);

  // -- Interpret command line arguments
  char* inputFilename = argv[1];
  char* outputFilename = argv[2]; 
  cudaFree(NULL);  

  // -- Separate the master from the workers depending on their MPI id
  if(iMyProc == iMasterProc)
    {
      master(nProcs, 
	     inputFilename, outputFilename);
    }
  else
    {
      worker(iMyProc);
    }
  
  MPI::Finalize();
  return 0;
}

// EOF
