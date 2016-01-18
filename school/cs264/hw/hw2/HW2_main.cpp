#include "def.h"
#include <stdio.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>

void LoadBMPFile(uchar4 **dst, int *width, int *height, const char *name);
void SaveBMPFile(uchar4 *dst, int width, int height, const char *outputname, const char *inputname);
void kernel_2d_convolution_cpu(uchar4 *_in, uchar4 *_out, int width, int height, int4 halfkernelsize, float id, float cd=1);
void kernel_2d_convolution_gpu(uchar4 *_in, uchar4 *_out, int width, int height, int4 halfkernelsize, float id, float cd=1);


int main(int argc, char **argv)
{
  int width, height;
  uchar4 *h_src, *h_cpuout, *h_gpuout;
    
  if(argc < 8 || argc > 11)
    {
      printf("Usage : bilateral.exe input.bmp outputfilename filterup [filterright] [filterdown filterleft] id cd numtest checkaccuracy\n");
      exit(0);
    }

  // load input BMP file
  LoadBMPFile(&h_src, &width, &height, argv[1]);
  h_cpuout = (uchar4*)malloc(width*height*sizeof(uchar4));
  h_gpuout = (uchar4*)malloc(width*height*sizeof(uchar4)); 

  bool checkaccuracy;
  int argi=3;
  float id, cd;
  int4 halfkernelsize;
  halfkernelsize.x = atoi(argv[argi++]);
  if (argc > 8) halfkernelsize.y = atoi(argv[argi++]);
  else halfkernelsize.y = halfkernelsize.x;
  if (argc > 10) {
    halfkernelsize.z = atoi(argv[argi++]); halfkernelsize.w = atoi(argv[argi++]);
  }
  else {
    halfkernelsize.z = halfkernelsize.x; halfkernelsize.w = halfkernelsize.y;
  }
  id = atof(argv[argi++]);
  cd = atof(argv[argi++]);
  int nTest = atoi(argv[argi++]);
  if(atoi(argv[argi++]) == 0) checkaccuracy = false;
  else checkaccuracy = true;
  
  if(checkaccuracy) {
        char cacheFile[256];
        snprintf(cacheFile, 256, "%s.%d.%d.%d.%d.%f.%f.cache",
            argv[1] ,halfkernelsize.x, halfkernelsize.y, halfkernelsize.z, halfkernelsize.w, id, cd);

        FILE * cache = fopen(cacheFile,"r");

        if (cache) {
            printf ("Loading cached results from %s\n", cacheFile);
            fread(h_cpuout, sizeof(uchar4), width*height, cache);
		    fclose(cache);
        }
        else {
            printf("checking CPU\n");
            kernel_2d_convolution_cpu(h_src, h_cpuout, width, height, halfkernelsize, id, cd);
            printf ("Done..\n");
            printf ("Saving results to cache %s\n", cacheFile);
		    cache = fopen(cacheFile,"w");
		    fwrite (h_cpuout, sizeof(uchar4), width*height, cache);
		    fclose(cache);
        }
    }

  float totalTime = 0;
  float maxErr = 0;
  unsigned int timer = 0;
  printf("Timer starts...\n");
  cutilCheckError(  cutCreateTimer(&timer)  );

  for(int iter=0; iter<nTest; iter++)
    {
      cutilCheckError(  cutResetTimer(timer)    );
      cutilSafeCall( cudaThreadSynchronize() );
      cutilCheckError( cutStartTimer(timer) );

      kernel_2d_convolution_gpu(h_src, h_gpuout, width, height, halfkernelsize, id, cd);

      cutilSafeCall( cudaThreadSynchronize() );
      cutilCheckError( cutStopTimer(timer) );
     
      totalTime += cutGetTimerValue(timer);

      if(checkaccuracy)
    {
      // check sanity
      float err = 0;
      for(int idx=0; idx<width*height; idx++)
        {
          float curerr= sqrt( (float)( ( (float)h_gpuout[idx].x -  (float)h_cpuout[idx].x)*( (float)h_gpuout[idx].x -  (float)h_cpuout[idx].x) +
                       ( (float)h_gpuout[idx].y -  (float)h_cpuout[idx].y)*( (float)h_gpuout[idx].y -  (float)h_cpuout[idx].y) +
                       ( (float)h_gpuout[idx].z -  (float)h_cpuout[idx].z)*( (float)h_gpuout[idx].z -  (float)h_cpuout[idx].z) +
                       ( (float)h_gpuout[idx].w -  (float)h_cpuout[idx].w)*( (float)h_gpuout[idx].w -  (float)h_cpuout[idx].w) ) );

          err += curerr;                
        }
      maxErr = max(maxErr, err/(float)(width*height));
    }
    }               

  printf("Filter size : %d x %d, Total average computing time : %.3f msec", halfkernelsize.y+halfkernelsize.w+1,
            halfkernelsize.z + halfkernelsize.x +1, totalTime/(float)nTest);
  if(checkaccuracy) printf(", Error : %f\n",  maxErr);
  else printf("\n");

  // save output to BMP file    
  SaveBMPFile(h_gpuout, width, height, argv[2], argv[1]);
    
  free(h_src);
  free(h_cpuout);
  free(h_gpuout);

  return 0;
}
