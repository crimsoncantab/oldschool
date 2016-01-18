#include "def.h"
#include "vector_types.h"
#include <stdlib.h>
#include <math.h>

//
// CPU boxcar 2D convolution
// 
void kernel_2d_convolution_cpu(uchar4 *_in, 
			       uchar4 *_out, 
			       int width, 
			       int height, 
			       int halfkernelsize)
{ 
  int kernelDim = 2*halfkernelsize+1;

  //int nElts = kernelDim * kernelDim;

  for(int y=0; y<height; y++)
    {
      for(int x=0; x<width; x++)
	{
	  // for each pixel
	  int _sum[3] = {0, 0, 0};

	  unsigned int ctrIdx = y*width + x;

	  float ctrPix[3];
	  ctrPix[0] = _in[ctrIdx].x;
	  ctrPix[1] = _in[ctrIdx].y;
	  ctrPix[2] = _in[ctrIdx].z;

	  // neighborhood of current pixel
	  int kernelStartX, kernelEndX, kernelStartY, kernelEndY;
	  kernelStartX = x-halfkernelsize;
	  kernelEndX   = x+halfkernelsize;
	  kernelStartY = y-halfkernelsize;
	  kernelEndY   = y+halfkernelsize; 

	  for(int j= kernelStartY; j<= kernelEndY; j++)
	    {       
	      for(int i= kernelStartX; i<= kernelEndX; i++)
		{   					
		  unsigned int idx = max(0, min(j, height-1))*width + max(0, min(i,width-1));

		  int curPix[3];
		  curPix[0] = _in[idx].x;
		  curPix[1] = _in[idx].y;
		  curPix[2] = _in[idx].z;

		  _sum[0] += curPix[0]; 
		  _sum[1] += curPix[1];
		  _sum[2] += curPix[2];
		}  
	    }

	  _sum[0] /= kernelDim * kernelDim;
	  _sum[1] /= kernelDim * kernelDim;
	  _sum[2] /= kernelDim * kernelDim;
	    
	  _out[ctrIdx].x = _sum[0];
	  _out[ctrIdx].y = _sum[1];
	  _out[ctrIdx].z = _sum[2];
	  _out[ctrIdx].w = _in[ctrIdx].w;
	}
    }
}
