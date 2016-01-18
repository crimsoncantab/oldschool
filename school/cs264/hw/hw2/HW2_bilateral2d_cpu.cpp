#include "def.h"
#include "vector_types.h"
#include <stdlib.h>
#include <math.h>

//
// CPU brute-force 2D convolution
// 
// sd = image space distance
// cd = color space distance
//
void kernel_2d_convolution_cpu(uchar4 *_in, uchar4 *_out, int width, int height, int4 halfkernelsize, float id, float cd)
{ 
  //int kernelDim = 2*halfkernelsize+1;	

  for(int y=0; y<height; y++)
    {
      for(int x=0; x<width; x++)
	{
	  // for each pixel
	  float _sum[3];
	  _sum[0] = _sum[1] = _sum[2] = 0;

	  float sumWeight = 0;

	  unsigned int ctrIdx = y*width + x;

	  float ctrPix[3];
	  ctrPix[0] = _in[ctrIdx].x;
	  ctrPix[1] = _in[ctrIdx].y;
	  ctrPix[2] = _in[ctrIdx].z;

	  // neighborhood of current pixel
	  int kernelStartX, kernelEndX, kernelStartY, kernelEndY;
	  kernelStartX = x-halfkernelsize.z;
	  kernelEndX   = x+halfkernelsize.x;
	  kernelStartY = y-halfkernelsize.w;
	  kernelEndY   = y+halfkernelsize.y; 
			
	  for(int j= kernelStartY; j<= kernelEndY; j++)
	    {       
	      for(int i= kernelStartX; i<= kernelEndX; i++)
		{   					
		  unsigned int idx = max(0, min(j, height-1))*width + max(0, min(i,width-1));

		  float curPix[3];
		  curPix[0] = _in[idx].x;
		  curPix[1] = _in[idx].y;
		  curPix[2] = _in[idx].z;
				
						
		  float currWeight;

		  // define bilateral filter kernel weights
		  float imageDist = sqrt( (float)((i-x)*(i-x) + (j-y)*(j-y)) );
						
		  float colorDist = sqrt( (float)( (curPix[0] - ctrPix[0])*(curPix[0] - ctrPix[0]) +
						   (curPix[1] - ctrPix[1])*(curPix[1] - ctrPix[1]) +
						   (curPix[2] - ctrPix[2])*(curPix[2] - ctrPix[2]) ) );

		  currWeight = 1.0f/(exp((imageDist/id)*(imageDist/id)*0.5)*exp((colorDist/cd)*(colorDist/cd)*0.5));
		  sumWeight += currWeight;

		  _sum[0] += currWeight*curPix[0]; 
		  _sum[1] += currWeight*curPix[1];
		  _sum[2] += currWeight*curPix[2];
		}  
	    }
			
	  _sum[0] /= sumWeight;
	  _sum[1] /= sumWeight;
	  _sum[2] /= sumWeight;
			
	  _out[ctrIdx].x = (int)(floor(_sum[0]));
	  _out[ctrIdx].y = (int)(floor(_sum[1]));
	  _out[ctrIdx].z = (int)(floor(_sum[2]));
	  _out[ctrIdx].w = _in[ctrIdx].w;
	}
    }
}
