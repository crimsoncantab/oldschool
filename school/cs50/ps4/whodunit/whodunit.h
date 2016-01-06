/****************************************************************************
 * whodunit.h
 *
 * Computer Science 50
 * Problem Set 4
 *
 ***************************************************************************/


/* Windows datatypes */

typedef unsigned char BYTE;
typedef unsigned int DWORD;
typedef unsigned int LONG;
typedef unsigned short int WORD;


/*
 * BITMAPFILEHEADER
 *
 * The BITMAPFILEHEADER structure contains information about the type, size,
 * and layout of a file that contains a DIB [device-independent bitmap].
 *
 * Adapted from http://msdn2.microsoft.com/en-us/library/ms532321.aspx.
 */

typedef struct 
{ 
  WORD    bfType; 
  DWORD   bfSize; 
  WORD    bfReserved1; 
  WORD    bfReserved2; 
  DWORD   bfOffBits; 
} __attribute__((__packed__)) 
BITMAPFILEHEADER; 


/*
 * BITMAPINFOHEADER
 *
 * The BITMAPINFOHEADER structure contains information about the 
 * dimensions and color format of a DIB [device-independent bitmap].
 *
 * Adapted from http://msdn2.microsoft.com/en-us/library/ms532290.aspx.
 */
       
typedef struct
{
  DWORD  biSize; 
  LONG   biWidth; 
  LONG   biHeight; 
  WORD   biPlanes; 
  WORD   biBitCount; 
  DWORD  biCompression; 
  DWORD  biSizeImage; 
  LONG   biXPelsPerMeter; 
  LONG   biYPelsPerMeter; 
  DWORD  biClrUsed; 
  DWORD  biClrImportant; 
} __attribute__((__packed__))
BITMAPINFOHEADER; 

