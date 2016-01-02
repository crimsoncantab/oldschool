#ifndef PPM_H
#define PPM_H


#ifdef __MAC__
#	include <GLUT/glut.h>
#else
#	include <GL/glut.h>
#endif
#include <cstdio>


struct packed_pixel_t	{ unsigned char r, g, b; };


static inline void WritePPMScreenshot(const int width, const int height, const char filename[])
{
  unsigned char * image = new unsigned char[width*height*3];
  //    packed_pixel_t * const image = new packed_pixel_t[width * height];
    glReadPixels(0,0, width, height, GL_RGB, GL_UNSIGNED_BYTE, image);													// capture the image
    
    FILE * const f = std::fopen(filename, "wb");
    std::fprintf(f, "P6 %d %d 255\n", width, height);
    for (int i = 0; i < height; ++i)
    std::fwrite(&image[3*width*(height-1-i)], 3*width, 1/*sizeof(packed_pixel_t)*/, f);														// write it to disk
    std::fclose(f);
    delete [] image;
}

/*
 * Returns an array of struct packed_pixel_t (call free on it when you're done, user is responsible for deallocating it, but ppmread allocates memory for it)
 * with the pixel data from the specified file. Sets width and height.
 * Returns NULL on error (no file, invalid file, etc.)
 */ 
packed_pixel_t * ppmread(const char *filename, int *wp, int *hp);


#endif
