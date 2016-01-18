#ifndef _IMAGE_
#define _IMAGE_

#include <iostream>
#include <string>
#include <vector>

#include "Color.h"

using namespace std;

class Image
{
 private:
    int w, h;
    vector<Color> pixels;
    
 public:
    Image(int width, int height);
    void SetPixel(int x, int y, const Color& color);
    void SetPixel(int index, const Color& color);
    void OutputBitmap(string filename);
    
    int GetWidth() { return w; }
    int GetHeight() { return h; }
    int GetMaxIndex() { return w * h; }
};

#endif //_IMAGE_
