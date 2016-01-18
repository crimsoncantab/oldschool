#include "Image.h"

Image::Image(int width, int height)
    :w(width), h(height)
{
    w = w < 1 ? 1 : w;
    h = h < 1 ? 1 : h;
    
    for(int i = 0; i < GetMaxIndex(); i++)
    {
        pixels.push_back(Color());
    }
}

void Image::SetPixel(int x, int y, const Color& color)
{
    if(x < 0 || x >= w || y < 0 || y >= h)
    {
        return;
    }
    
    pixels[y * w + x] = color;
}

void Image::SetPixel(int index, const Color& color)
{
    if(index < 0 || index >= GetMaxIndex())
    {
        return;
    }
    
    pixels[index] = color;
}

void Image::OutputBitmap(string filename)
{
    char header[14];
    char info[40];
    int size = 3 * w * h + 54;
    
    ofstream image(filename.c_str(), ios::out | ios::binary);
    if(!image.is_open())
    {
        // error
        return;
    }
    
    // create and write header
    memset(header, 0, 14);
    header[0] = 'B';
    header[1] = 'M';
    header[2] = size;
    header[3] = size >> 8;
    header[4] = size >> 16;
    header[5] = size >> 24;
    header[10] = 54;
    image.write(header, 14);
    
    // create and write info
    memset(info, 0, 40);
    info[0] = 40;
    info[4] = w;
    info[5] = w >> 8;
    info[6] = w >> 16;
    info[7] = w >> 24;
    info[8] = h;
    info[9] = h >> 8;
    info[10] = h >> 16;
    info[11] = h >> 24;
    info[12] = 1;
    info[14] = 24;
    image.write(info, 40);
    
    // write out pixels
    for(unsigned int i = 0; i < pixels.size(); i++)
    {
        image << pixels[i];
    }
    
    // flush and close file
    image.close();
}
