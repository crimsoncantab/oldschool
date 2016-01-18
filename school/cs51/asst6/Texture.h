#ifndef _TEXTURE_
#define _TEXTURE_

#include "Color.h"
#include "Ray.h"
#include "Vec3.h"

class Texture
{
 private:
    Color base;
    double ambient;
    
 public:
    Texture();
    Texture(const Color& baseColor, double ambientCoeff);
    
    const Color& GetBaseColor() const { return base; }
    double GetAmbientCoeff() const { return ambient; }
};

#endif //_TEXTURE_
