#ifndef _LIGHT_
#define _LIGHT_

#include "Vec3.h"

class Light
{
 private:
    Vec3 pos;
    double intensity;
    
 public:
    Light(const Vec3& position, double intensity);
    const Vec3& GetPos() const { return pos; }
    double GetIntensity() const { return intensity; }
};

#endif //_LIGHT_
