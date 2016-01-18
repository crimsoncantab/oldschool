#ifndef _RAY_
#define _RAY_

#include "Vec3.h"

class Ray
{
 private:
    Vec3 pt, dir;
    
 public:
    Ray(const Vec3& point, const Vec3& direction);
    const Vec3& GetPoint() const;
    const Vec3& GetDir() const;
    
    // Returns the point after traveling
    // lambda distance along the ray
    const Vec3 GetVec(double lambda) const;
};

#endif //_RAY_
