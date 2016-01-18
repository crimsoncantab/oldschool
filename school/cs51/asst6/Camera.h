#ifndef _CAMERA_
#define _CAMERA_

#include "Ray.h"
#include "Vec3.h"

class Camera
{
 private:
    Vec3 pos, forward, up, right;
    
 public:
    Camera();
    Camera(const Vec3& position, const Vec3& lookat);
    Ray Shoot(double x, double y) const;
};

#endif //_CAMERA_
