#ifndef _PLANE_
#define _PLANE_

#include "Ray.h"
#include "Shape.h"
#include "Texture.h"
#include "Vec3.h"

class Plane : public Shape
{
 private:
    Vec3 pt, norm;
    
 public:
    Plane();
    Plane(const Vec3& point, const Vec3& normal, const Texture& texture);
    virtual double GetIntersect(const Ray& ray);
    virtual const Vec3& GetNormal(const Vec3& pos);
};

#endif //_PLANE_
